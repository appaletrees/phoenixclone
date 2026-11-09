"""Prepare a portable LeRobot view of native dual-arm Piper demonstrations.

No source file is modified. Numeric records are projected into joint14, while
the fifteenth coordinate is retained separately. Videos are relative symlinks.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = "phoenix.piper.joint14.v1"
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
JOINT_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
JOINT_NAMES = [f"{arm}.joint_{i}" for arm in ("left_arm", "right_arm") for i in range(1, 7)]
NATIVE_NAMES = [
    *JOINT_NAMES[:6], "left_arm.gripper_open",
    *JOINT_NAMES[6:], "right_arm.gripper_open",
]


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Use a project-relative path without '..': {value}")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def project_native(values, native_dim, unit):
    """Keep joint order/grippers; convert ONLY the twelve joint coordinates."""
    raw = np.asarray(values, dtype=np.float32)
    if raw.ndim != 2 or native_dim not in (14, 15) or raw.shape[1] != native_dim:
        raise ValueError(f"Expected finite [frames,{native_dim}] Piper data, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite Piper action/state")
    joints = raw[:, :14].copy()
    if unit == "degrees":
        joints[:, JOINT_INDICES] = np.deg2rad(joints[:, JOINT_INDICES])
    elif unit != "radians":
        raise ValueError(f"Declare degrees or radians explicitly, got {unit!r}")
    aux = raw[:, 14:15].copy() if native_dim == 15 else np.zeros((len(raw), 1), np.float32)
    return joints, aux


def feature_schema():
    from datasets import Features, Sequence, Value

    return Features({
        "observation.state": Sequence(Value("float32"), length=14),
        "action": Sequence(Value("float32"), length=14),
        "observation.auxiliary": Value("float32"),
        "action.auxiliary": Value("float32"),
        "auxiliary_present": Value("bool"),
        "source_dataset_index": Value("int64"),
        "source_episode_index": Value("int64"),
        "timestamp": Value("float32"),
        "frame_index": Value("int64"),
        "episode_index": Value("int64"),
        "index": Value("int64"),
        "task_index": Value("int64"),
    })


def numeric_stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": x.mean(axis=0).tolist(), "std": x.std(axis=0).tolist(),
        "min": x.min(axis=0).tolist(), "max": x.max(axis=0).tolist(),
        "q01": np.quantile(x, .01, axis=0).tolist(),
        "q99": np.quantile(x, .99, axis=0).tolist(), "count": [len(x)],
    }


def expand_sources(specs):
    """Keep a selectable top-level folder while locating its LeRobot datasets."""
    expanded = []
    for spec in specs:
        group = relative_path(spec["root"])
        if spec.get("recursive", False):
            roots = sorted(path.parent.parent for path in group.rglob("meta/info.json"))
            if not roots:
                raise FileNotFoundError(f"No LeRobot meta/info.json below {group}")
        else:
            roots = [group]
        for root in roots:
            expanded.append({**{k: v for k, v in spec.items() if k != "recursive"},
                             "root": str(root), "source_group": str(group)})
    return expanded


def prepare(sources_path="configs/data/piper_sources.json", output="resources/piper/dataset"):
    sources_path, output = relative_path(sources_path), relative_path(output)
    plan = read_json(sources_path)
    if plan.get("schema") != "phoenix.piper.sources.v1" or not plan.get("sources"):
        raise ValueError("Expected a nonempty phoenix.piper.sources.v1 plan")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite prepared data: {output}")
    specs = expand_sources(plan["sources"])
    roots = [relative_path(s["root"]) for s in specs]
    if len(set(roots)) != len(roots):
        raise ValueError("Repeated source roots would duplicate training examples")
    source_paths = roots + [relative_path(s["root"]) for s in plan["sources"]]
    if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in source_paths):
        raise ValueError("Prepared data and original source roots must be separate")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    episodes, source_map, all_states, all_actions, per_episode_stats = [], [], [], [], []
    tasks, source_summaries, frame_cursor = {}, [], 0
    features = feature_schema()
    fps = None
    try:
        for source_index, (spec, root) in enumerate(zip(specs, roots)):
            info = read_json(root / "meta/info.json")
            if info["codebase_version"] != "v2.1" or info["robot_type"] != "piper":
                raise ValueError(f"Expected a native Piper LeRobot v2.1 source: {root}")
            source_fps = float(info["fps"])
            if not np.isfinite(source_fps) or source_fps <= 0 or (fps is not None and source_fps != fps):
                raise ValueError("All sources must have the same positive FPS; no implicit resampling")
            fps = source_fps
            cameras = spec["cameras"]
            if len(cameras) != 3 or len(set(cameras)) != 3:
                raise ValueError("Declare cameras in main/left/right order")
            task_lookup = {int(t["task_index"]): t["task"] for t in (
                json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line
            )}
            source_episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
            if len(source_episodes) != info["total_episodes"]:
                raise ValueError(f"Episode count mismatch: {root}")
            source_frames, max_abs_joint, aux_values = 0, 0., set()
            for ep in source_episodes:
                old_index, new_index = int(ep["episode_index"]), len(episodes)
                old_chunk = old_index // int(info["chunks_size"])
                source_file = root / info["data_path"].format(episode_chunk=old_chunk, episode_index=old_index)
                table = pq.read_table(source_file)
                n = len(table)
                if n != ep["length"] or n < 5:
                    raise ValueError(f"Invalid episode length: {source_file}")
                state_raw = np.asarray(table[spec["state_key"]].to_pylist(), np.float32)
                action_raw = np.asarray(table[spec["action_key"]].to_pylist(), np.float32)
                state, state_aux = project_native(state_raw, spec["native_dim"], spec["joint_unit"])
                action, action_aux = project_native(action_raw, spec["native_dim"], spec["joint_unit"])
                max_abs_joint = max(max_abs_joint, float(np.abs(state_raw[:, JOINT_INDICES]).max()), float(np.abs(action_raw[:, JOINT_INDICES]).max()))
                aux_values.update(float(v) for v in np.unique(action_aux))
                timestamps = np.asarray(table["timestamp"].to_pylist(), np.float32)
                frame_indices = np.asarray(table["frame_index"].to_pylist(), np.int64)
                if not np.array_equal(frame_indices, np.arange(n)) or not np.allclose(timestamps, np.arange(n) / fps, atol=1e-4, rtol=0):
                    raise ValueError(f"Noncontiguous frame/timestamp alignment: {source_file}")
                task_ids = []
                episode_tasks = []
                for old_task in table["task_index"].to_pylist():
                    task = task_lookup[int(old_task)]
                    task_ids.append(tasks.setdefault(task, len(tasks)))
                    if task not in episode_tasks:
                        episode_tasks.append(task)
                fields = {
                    "observation.state": state.tolist(), "action": action.tolist(),
                    "observation.auxiliary": state_aux[:, 0], "action.auxiliary": action_aux[:, 0],
                    "auxiliary_present": [spec["native_dim"] == 15] * n,
                    "source_dataset_index": [source_index] * n, "source_episode_index": [old_index] * n,
                    "timestamp": timestamps, "frame_index": frame_indices,
                    "episode_index": [new_index] * n, "index": np.arange(frame_cursor, frame_cursor + n),
                    "task_index": task_ids,
                }
                data_file = staging / f"data/chunk-{new_index // 1000:03d}/episode_{new_index:06d}.parquet"
                data_file.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pydict(fields, schema=features.arrow_schema), data_file)
                for old_camera, camera in zip(cameras, CAMERAS):
                    source_video = root / info["video_path"].format(episode_chunk=old_chunk, episode_index=old_index, video_key=old_camera)
                    if not source_video.is_file():
                        raise FileNotFoundError(source_video)
                    dest = staging / f"videos/chunk-{new_index // 1000:03d}/observation.images.{camera}/episode_{new_index:06d}.mp4"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.symlink_to(os.path.relpath(source_video.resolve(), dest.parent.resolve()))
                episodes.append({"episode_index": new_index, "tasks": episode_tasks, "length": n})
                source_map.append({"episode_index": new_index, "source_root": str(root), "source_episode_index": old_index})
                per_episode_stats.append({"episode_index": new_index, "stats": {
                    "observation.state": numeric_stats(state), "action": numeric_stats(action),
                }})
                all_states.append(state)
                all_actions.append(action)
                frame_cursor += n
                source_frames += n
            if source_frames != info["total_frames"]:
                raise ValueError(f"Frame count mismatch: {root}")
            source_summaries.append({**spec, "episodes": len(source_episodes), "frames": source_frames,
                "raw_joint_max_abs": max_abs_joint, "action_auxiliary_values": sorted(aux_values)})
            print(f"Prepared {root}: {len(source_episodes)} episodes, {source_frames} frames", flush=True)
        if not episodes:
            raise ValueError("No episodes")
        metadata_features = {name: {"dtype": desc.dtype if hasattr(desc, "dtype") else "float32", "shape": [1], "names": None} for name, desc in features.items()}
        for name in ("observation.state", "action"):
            metadata_features[name] = {"dtype": "float32", "shape": [14], "names": NATIVE_NAMES}
        for name in ("observation.auxiliary", "action.auxiliary"):
            metadata_features[name] = {"dtype": "float32", "shape": [1], "names": ["source_auxiliary"]}
        for camera in CAMERAS:
            metadata_features[f"observation.images.{camera}"] = {"dtype": "video", "shape": [3, 480, 640],
                "names": ["channels", "height", "width"], "info": {"video.fps": fps, "video.height": 480,
                "video.width": 640, "video.channels": 3, "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                "has_audio": False}}
        write_json(staging / "meta/info.json", {
            "codebase_version": "v2.1", "robot_type": "piper", "fps": fps,
            "total_episodes": len(episodes), "total_frames": frame_cursor, "total_tasks": len(tasks),
            "total_videos": 3 * len(episodes), "total_chunks": (len(episodes) + 999) // 1000,
            "chunks_size": 1000, "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": metadata_features,
        })
        for name, rows in (("episodes.jsonl", episodes), ("episodes_stats.jsonl", per_episode_stats),
                           ("tasks.jsonl", [{"task_index": index, "task": task} for task, index in tasks.items()])):
            (staging / "meta" / name).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
        stats = {"observation.state": numeric_stats(np.concatenate(all_states)), "action": numeric_stats(np.concatenate(all_actions))}
        write_json(staging / "meta/stats.json", stats)
        normalization = {kind: {"default": {f"global_{k}": v for k, v in stats[key].items() if k != "count"}}
                         for kind, key in (("state", "observation.state"), ("action", "action"))}
        write_json(staging / "meta/normalization.json", normalization)
        manifest = {"schema": SCHEMA, "sources_plan": str(sources_path), "sources": source_summaries,
            "episodes": len(episodes), "frames": frame_cursor, "fps": fps, "joint_unit": "radians",
            "gripper_unit": "source_open_fraction_unchanged", "action_alignment": "same_source_frame_unchanged",
            "native_names": NATIVE_NAMES, "camera_order": list(CAMERAS),
            "auxiliary_policy": "Preserved separately with auxiliary_present; not a robot joint or a canonical80 action target.",
            "unit_evidence": "Source units inferred from raw joint magnitudes; degree/radian decisions are explicit in sources_plan.",
            "video_storage": "relative_symlinks_to_originals", "normalization_membership": "all_training_episodes",
        }
        write_json(staging / "meta/piper_interface.json", manifest)
        write_json(staging / "meta/source_episodes.json", source_map)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise
