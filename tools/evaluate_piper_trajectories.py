"""Offline recorded-observation evaluation; never imports a hardware transport.

Every 32 recorded frames, condition Phoenix on the current RGB/state and predict
actions at offsets 0..31, exactly as the training dataset aligns its action field.
All source episodes are training data: these metrics are not held-out success.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
HORIZON = 32
JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
IMAGE_KEYS = ("image", "wrist_image_l", "wrist_image_r")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
# The initial evaluator has identical sampling semantics at 10 steps. Accept
# only this reviewed predecessor when resuming; all other identity fields must
# still match. This lets the already-running 10-step job finish unchanged.
LEGACY_10_STEP_EVALUATOR = "55f531225a6153d69262179d143b2b40ff874b50a59b4a564381a88ef269da8a"
TASKS = {
    "clean_table": ("piper_clean_table_lerobot_v3", "Clean table", 201),
    "placemat": ("piper_set_placemat_lerobot_v3", "Set the placemat", 120),
    "fold_towel": ("fold_the_towel", "Fold the towel", 225),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chunk_starts(length):
    if length < 1:
        raise ValueError("Empty trajectory")
    return np.arange(0, length, HORIZON, dtype=np.int64)


def selected_episodes(dataset, source_root):
    sources = read_json(dataset / "meta/source_episodes.json")
    metadata = {r["episode_index"]: r for r in
                (json.loads(s) for s in (dataset / "meta/episodes.jsonl").read_text().splitlines())}
    rows = [{**metadata[r["episode_index"]], **r} for r in sources
            if Path(r["source_root"]).as_posix() == Path(source_root).as_posix()]
    rows.sort(key=lambda r: r["source_episode_index"])
    if not rows or len({r["source_episode_index"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate source episodes")
    return rows


def decode_current_frames(path, starts, timestamps):
    """Decode sequentially, retaining only current frames used for inference."""
    import av
    result, wanted = [], set(map(int, starts))
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                if frame.time is None or abs(float(frame.time) - float(timestamps[index])) > 1e-3:
                    raise ValueError(f"Video/parquet timestamp mismatch: {path}, frame {index}")
                result.append(frame.to_ndarray(format="rgb24"))
            if index >= int(starts[-1]):
                break
    if len(result) != len(starts):
        raise ValueError(f"Missing video frames: {path}, {len(result)}/{len(starts)}")
    return result


def read_episode(dataset, info, row, prompt):
    import pyarrow.parquet as pq
    index = int(row["episode_index"])
    fields = dict(episode_index=index, episode_chunk=index // info["chunks_size"])
    path = dataset / info["data_path"].format(**fields)
    table = pq.read_table(path)
    length = int(row["length"])
    if len(table) != length:
        raise ValueError(f"Incorrect episode length: {path}")
    np.testing.assert_array_equal(np.asarray(table["frame_index"]), np.arange(length))
    np.testing.assert_array_equal(np.asarray(table["episode_index"]), np.full(length, index))
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    label = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    for value in (state, label):
        if value.shape != (length, 14) or not np.isfinite(value).all():
            raise ValueError(f"Invalid native joint14 data: {path}")
    if not np.allclose(timestamps, np.arange(length) / info["fps"], atol=1e-4):
        raise ValueError(f"Nonuniform or misaligned timestamps: {path}")
    if np.any(state[:, [6, 13]] < 0) or np.any(state[:, [6, 13]] > 1):
        raise ValueError(f"Invalid measured gripper range: {path}")
    tasks = {r["task_index"]: r["task"] for r in
             (json.loads(s) for s in (dataset / "meta/tasks.jsonl").read_text().splitlines())}
    if {tasks[int(i)] for i in np.unique(np.asarray(table["task_index"]))} != {prompt}:
        raise ValueError(f"Episode/task mismatch: {path}")
    starts = chunk_starts(length)
    images = {key: decode_current_frames(
        dataset / info["video_path"].format(**fields, video_key="observation.images." + camera),
        starts, timestamps) for camera, key in zip(CAMERAS, IMAGE_KEYS)}
    return dict(state=state, label=label, timestamps=timestamps, starts=starts,
                images=images, prompt=prompt)


def replay_episode(policy, data, on_progress=None):
    """No labels or future states/images are passed into the model."""
    state, starts = data["state"], data["starts"]
    predictions = np.full(state.shape, np.nan, dtype=np.float32)
    latencies = []
    for k, start in enumerate(starts):
        observation = {"state": state[start].copy(), "prompt": data["prompt"],
                       **{key: data["images"][key][k] for key in IMAGE_KEYS}}
        result = policy.infer(observation)
        action = np.asarray(result["actions"], dtype=np.float32)
        if action.shape != (HORIZON, 14) or not np.isfinite(action).all():
            raise ValueError("Invalid predicted action chunk")
        end = min(int(start) + HORIZON, len(state))
        predictions[start:end] = action[:end - start]
        latencies.append(float(result["latency_seconds"]))
        if on_progress:
            on_progress(k + 1, len(starts))
    if not np.isfinite(predictions).all():
        raise ValueError("Uncovered prediction frames")
    return predictions, np.asarray(latencies, dtype=np.float64)


def metrics(label, prediction):
    error = prediction.astype(np.float64) - label.astype(np.float64)
    joint = error[:, JOINTS]
    return {
        "joint_mae_rad": float(np.abs(joint).mean()),
        "joint_rmse_rad": float(np.sqrt(np.square(joint).mean())),
        "joint_mae_deg": float(np.rad2deg(np.abs(joint).mean())),
        "joint_rmse_deg": float(np.rad2deg(np.sqrt(np.square(joint).mean()))),
        "per_joint_mae_rad": np.abs(joint).mean(axis=0).tolist(),
        "per_joint_rmse_rad": np.sqrt(np.square(joint).mean(axis=0)).tolist(),
        "gripper_mae": np.abs(error[:, [6, 13]]).mean(axis=0).tolist(),
        "joint_abs_error_sum": float(np.abs(joint).sum()),
        "joint_squared_error_sum": float(np.square(joint).sum()),
        "joint_value_count": int(joint.size),
    }


def render_plot(stem, data, prediction, row, audit, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(6, 2, figsize=(13.6, 13.2), sharex=True)
    fig.subplots_adjust(left=.073, right=.972, bottom=.08, top=.885, hspace=.56, wspace=.22)
    fig.text(.073, .972, f"Dual-arm Piper  |  {title}  |  episode {row['source_episode_index']}",
             fontsize=18, va="top")
    fig.text(.073, .942,
             f"Checkpoint {audit['step']:,}  |  {audit['num_inference_steps']} denoising steps  |  "
             "30 Hz  |  recorded-observation replay  |  32-step chunks", color="#626970", fontsize=10, va="top")
    black, blue = "#292d32", "#2882b4"
    fig.legend(handles=[Line2D([0], [0], color=black, lw=1.1, label="Label (collected action)"),
                        Line2D([0], [0], color=blue, lw=1.1, label="Prediction")],
               loc="upper right", bbox_to_anchor=(.972, .925), frameon=False, ncol=2)
    times = data["timestamps"]
    for joint in range(6):
        for col, (arm, offset) in enumerate((("Left", 0), ("Right", 7))):
            ax = axes[joint, col]
            dim = joint + offset
            ax.plot(times, data["label"][:, dim], color=black, lw=.95, alpha=.95)
            ax.plot(times, prediction[:, dim], color=blue, lw=.85, alpha=.88)
            ax.set_title(f"{arm} arm · joint {joint + 1}", loc="left", fontsize=10.8, pad=7)
            ax.set_ylabel("Angle (rad)", fontsize=9.5)
            ax.set_xlim(0, max(float(times[-1]), 1 / 30))
            ax.grid(True, color="#e3e7eb", linewidth=.55, alpha=.8)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.tick_params(labelsize=9, length=3)
            if joint == 5:
                ax.set_xlabel("Action time (s)")
    m = metrics(data["label"], prediction)
    fig.text(.073, .036,
             f"Joint MAE {m['joint_mae_deg']:.2f}°  |  RMSE {m['joint_rmse_deg']:.2f}°  |  "
             f"{len(times):,} recorded frames  |  prepared episode {row['episode_index']}",
             fontsize=9, color="#626970")
    fig.text(.073, .018,
             "Training-source replay; no RTC or smoothing. Labels aligned at t+0; last chunk truncated to real frames.",
             fontsize=9, color="#626970")
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=170)
    plt.close(fig)


def completed_rows(out):
    rows = []
    for path in sorted((out / "episodes").glob("episode_*.json")):
        stem = path.with_suffix("")
        if all(stem.with_suffix(ext).is_file() for ext in (".npz", ".png", ".pdf")):
            rows.append(read_json(path))
    return rows


def write_index(out, manifest):
    rows = completed_rows(out)
    fields = ["source_episode_index", "episode_index", "frames", "chunks", "joint_mae_deg",
              "joint_rmse_deg", "mean_latency_seconds", "wall_seconds"]
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    count = sum(r["joint_value_count"] for r in rows)
    summary = dict(completed_episodes=len(rows), expected_episodes=manifest["expected_episodes"],
                   evaluated_frames=sum(r["frames"] for r in rows),
                   inference_chunks=sum(r["chunks"] for r in rows),
                   complete=len(rows) == manifest["expected_episodes"])
    if count:
        summary.update(joint_mae_deg=float(np.rad2deg(sum(r["joint_abs_error_sum"] for r in rows) / count)),
                       joint_rmse_deg=float(np.rad2deg(np.sqrt(sum(r["joint_squared_error_sum"] for r in rows) / count))))
    write_json(out / "summary.json", summary)
    cards = []
    for r in rows:
        stem = f"episodes/episode_{r['source_episode_index']:06d}"
        cards.append(f'<article><h2>Episode {r["source_episode_index"]} · MAE {r["joint_mae_deg"]:.2f}° · '
                     f'RMSE {r["joint_rmse_deg"]:.2f}°</h2><p><a href="{stem}.png">PNG</a> · '
                     f'<a href="{stem}.pdf">PDF</a> · <a href="{stem}.npz">数值 NPZ</a> · '
                     f'<a href="{stem}.json">指标 JSON</a></p><a href="{stem}.png">'
                     f'<img loading="lazy" src="{stem}.png" alt="Episode {r["source_episode_index"]}"></a></article>')
    title = html.escape(manifest["title"])
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title} — offline evaluation</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#253442}}
article{{border-top:1px solid #ddd;padding-top:20px;margin-top:30px}}img{{width:100%;height:auto}}a{{color:#2375a3}}</style>
<h1>{title}：采集轨迹离线预测</h1><p>已完成 {len(rows)} / {manifest['expected_episodes']} 条。</p>
<p>黑线：采集 action 标签（带点版本的圆点为隔帧显示的采样点，连线保留全部数据）；蓝线：模型预测。
关节单位 rad。每 32 帧读取真实采集图像和 state，预测从当前帧开始的 32 步动作。
不做 RTC、平滑或时间偏移，尾段仅比较实际存在的帧。数据来自训练集，不能代表未见场景或真机成功率。</p>
<p>模型 step {manifest['model']['step']}；{manifest['model']['num_inference_steps']} 步去噪；
{html.escape(manifest['model']['inference_attention_backend'])} attention；seed {manifest['identity']['seed']}。</p>
<p><a href="metrics.csv">逐轨迹指标 CSV</a> · <a href="summary.json">汇总 JSON</a> · <a href="manifest.json">评估配置</a></p>
{''.join(cards)}</html>'''
    (out / "index.html").write_text(content)
    return summary


def verify_identity(out, identity):
    path = out / "manifest.json"
    if not path.exists():
        return None
    previous = read_json(path)
    saved = previous["identity"]
    if saved == identity:
        return previous
    before, after = dict(saved), dict(identity)
    old_code = before.pop("evaluator_sha256", None)
    after.pop("evaluator_sha256", None)
    if not (old_code == LEGACY_10_STEP_EVALUATOR and
            identity.get("inference_steps") == 10 and before == after):
        raise ValueError("Output already contains a different evaluation; use a new output directory")
    return previous


def effective_inference_steps(saved_steps, override):
    steps = int(saved_steps if override is None else override)
    if steps < 1:
        raise ValueError("inference-steps must be positive")
    return steps


def evaluate(args):
    from omegaconf import OmegaConf
    from wam.inference.piper import PiperPolicy, select_checkpoint
    root = args.project_root.resolve()
    os.chdir(root)
    dataset = args.dataset_dir.resolve()
    run_name, title, expected = TASKS[args.task]
    run_dir = root / "postcheckpoints" / run_name
    cfg = OmegaConf.load(run_dir / "config.yaml")
    steps = effective_inference_steps(cfg.eval_num_inference_steps, args.inference_steps)
    rows = selected_episodes(dataset, str(cfg.data.train.source_root))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} trajectories, found {len(rows)}")
    prompts = {task for row in rows for task in row["tasks"]}
    if len(prompts) != 1:
        raise ValueError("Selected dataset must have one task instruction")
    prompt = prompts.pop()
    checkpoint, _ = select_checkpoint(run_dir)
    info = read_json(dataset / "meta/info.json")
    if float(info["fps"]) != 30.:
        raise ValueError("This Piper evaluation requires the saved 30 Hz dataset")
    out = args.output_dir.resolve() / args.task
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(exist_ok=True)
    identity = dict(version=1, dataset=str(dataset), source_root=str(cfg.data.train.source_root),
                    checkpoint=str(checkpoint), checkpoint_size=checkpoint.stat().st_size,
                    checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
                    config_sha256=digest(run_dir / "config.yaml"),
                    normalization_sha256=digest(run_dir / "dataset_stats.json"),
                    source_map_sha256=digest(dataset / "meta/source_episodes.json"),
                    dataset_info_sha256=digest(dataset / "meta/info.json"),
                    evaluator_sha256=digest(__file__),
                    policy_sha256=digest(root / "src/wam/inference/piper.py"),
                    sampler_sha256=digest(root / "src/wam/inference/frame_joint.py"),
                    prompt=prompt, inference_steps=steps,
                    seed=args.seed, attention_backend=args.attention_backend,
                    chunk_stride=HORIZON, action_offsets="0..31", rtc=False,
                    smoothing=False, decode_future_rgb=False)
    plot_style = getattr(args, "plot_style", "plain")
    if plot_style == "markers":
        identity.update(plot_style=plot_style,
                        plot_renderer_sha256=digest(ROOT / "tools/replot_piper_trajectory_markers.py"))
    previous = verify_identity(out, identity)
    done = {r["source_episode_index"] for r in completed_rows(out)}
    pending = [r for r in rows if r["source_episode_index"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        print("EVAL_COMPLETE", json.dumps(write_index(out, read_json(out / "manifest.json"))), flush=True)
        return
    if shutil.disk_usage(out).free < 2 * 1024 ** 3:
        raise RuntimeError("At least 2 GiB free disk space is required for evaluation artifacts")
    policy = PiperPolicy(root, run_dir, prompt, device=args.device, seed=args.seed,
                         checkpoint=checkpoint, rtc=False, decode_video=False,
                         attention_backend=args.attention_backend, inference_steps=args.inference_steps)
    if int(policy.audit["num_inference_steps"]) != steps:
        raise ValueError("Loaded policy denoising steps do not match the requested evaluation")
    manifest = dict(identity=identity, title=title, expected_episodes=expected,
                    expected_frames=sum(r["length"] for r in rows), model=policy.audit,
                    evaluation="training-source recorded-observation replay; not held-out or closed-loop",
                    label="collected action", prediction="Phoenix absolute native joint14",
                    joint_indices=JOINTS.tolist(), gripper_indices=[6, 13],
                    fps=float(info["fps"]), source_episodes=rows)
    if previous:
        history = previous.get("evaluator_code_history", [previous["identity"]["evaluator_sha256"]])
        manifest["evaluator_code_history"] = list(dict.fromkeys([*history, identity["evaluator_sha256"]]))
    write_json(out / "manifest.json", manifest)
    write_index(out, manifest)
    started, heartbeat = time.monotonic(), time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as loader:
        future = loader.submit(read_episode, dataset, info, pending[0], prompt)
        for k, row in enumerate(pending):
            episode_start = time.monotonic()
            data = future.result()
            if k + 1 < len(pending):
                future = loader.submit(read_episode, dataset, info, pending[k + 1], prompt)
            def progress(n, total):
                nonlocal heartbeat
                if time.monotonic() - heartbeat >= 30:
                    status = dict(task=args.task, completed=len(done), total=expected,
                                  source_episode=row["source_episode_index"], chunk=n, episode_chunks=total,
                                  elapsed_seconds=time.monotonic() - started)
                    write_json(out / "progress.json", status)
                    print("EVAL_PROGRESS", json.dumps(status), flush=True)
                    heartbeat = time.monotonic()
            prediction, latencies = replay_episode(policy, data, progress)
            stem = out / "episodes" / f"episode_{row['source_episode_index']:06d}"
            temporary = stem.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, label=data["label"], prediction=prediction,
                                    state=data["state"], timestamps=data["timestamps"],
                                    chunk_starts=data["starts"], latency_seconds=latencies,
                                    source_episode_index=row["source_episode_index"],
                                    prepared_episode_index=row["episode_index"])
            temporary.replace(stem.with_suffix(".npz"))
            report = dict(source_episode_index=row["source_episode_index"], episode_index=row["episode_index"],
                          frames=len(data["state"]), chunks=len(data["starts"]),
                          mean_latency_seconds=float(latencies.mean()),
                          wall_seconds=time.monotonic() - episode_start,
                          **metrics(data["label"], prediction))
            write_json(stem.with_suffix(".json"), report)
            if plot_style == "markers":
                from tools.replot_piper_trajectory_markers import TASK_TITLES, render
                render((str(stem.with_suffix(".json")), TASK_TITLES.get(args.task, title),
                        policy.audit["step"], policy.audit["num_inference_steps"]))
            else:
                render_plot(stem, data, prediction, row, policy.audit, title)
            done.add(row["source_episode_index"])
            summary = write_index(out, manifest)
            print("EVAL_EPISODE_DONE", json.dumps({"task": args.task, **report,
                  "completed": len(done), "total": expected}), flush=True)
            del data, prediction
    print("EVAL_COMPLETE" if summary["complete"] else "EVAL_PARTIAL", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-dir", type=Path, default=Path("resources/piper/dataset_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/piper-dataset-eval/20261028"))
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--attention-backend", choices=["sdpa", "flex"], default="sdpa")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plot-style", choices=["plain", "markers"], default="plain",
                        help="markers uses the Chinese joint plots with collected-action sample dots")
    parser.add_argument("--inference-steps", type=int,
                        help="Inference-only denoising override; default is the saved configuration (10)")
    parser.add_argument("--limit", type=int, help="Only process this many pending episodes per task; resume is automatic")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.inference_steps is not None and args.inference_steps < 1:
        parser.error("--inference-steps must be positive")
    if args.task != "all":
        evaluate(args)
        return
    # Separate processes release the previous checkpoint before loading the next.
    out = args.output_dir if args.output_dir.is_absolute() else args.project_root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    links = ''.join(f'<li><a href="{key}/index.html">{title} ({count} trajectories)</a></li>'
                    for key, (_, title, count) in TASKS.items())
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Piper offline evaluation</title>'
        '<style>body{font:18px system-ui;margin:50px;line-height:1.7}</style>'
        '<h1>Piper 采集数据离线预测</h1><p>黑线是采集 action 标签，蓝线是预测。每条轨迹含 12 关节图、PNG/PDF 和原始数值。'
        '训练数据回放，不代表真机或未见数据的成功率。</p><ul>' + links + '</ul>')
    for task in TASKS:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--task", task]
        subprocess.run(command, check=True)
    summaries = {task: read_json(out / task / "summary.json") for task in TASKS}
    write_json(out / "summary.json", summaries)
    print("EVAL_ALL_FINISHED", json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
