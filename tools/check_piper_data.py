"""Check prepared records, real decoded RGB and Joint96 shapes without a model/GPU.

Text caches are reported separately. Missing text is not replaced with synthetic
embeddings, and a data-interface pass is not a training-readiness claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.datasets.piper_prepare import project_native, read_json, relative_path, write_json
from wam.posttrain.guard import validate_scope
from wam.training.config_validation import validate_training_config
from wam.training.soft_prompt import validate_soft_prompt_config
from wam.utils import misc
from wam.utils.config_resolvers import register_default_resolvers


def check(output="resources/piper/interface_check.json", decode=True,
          dataset="resources/piper/dataset", source_root=None):
    output = relative_path(output)
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
        overrides = ["task=piper_posttrain", f"paths.piper={relative_path(dataset)}"]
        if source_root is not None:
            overrides.append(f"data.source_root={relative_path(source_root)}")
        cfg = compose(config_name="train", overrides=overrides)
    validate_scope(cfg)
    validate_training_config(cfg)
    registry = validate_soft_prompt_config(cfg)
    root = Path(cfg.paths.piper)
    manifest = read_json(root / "meta/piper_interface.json")
    mapping = read_json(root / "meta/source_episodes.json")
    by_root = {s["root"]: s for s in manifest["sources"]}
    verified = 0
    for row in mapping:
        spec = by_root[row["source_root"]]
        source = Path(row["source_root"])
        info = read_json(source / "meta/info.json")
        ep, old_ep = row["episode_index"], row["source_episode_index"]
        original = pq.read_table(source / info["data_path"].format(
            episode_chunk=old_ep // info["chunks_size"], episode_index=old_ep))
        prepared = pq.read_table(root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")
        for old_key, key, aux_key in ((spec["state_key"], "observation.state", "observation.auxiliary"),
                                      (spec["action_key"], "action", "action.auxiliary")):
            expected, aux = project_native(original[old_key].to_pylist(), spec["native_dim"], spec["joint_unit"])
            np.testing.assert_array_equal(np.asarray(prepared[key].to_pylist(), np.float32), expected)
            np.testing.assert_array_equal(np.asarray(prepared[aux_key].to_pylist(), np.float32), aux[:, 0])
        np.testing.assert_array_equal(prepared["timestamp"].to_numpy(), original["timestamp"].to_numpy())
        verified += len(prepared)
    links = list((root / "videos").rglob("*.mp4"))
    if len(links) != 3 * len(mapping) or any(not p.is_symlink() or not p.is_file() or Path(p.readlink()).is_absolute() for p in links):
        raise ValueError("Expected three valid relative video links per episode")
    misc.register_work_dir(str(output.parent / "validation"))
    ds = instantiate(cfg.data.val)
    selected_episodes = ds.lerobot_dataset.multi_dataset._datasets[0].episodes
    positions = {episode: position for position, episode in enumerate(selected_episodes)}
    expected_dims = torch.tensor([*range(6), 16, *range(40, 46), 56])
    probes = []
    for source in by_root:
        rows = [r for r in mapping if r["source_root"] == source and r["episode_index"] in positions]
        if not rows:
            continue
        for row, tail in ((rows[0], False), (rows[-1], True)):
            ep = row["episode_index"]
            begin, end = ds.sample_episode_ranges[positions[ep]]
            index = end - 1 if tail else begin
            raw = ds._raw_index(index)
            sample = ds._read_window(raw)
            assert sample["action"].shape == (32, 80)
            assert sample["proprio"].shape == (33, 80)
            valid_dims = torch.where(~sample["action_dim_is_pad"])[0]
            assert torch.equal(valid_dims, expected_dims)
            assert torch.equal(sample["action_dim_is_pad"], sample["proprio_dim_is_pad"])
            assert torch.isfinite(sample["action"]).all() and torch.isfinite(sample["proprio"]).all()
            result = {"source": source, "episode": ep, "tail": tail,
                "action_shape": list(sample["action"].shape), "proprio_shape": [1, 80],
                "real_actions": int((~sample["action_is_pad"]).sum())}
            if decode:
                visual = ds._visual_inputs(sample, raw)
                assert visual["video"].shape == (3, 3, 9, 256, 256)
                assert not visual["camera_is_pad"].any()
                assert torch.isfinite(visual["video"]).all()
                assert -1 <= visual["video"].min() <= visual["video"].max() <= 1
                result["video_shape"] = list(visual["video"].shape)
            probes.append(result)
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
    tasks = sorted({task for row in episodes if row["episode_index"] in positions for task in row["tasks"]})
    missing = []
    for task in tasks:
        prompt = DEFAULT_PROMPT.format(task=task)
        path = Path(cfg.paths.piper_text) / (hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt")
        if not path.is_file():
            missing.append(str(path))
        else:
            context, mask = ds._get_cached_text_context(prompt)
            assert context.shape == (128, 4096) and mask.shape == (128,)
    report = {"data_interface": "passed", "episodes": len(mapping), "verified_frames": verified,
        "source_root": source_root, "selected_episodes": len(selected_episodes),
        "valid_windows": len(ds), "relative_video_links": len(links),
        "embodiment_id": 14, "embodiment_name": registry.content["embodiments"][14],
        "probes": probes, "text_cache": {"tasks": len(tasks), "missing": missing},
        "training_started": False}
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="resources/piper/interface_check.json")
    parser.add_argument("--no-decode", action="store_true", help="Check numeric records only")
    parser.add_argument("--dataset", default="resources/piper/dataset")
    parser.add_argument("--source-root", help="Select an original folder or its declared source group")
    args = parser.parse_args()
    print(json.dumps(check(args.output, not args.no_decode, args.dataset, args.source_root), ensure_ascii=False, indent=2))
