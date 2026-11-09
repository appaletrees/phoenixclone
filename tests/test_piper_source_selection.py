"""Check source isolation, original label/RGB alignment and unchanged recipes."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from wam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from wam.datasets.piper_joint96 import source_episode_indices
from wam.datasets.piper_prepare import expand_sources, prepare
from wam.posttrain.guard import validate_scope
from wam.training.config_validation import validate_training_config
from wam.utils import misc
from wam.utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[1]
SOURCES = tuple(Path(spec["root"]).name for spec in
                json.loads((ROOT / "configs/data/piper_sources.json").read_text())["sources"])
PREPARED = str(OmegaConf.load(ROOT / "configs/task/piper_folder_posttrain.yaml").paths.piper)


def config(source=None):
    register_default_resolvers()
    overrides = ["task=piper_posttrain", f"paths.piper={PREPARED}", "output_dir=runs/test-piper-source"]
    if source is not None:
        overrides.append(f"data.source_root=datasets/{source}")
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="train", overrides=overrides)


class SourceSelectionTests(unittest.TestCase):
    def test_folder_task_requires_selection_and_preserves_model_recipe(self):
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            missing = compose(config_name="train", overrides=["task=piper_folder_posttrain"])
            with self.assertRaisesRegex(ValueError, "requires data.source_root"):
                validate_scope(missing)
            null = compose(config_name="train", overrides=["task=piper_folder_posttrain", "data.source_root=null"])
            with self.assertRaisesRegex(ValueError, "requires data.source_root"):
                validate_scope(null)
            for source in SOURCES:
                overrides = [f"data.source_root=datasets/{source}", f"paths.piper={PREPARED}",
                             "output_dir=runs/test-piper-source"]
                folder = compose(config_name="train", overrides=["task=piper_folder_posttrain", *overrides])
                original = compose(config_name="train", overrides=["task=piper_posttrain", *overrides])
                validate_scope(folder)
                validate_training_config(folder)
                actual = OmegaConf.to_container(folder, resolve=True)
                actual["joint96_posttrain"]["piper_softprompt"].pop("require_source_root")
                self.assertEqual(actual, OmegaConf.to_container(original, resolve=True))

    def test_nested_preparation_preserves_leaf_membership_and_advantage_text(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                task_names = ["fold, Advantage: negative", "fold, Advantage: positive"]
                for leaf in ("a", "b"):
                    root = Path(f"datasets/group/{leaf}")
                    (root / "meta").mkdir(parents=True)
                    info = {"codebase_version": "v2.1", "robot_type": "piper", "fps": 30,
                            "total_episodes": 1, "total_frames": 8, "chunks_size": 1000,
                            "data_path": "data/episode_{episode_index:06d}.parquet",
                            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4"}
                    (root / "meta/info.json").write_text(json.dumps(info))
                    (root / "meta/tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": name}) + "\n"
                                                                 for i, name in enumerate(task_names)))
                    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 8, "tasks": task_names}))
                    (root / "data").mkdir()
                    values = np.full((8, 14), .1 if leaf == "a" else .2, np.float32)
                    pq.write_table(pa.table({"state": values.tolist(), "actions": values.tolist(),
                        "timestamp": np.arange(8, dtype=np.float32) / 30, "frame_index": np.arange(8),
                        "task_index": [0] * 4 + [1] * 4}), root / "data/episode_000000.parquet")
                    for camera in ("main", "left", "right"):
                        video = root / f"videos/{camera}/episode_000000.mp4"
                        video.parent.mkdir(parents=True)
                        video.write_bytes(b"fixture")
                spec = {"root": "datasets/group", "recursive": True, "state_key": "state",
                        "action_key": "actions", "native_dim": 14, "joint_unit": "radians",
                        "cameras": ["main", "left", "right"]}
                Path("sources.json").write_text(json.dumps({"schema": "phoenix.piper.sources.v1", "sources": [spec]}))
                output = "resources/prepared"
                manifest = prepare("sources.json", output)
                self.assertEqual(manifest["frames"], 16)
                self.assertEqual(source_episode_indices(output, manifest, "datasets/group"), [0, 1])
                self.assertEqual(source_episode_indices(output, manifest, "datasets/group/b"), [1])
                with self.assertRaises(ValueError):
                    source_episode_indices(output, manifest, "datasets/grou")
                self.assertEqual([json.loads(line)["task"] for line in (Path(output) / "meta/tasks.jsonl").read_text().splitlines()], task_names)
                for index, leaf in enumerate(("a", "b")):
                    record = pq.read_table(Path(output) / f"data/chunk-000/episode_{index:06d}.parquet")
                    self.assertEqual(record["source_dataset_index"].to_pylist(), [index] * 8)
                    self.assertEqual(record["task_index"].to_pylist(), [0] * 4 + [1] * 4)
                    link = Path(output) / f"videos/chunk-000/observation.images.cam_high/episode_{index:06d}.mp4"
                    self.assertFalse(link.readlink().is_absolute())
                    self.assertEqual(link.resolve(), (Path(f"datasets/group/{leaf}/videos/main/episode_000000.mp4")).resolve())
                with self.assertRaises(FileNotFoundError):
                    expand_sources([{**spec, "root": "datasets/absent"}])
            finally:
                os.chdir(previous)

    def test_explicit_indices_preserve_order_and_reject_invalid_membership(self):
        reader = object.__new__(BaseLerobotDataset)
        reader.episode_selection = {"mode": "explicit", "indices": [3, 1]}
        self.assertEqual(reader._select_episode_indices(list(range(5)), "fixture"), [1, 3])
        for indices in ([], [1, 1], [5], [True], [1.0], "1"):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                reader.episode_selection = {"mode": "explicit", "indices": indices}
                reader._select_episode_indices(list(range(5)), "fixture")

    def test_mapping_supports_noncontiguous_episodes_and_rejects_corruption(self):
        manifest = {"episodes": 4, "sources": [
            {"root": "datasets/a", "episodes": 2}, {"root": "datasets/b", "episodes": 2}]}
        rows = [{"episode_index": i, "source_root": f"datasets/{'a' if i % 2 == 0 else 'b'}"}
                for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            meta = Path(directory) / "meta"
            meta.mkdir()
            mapping = meta / "source_episodes.json"
            mapping.write_text(json.dumps(rows))
            self.assertEqual(source_episode_indices(directory, manifest, "datasets/b"), [1, 3])
            with self.assertRaisesRegex(ValueError, "Unknown Piper source"):
                source_episode_indices(directory, manifest, "datasets/missing")
            for broken in (rows[:-1], [rows[0], *rows[0:3]],
                           [{**rows[0], "source_root": "datasets/b"}, *rows[1:]]):
                mapping.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    source_episode_indices(directory, manifest, "datasets/b")

    def test_train_and_diagnostics_share_selection_without_recipe_changes(self):
        original = OmegaConf.to_container(config(), resolve=True)
        for source in SOURCES:
            cfg = config(source)
            validate_scope(cfg)
            validate_training_config(cfg)
            changed = OmegaConf.to_container(cfg, resolve=True)
            self.assertEqual(changed["data"]["train"]["source_root"], f"datasets/{source}")
            self.assertEqual(changed["data"]["val"]["source_root"], f"datasets/{source}")
            expected = copy.deepcopy(original)
            changed["data"]["source_root"] = None
            for split in ("train", "val"):
                changed["data"][split]["source_root"] = None
            self.assertEqual(changed, expected)

    def test_different_diagnostic_source_is_rejected(self):
        cfg = config("connector_plugin")
        cfg.data.val.source_root = "datasets/fold_towel_mit"
        with self.assertRaisesRegex(ValueError, "same source_root"):
            validate_scope(cfg)


@unittest.skipUnless((ROOT / PREPARED / "meta/piper_interface.json").is_file(),
                     "Prepared local Piper data is not present")
class RealSourceWindowTests(unittest.TestCase):
    def test_every_loaded_row_and_real_windows_belong_to_selected_source(self):
        with tempfile.TemporaryDirectory() as directory:
            misc.register_work_dir(directory)
            combined = instantiate(config().data.val)
            context = (torch.ones(128, 4096), torch.ones(128, dtype=torch.bool))
            for source in SOURCES:
                ds = instantiate(config(source).data.val)
                raw = ds.lerobot_dataset.multi_dataset._datasets[0]
                # Inspect every loaded row, including episodes beyond the probes.
                membership = raw.hf_dataset.data.column("source_dataset_index").to_numpy()
                expected_indices = [i for i, spec in enumerate(ds.piper_interface["sources"])
                                    if f"datasets/{source}" in (spec["root"], spec.get("source_group"))]
                self.assertEqual(set(np.unique(membership)), set(expected_indices))
                self.assertEqual(len(raw), sum(ds.piper_interface["sources"][i]["frames"] for i in expected_indices))
                self.assertEqual(len(ds), len(raw) - 4 * len(raw.episodes))
                self.assertTrue(all(0 <= row["index"] < len(ds) for row in ds.fixed_validation_panel))
                for selected_pos in (0, len(raw.episodes) - 1):
                    episode = raw.episodes[selected_pos]
                    begin, end = ds.sample_episode_ranges[selected_pos]
                    for index in (begin, end - 1):
                        offset = index - begin
                        reference = combined._read_window(combined._episode_starts[episode] + offset)
                        with patch.object(ds, "_get_cached_text_context", return_value=context):
                            sample = ds[index]
                        valid = ~reference["action_is_pad"][:, None] & ~reference["action_dim_is_pad"][None, :]
                        torch.testing.assert_close(sample["action"], reference["action"].masked_fill(~valid, 0))
                        torch.testing.assert_close(sample["proprio"], reference["proprio"][:1])
                        torch.testing.assert_close(sample["video"], combined._visual_inputs(reference, 0)["video"])
                        self.assertIn(reference["instruction"], sample["prompt"])
                        self.assertEqual(int((~sample["action_is_pad"]).sum()), 5 if index == end - 1 else 32)


if __name__ == "__main__":
    unittest.main()
