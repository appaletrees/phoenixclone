"""CPU checks for data semantics, portable preparation and reference settings."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from wam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from wam.datasets.piper_prepare import JOINT_INDICES, prepare, project_native, relative_path
from wam.posttrain.guard import validate_scope
from wam.training.config_validation import validate_training_config
from wam.training.soft_prompt import validate_soft_prompt_config
from wam.utils.config_resolvers import register_default_resolvers
from wam.utils import misc

ROOT = Path(__file__).resolve().parents[1]


class ProjectionTests(unittest.TestCase):
    def test_degree_conversion_does_not_scale_grippers(self):
        raw = np.full((2, 14), 180., np.float32)
        raw[:, 6], raw[:, 13] = .2, .8
        projected, aux = project_native(raw, 14, "degrees")
        np.testing.assert_allclose(projected[:, JOINT_INDICES], np.pi, atol=1e-6)
        np.testing.assert_array_equal(projected[:, [6, 13]], raw[:, [6, 13]])
        np.testing.assert_array_equal(raw[:, JOINT_INDICES], 180.)
        np.testing.assert_array_equal(aux, np.zeros((2, 1)))

    def test_fifteenth_coordinate_is_retained(self):
        raw = np.arange(30, dtype=np.float32).reshape(2, 15)
        projected, aux = project_native(raw, 15, "radians")
        np.testing.assert_array_equal(projected, raw[:, :14])
        np.testing.assert_array_equal(aux, raw[:, 14:])

    def test_bad_numeric_data_and_unknown_units_are_rejected(self):
        for raw, dim, unit in ((np.zeros((2, 13)), 14, "radians"),
                               (np.full((2, 14), np.nan), 14, "radians"),
                               (np.zeros((2, 14)), 14, "auto")):
            with self.assertRaises(ValueError):
                project_native(raw, dim, unit)

    def test_canonical80_scatter_is_invertible_and_masks_unused_slots(self):
        slices = [{"source_slice": a, "target_slice": b} for a, b in (
            ([0, 6], [0, 6]), ([6, 7], [16, 17]), ([7, 13], [40, 46]), ([13, 14], [56, 57]))]
        native = torch.arange(42, dtype=torch.float32).reshape(3, 14)
        canonical, padding = ConcatLeftAlign._scatter(native, 80, slices)
        self.assertEqual(canonical.shape, (3, 80))
        self.assertEqual(int((~padding).sum()), 14)
        self.assertTrue(torch.equal(ConcatLeftAlign._gather(canonical[None], slices)[0], native))
        self.assertEqual(float(canonical[:, padding].abs().sum()), 0.)

    def test_generated_paths_must_be_project_relative(self):
        for path in ("/tmp/data", "../data", "resources/../../data"):
            with self.assertRaises(ValueError):
                relative_path(path)


class PreparationTests(unittest.TestCase):
    def test_projection_links_auxiliary_and_original_records(self):
        with tempfile.TemporaryDirectory() as directory:
            # Path uses the real working directory; restore it even on failures.
            previous = Path.cwd()
            os.chdir(directory)
            try:
                root = Path("datasets/example")
                (root / "meta").mkdir(parents=True)
                info = {"codebase_version": "v2.1", "robot_type": "piper", "fps": 30,
                    "total_episodes": 1, "total_frames": 8, "chunks_size": 1000,
                    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4"}
                (root / "meta/info.json").write_text(json.dumps(info))
                (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "fold"}) + "\n")
                (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 8, "tasks": ["fold"]}) + "\n")
                raw = np.zeros((8, 15), np.float32)
                raw[:, JOINT_INDICES] = 90.
                raw[:, 6], raw[:, 13], raw[:, 14] = .25, .75, 1.
                original = pa.table({"state": raw.tolist(), "actions": raw.tolist(),
                    "timestamp": np.arange(8, dtype=np.float32) / 30, "frame_index": np.arange(8),
                    "task_index": [0] * 8})
                parquet = root / "data/chunk-000/episode_000000.parquet"
                parquet.parent.mkdir(parents=True)
                pq.write_table(original, parquet)
                before = parquet.read_bytes()
                for camera in ("image", "wrist_l", "wrist_r"):
                    p = root / f"videos/{camera}/episode_000000.mp4"
                    p.parent.mkdir(parents=True)
                    p.write_bytes(camera.encode())
                Path("configs").mkdir()
                plan = {"schema": "phoenix.piper.sources.v1", "sources": [{"root": str(root),
                    "state_key": "state", "action_key": "actions", "native_dim": 15,
                    "joint_unit": "degrees", "cameras": ["image", "wrist_l", "wrist_r"]}]}
                Path("configs/sources.json").write_text(json.dumps(plan))
                report = prepare("configs/sources.json", "resources/piper/dataset")
                self.assertEqual(report["frames"], 8)
                out = pq.read_table("resources/piper/dataset/data/chunk-000/episode_000000.parquet")
                np.testing.assert_array_equal(out["action.auxiliary"].to_pylist(), [1.] * 8)
                values = np.asarray(out["action"].to_pylist())
                np.testing.assert_allclose(values[:, JOINT_INDICES], np.pi / 2, atol=1e-6)
                np.testing.assert_allclose(values[:, [6, 13]], raw[:, [6, 13]])
                for link in Path("resources/piper/dataset/videos").rglob("*.mp4"):
                    self.assertTrue(link.is_file())
                    self.assertFalse(link.readlink().is_absolute())
                self.assertEqual(before, parquet.read_bytes())
                from wam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset
                loaded = LeRobotDataset(repo_id="piper-fixture", root="resources/piper/dataset",
                    load_episode_stats=False, check_local_files=False, video_backend="pyav")
                self.assertEqual(len(loaded), 8)
                with self.assertRaises(FileExistsError):
                    prepare("configs/sources.json", "resources/piper/dataset")
            finally:
                os.chdir(previous)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        register_default_resolvers()
        self.environment = patch.dict(os.environ, {
            "WAM_PRETRAIN_CHECKPOINT": "resources/hf/Phoenix-Pretrain/pretrain_256013.pt",
            "CSVAE_CHECKPOINT": "resources/encoders/csvae.pt",
            "DINO_ENCODER_CHECKPOINT": "resources/encoders/dino110k.pt",
            "EMBODIMENT_REGISTRY": "configs/embodiment_registry/local_robot37_arm18_capacity28_v1.json"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            self.cfg = compose(config_name="train", overrides=["task=piper_posttrain"])
            self.reference = compose(config_name="train", overrides=["task=robotwin_posttrain"])

    def test_model_and_all_non_interface_training_settings_are_unchanged(self):
        own = OmegaConf.to_container(self.cfg, resolve=True)
        reference = OmegaConf.to_container(self.reference, resolve=True)
        self.assertEqual(own["model"], reference["model"])
        for config in (own, reference):
            for key in ("data", "paths", "init_checkpoint", "joint96_posttrain"):
                config.pop(key)
        self.assertEqual(own, reference)

    def test_piper_and_reference_recipe_validators_pass(self):
        for cfg in (self.cfg, self.reference):
            validate_scope(cfg)
            validate_training_config(cfg)
            validate_soft_prompt_config(cfg)

    def test_reference_reader_cannot_masquerade_as_piper(self):
        self.cfg.data.train._target_ = "wam.datasets.robotwin_joint96_robot37.RobotwinJoint96OnlineDataset"
        with self.assertRaises(ValueError):
            validate_scope(self.cfg)

    def test_piper_cannot_inherit_benchmark_episode_selection(self):
        self.cfg.data.train.episode_selection = {"mode": "periodic_prefix", "period": 550, "keep_first": 50}
        with self.assertRaises(ValueError):
            validate_scope(self.cfg)


@unittest.skipUnless((ROOT / "resources/piper/dataset/meta/piper_interface.json").is_file(),
                     "Prepared local Piper data is not present")
class RealWindowTests(unittest.TestCase):
    def test_real_head_tail_windows_inverse_actions_and_prompt_batch(self):
        from wam.datasets.joint96_posttrain import decode_joint96_actions
        from wam.training.piper_softprompt import PiperPromptDataset

        register_default_resolvers()
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=["task=piper_posttrain"])
        registry = validate_soft_prompt_config(cfg)
        with tempfile.TemporaryDirectory() as work_dir:
            misc.register_work_dir(work_dir)
            ds = instantiate(cfg.data.val)
            # Only text is mocked in memory. All numeric and RGB data are real;
            # no fake embeddings are saved into a production cache directory.
            context = (torch.ones(128, 4096), torch.ones(128, dtype=torch.bool))
            with patch.object(ds, "_get_cached_text_context", return_value=context):
                wrapped = PiperPromptDataset(ds, registry)
                head = wrapped[0]
                tail = wrapped[ds.sample_episode_ranges[0][1] - 1]
            self.assertEqual(head["video"].shape, (3, 3, 9, 256, 256))
            self.assertEqual(head["action"].shape, (32, 80))
            self.assertEqual(head["proprio"].shape, (1, 80))
            self.assertEqual(int((~tail["action_is_pad"]).sum()), 5)
            self.assertEqual(float(tail["action"][~tail["action_mask"]].abs().sum()), 0.)
            restored = decode_joint96_actions(ds.lerobot_dataset.processor, head["action"][None], head["proprio"][None])
            table = pq.read_table(ROOT / "resources/piper/dataset/data/chunk-000/episode_000000.parquet")
            np.testing.assert_allclose(restored[0].numpy(), np.asarray(table["action"][:32].to_pylist()), atol=1e-5)
            batch = wrapped.collate_fn([head, tail])
            self.assertEqual(batch["video"].shape, (2, 3, 3, 9, 256, 256))
            self.assertEqual(batch["context"].shape, (2, 128, 4096))
            self.assertTrue(torch.equal(batch["embodiment_id"], torch.tensor([14, 14])))


if __name__ == "__main__":
    unittest.main()
