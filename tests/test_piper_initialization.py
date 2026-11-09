"""Strict restore failures, actual weight copies, and the real policy architecture."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

from wam.training.checkpoint import joint96_checkpoint_metadata
from wam.training.piper_initialization import instantiate_piper_model, load_piper_initialization
from wam.utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "resources/hf/Phoenix-Pretrain/pretrain_256013.pt"


def piper_config():
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="train", overrides=["task=piper_posttrain"])


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.joint96_three_view = True
        self.mot = nn.Linear(3, 2)
        self.proprio_encoder = nn.Linear(3, 2)
        self.action_proprio_encoder = None
        self.frozen_encoder = nn.Linear(3, 2).requires_grad_(False)
        self.model_paths = {"video_dit": "SKIPPED_PRETRAIN", "joint_visual_encoder": {
            "csvae": {"model_type": "causal_semantic_vae", "format_version": 2},
            "posterior": "mean", "per_camera_input_hw": [256, 256],
            "temporal_compression": 4, "extra_latent_scaling": "none"}}

    def set_trainable_modules(self):
        self.requires_grad_(False)
        self.mot.requires_grad_(True)
        self.proprio_encoder.requires_grad_(True)

    def forward(self, x):
        return self.mot(x) + self.proprio_encoder(x)


class StrictRestoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "policy.pt"
        torch.manual_seed(1)
        self.source = TinyPolicy()
        self.payload = {"mot": self.source.mot.state_dict(),
                        "proprio_encoder": self.source.proprio_encoder.state_dict(),
                        "joint96_metadata": joint96_checkpoint_metadata(self.source), "step": 256013}
        self.save(self.payload)

    def save(self, payload):
        torch.save(payload, self.path)

    def target(self):
        torch.manual_seed(2)
        model = TinyPolicy()
        model._piper_checkpoint_source = str(self.path.resolve())
        return model

    def test_relative_path_strict_restore_and_first_optimizer_step(self):
        model = self.target()
        frozen = model.frozen_encoder.weight.detach().clone()
        self.assertFalse(torch.equal(model.mot.weight, self.source.mot.weight))
        receipt = load_piper_initialization(model, os.path.relpath(self.path))
        self.assertEqual(receipt["initialization_mode"], "strict_phoenix_checkpoint")
        self.assertFalse(receipt["wan_dit_base_loaded"])
        self.assertEqual((receipt["pretrain_step"], receipt["new_global_step"]), (256013, 0))
        for name in ("mot", "proprio_encoder"):
            for key, expected in getattr(self.source, name).state_dict().items():
                torch.testing.assert_close(getattr(model, name).state_dict()[key], expected, rtol=0, atol=0)
        torch.testing.assert_close(model.frozen_encoder.weight, frozen, rtol=0, atol=0)
        x = torch.tensor([[1., 2., 3.], [-1., 0., 1.]])
        for policy in (self.source, model):
            optimizer = torch.optim.AdamW([*policy.mot.parameters(), *policy.proprio_encoder.parameters()],
                                          lr=5e-5, betas=(.9, .95))
            policy(x).square().mean().backward()
            optimizer.step()
        torch.testing.assert_close(model.mot.weight, self.source.mot.weight, rtol=0, atol=0)
        torch.testing.assert_close(model.proprio_encoder.weight, self.source.proprio_encoder.weight, rtol=0, atol=0)

    def test_missing_extra_shape_and_metadata_fail_before_any_weight_copy(self):
        for kind in ("missing", "extra", "shape", "metadata"):
            payload = copy.deepcopy(self.payload)
            if kind == "missing":
                payload["mot"].pop("weight")
            elif kind == "extra":
                payload["mot"]["unexpected"] = torch.zeros(1)
            elif kind == "shape":
                payload["proprio_encoder"]["weight"] = torch.zeros(1, 1)
            else:
                payload["joint96_metadata"]["camera_order"] = ["left", "main", "right"]
            self.save(payload)
            model = self.target()
            before = copy.deepcopy(model.state_dict())
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                load_piper_initialization(model, self.path)
            for key, expected in before.items():
                torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)

    def test_extra_trainable_parameters_and_checkpoint_switch_are_rejected(self):
        model = self.target()
        model.extra = nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, "outside the strict Phoenix restore"):
            load_piper_initialization(model, self.path)
        model = self.target()
        model._piper_checkpoint_source = "another-policy.pt"
        with self.assertRaisesRegex(ValueError, "construction source"):
            load_piper_initialization(model, self.path)

    def test_missing_checkpoint_fails_before_model_allocation_and_keeps_config_relative(self):
        cfg = piper_config()
        cfg.init_checkpoint = "resources/compat/definitely-missing-policy.pt"
        before = OmegaConf.to_container(cfg, resolve=True)
        with patch("wam.training.piper_initialization.instantiate") as allocate:
            with self.assertRaises(FileNotFoundError):
                instantiate_piper_model(cfg, model_dtype=torch.bfloat16, device="cpu")
            allocate.assert_not_called()
        self.assertEqual(before, OmegaConf.to_container(cfg, resolve=True))

    def test_trainer_requires_successful_restore_before_creating_optimizer(self):
        from wam.trainer import WAMTrainer

        cfg = piper_config()
        cfg.init_checkpoint = os.path.relpath(self.path)
        cfg.output_dir = str(Path(self.directory.name) / "run")
        Path(cfg.output_dir).mkdir()
        accelerator = SimpleNamespace(
            state=SimpleNamespace(deepspeed_plugin=None), device=torch.device("cpu"),
            num_processes=1, process_index=0, is_main_process=True,
            distributed_type="NO", mixed_precision="bf16", gather=lambda value: value)

        class Dataset:
            embodiment_registry = SimpleNamespace(sha256="00" * 32)
            def __len__(self):
                return 4

        class OptimizerReached(Exception):
            pass

        model = self.target()
        def inspect_optimizer(*args, **kwargs):
            torch.testing.assert_close(model.mot.weight, self.source.mot.weight, rtol=0, atol=0)
            raise OptimizerReached()

        with patch("wam.training.soft_prompt.wrap_soft_prompt_datasets", side_effect=lambda model, train, val, cfg: (train, val)):
            with patch("torch.optim.AdamW", side_effect=inspect_optimizer) as optimizer:
                with self.assertRaises(OptimizerReached):
                    WAMTrainer(model, Dataset(), Dataset(), cfg=cfg, accelerator=accelerator)
                optimizer.assert_called_once()
            payload = copy.deepcopy(self.payload)
            payload["mot"].pop("weight")
            self.save(payload)
            with patch("torch.optim.AdamW") as optimizer:
                with self.assertRaises(ValueError):
                    WAMTrainer(self.target(), Dataset(), Dataset(), cfg=cfg, accelerator=accelerator)
                optimizer.assert_not_called()


@unittest.skipUnless(POLICY.is_file(), "Released Phoenix weights are not present")
class ReleasedPolicyTests(unittest.TestCase):
    def test_actual_factory_and_strict_loader_with_real_checkpoint_structure(self):
        # Meta tensors exercise every real policy layer and strict checkpoint
        # key/shape without allocating 12 GB or reading a GPU. Frozen encoders
        # are stand-ins; their production loaders are unchanged by this patch.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from wam.model.backbones.wan22.helpers.io import ModelConfig
        from wam.model.backbones.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

        cfg = piper_config()
        before = OmegaConf.to_container(cfg, resolve=True)
        seen_downloads = []
        def existing_common_file(config):
            seen_downloads.append(config.origin_file_pattern)
            self.assertNotIn("diffusion_pytorch_model", config.origin_file_pattern)
            config.path = "fixture-frozen-vae.safetensors"

        def cpu_scheduler(*args, **kwargs):
            # Scheduler construction computes scalar statistics using .item().
            # Keep those small, unchanged calculations on CPU in this meta test.
            with torch.device("cpu"):
                return WanContinuousFlowMatchScheduler(*args, **kwargs)

        with FakeTensorMode():
            payload = torch.load(POLICY, map_location="cpu", mmap=True, weights_only=True)
            encoder_metadata = copy.deepcopy(payload["joint96_metadata"]["model_paths"]["joint_visual_encoder"])
        del payload
        with torch.device("meta"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
            encoder = nn.Module()
            encoder.metadata = encoder_metadata
            with patch.object(ModelConfig, "download_if_necessary", existing_common_file), \
                 patch("wam.model.backbones.wan22.helpers.loader._load_registered_model", return_value=nn.Linear(1, 1)) as load_frozen, \
                 patch("wam.model.world_action_model.WanContinuousFlowMatchScheduler", side_effect=cpu_scheduler), \
                 patch("wam.model.modules.codecs.joint_visual_encoder.JointVisualEncoder.from_config", return_value=encoder):
                model = instantiate_piper_model(cfg, model_dtype=torch.bfloat16, device="meta")
                model.set_trainable_modules()
                receipt = load_piper_initialization(model, cfg.init_checkpoint)
            self.assertEqual(receipt["tensor_counts"], {"mot": 1650, "proprio_encoder": 3})
            self.assertEqual(len(model.video_expert.state_dict()), 825)
            self.assertEqual(len(model.action_expert.state_dict()), 825)
            self.assertEqual(receipt["pretrain_step"], 256013)
            self.assertEqual(load_frozen.call_args.args[1], "wan_video_vae")
            load_frozen.assert_called_once()
        self.assertEqual(seen_downloads, ["Wan2.2_VAE.safetensors"])
        self.assertEqual(before, OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    unittest.main()
