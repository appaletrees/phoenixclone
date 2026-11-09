"""Construct Piper's unchanged architecture and require a complete policy restore."""
from pathlib import Path

from hydra.utils import instantiate

from .checkpoint import load_joint96_initialization
from .piper_softprompt import validate_piper_prompt
from wam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _checkpoint_path(path):
    # Configuration stays project-relative; the existing strict loader takes
    # a resolved filesystem path. Resolve before allocating any model weights.
    if path is None or not str(path).strip():
        raise ValueError("Piper initialization requires a Phoenix weights file")
    resolved = Path(str(path)).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Piper initialization requires a Phoenix weights file")
    return str(resolved)


def instantiate_piper_model(cfg, *, model_dtype, device):
    validate_piper_prompt(cfg)
    checkpoint = _checkpoint_path(cfg.init_checkpoint)
    logger.info("Constructing Piper policy for strict Phoenix restore; no Wan DiT base weights: %s", checkpoint)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device, checkpoint_only_init=True)
    model._piper_checkpoint_source = checkpoint
    return model


def load_piper_initialization(model, path, *, expected_step=None):
    checkpoint = _checkpoint_path(path)
    if getattr(model, "_piper_checkpoint_source", None) != checkpoint:
        raise ValueError("Piper checkpoint must match the checkpoint-only construction source")
    # Only these three groups are restored by the unchanged strict Joint96
    # loader. Reject any additional trainable parameter before copying weights.
    covered = {id(parameter) for name in ("mot", "proprio_encoder", "action_proprio_encoder")
               for module in (getattr(model, name),) if module is not None
               for parameter in module.parameters()}
    uncovered = [name for name, parameter in model.named_parameters()
                 if parameter.requires_grad and id(parameter) not in covered]
    if uncovered:
        raise ValueError("Trainable parameters outside the strict Phoenix restore: " + ", ".join(uncovered))
    receipt = load_joint96_initialization(model, checkpoint, expected_step=expected_step)
    receipt.update(initialization_mode="strict_phoenix_checkpoint", wan_dit_base_loaded=False)
    model.model_paths["video_dit"] = str(path)
    return receipt
