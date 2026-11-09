"""Piper I/O for Joint96 sampling and optional inference-time RTC guidance."""
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from eval.robotwin.joint96_robot37_policy import current_rgb, VALID_DIMS
from eval.robotwin.policy import RobotWinPolicySession, _static_dimension_is_pad
from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from wam.datasets.robotwin_joint96_io import official_text_padding
from wam.inference.frame_joint import infer_frame_joint_window
from wam.posttrain.guard import install_import_guard, validate_scope
from wam.training.checkpoint import _joint96_identity, _load_joint96_checkpoint
from wam.training.piper_initialization import instantiate_piper_model
from wam.utils.config_resolvers import register_default_resolvers

IMAGE_KEYS = ("image", "wrist_image_l", "wrist_image_r")


def validate_observation(observation):
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape not in ((14,), (15,)) or not np.isfinite(state).all():
        raise ValueError("Expected finite native joint14 state (optional auxiliary 15th field)")
    if np.any(state[[6, 13]] < 0) or np.any(state[[6, 13]] > 1):
        raise ValueError("Measured grippers must be in [0,1]")
    prompt = observation["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A raw task instruction is required")
    result = {"state": state[:14].copy(), "prompt": prompt}
    for key in IMAGE_KEYS:
        image = np.asarray(observation[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB uint8: {key}")
        result[key] = np.ascontiguousarray(image)
    return result


def select_checkpoint(run_dir, checkpoint=None):
    if checkpoint is not None:
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        # Explicit selection uses the step in the payload, not a potentially
        # stale manifest entry. Metadata and module restoration stay strict.
        return path, None
    root = Path(run_dir) / "checkpoints/weights"
    manifest = json.loads((root / "weights_manifest.json").read_text())
    latest = manifest["latest"]
    path = root / latest["path"]
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Invalid weights manifest path")
    return path, int(latest["step"])


def available_checkpoints(root, run_dir):
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    cfg = load_config(root, run_dir)
    manifest = run_dir / "checkpoints/weights/weights_manifest.json"
    latest = None
    if manifest.is_file():
        latest = (manifest.parent / json.loads(manifest.read_text())["latest"]["path"]).resolve()
    result = []
    for path in sorted((run_dir / "checkpoints/weights").glob("*.pt")):
        row = {"path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
               "manifest_default": path.resolve() == latest}
        try:
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            row["step"] = payload.get("step")
            validate_checkpoint(cfg, payload, None)
            row["piper_compatible"] = True
        except Exception as exc:
            row.update(piper_compatible=False, error=str(exc))
        result.append(row)
    return result


def load_config(root, run_dir):
    register_default_resolvers()
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    validate_scope(cfg)
    # Only relocate resources. Architecture, encoders, attention and schedulers
    # are taken from the saved training configuration.
    codec = cfg.model.joint_visual_codec
    for obj, key in ((codec, "csvae_checkpoint_path"),
                     (codec.dino_kwargs, "checkpoint_path"),
                     (cfg.local_robot_softprompt, "registry_path"),
                     (cfg.model.action_dit_config, "embodiment_registry_path")):
        path = Path(str(obj[key]))
        obj[key] = str(path if path.is_absolute() else Path(root) / path)
    cfg.init_checkpoint = str(Path(root) / cfg.init_checkpoint)
    return cfg


def validate_checkpoint(cfg, payload, expected_step):
    validate_scope(cfg)
    identity = _joint96_identity(payload.get("joint96_metadata"))
    contract = payload["joint96_metadata"].get("soft_prompt", {}).get("training_contract", {})
    required = {"domain": "piper", "embodiment_id": 14,
                "embodiment_name": "openneo_aloha_piper",
                "action_space": "absolute_joint14_radians_to_canonical80", "cached_visuals": False}
    if any(contract.get(k) != v for k, v in required.items()):
        raise ValueError("Checkpoint is not the trained Piper slot14 policy")
    if type(payload.get("step")) is not int or payload["step"] <= 0:
        raise ValueError("Expected a completed Piper optimizer checkpoint with a positive step")
    if expected_step is not None and payload["step"] != expected_step:
        raise ValueError("Checkpoint step does not match the selected weights manifest")
    if int(cfg.model.tokenizer_max_len) != 128:
        raise ValueError("Piper requires the saved 128-token T5 convention")
    return identity


def text_cache_path(root, instruction):
    prompt = DEFAULT_PROMPT.format(task=instruction)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    return Path(root) / "resources/piper/text_cache" / (digest + ".t5_len128.wan22ti2v5b.pt")


@torch.no_grad()
def prepare_text_cache(root, instruction, device="cuda:0"):
    """Encode actual local UMT5 weights once, before allocating the policy."""
    path = text_cache_path(root, instruction)
    if path.is_file():
        return path
    from safetensors.torch import load_file
    from wam.model.backbones.wan22.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
    from wam.model.world_action_model import WAM
    weights = Path(root) / "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
    tokenizer_root = Path(root) / "checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    print("PIPER_T5_CACHE_BUILD", str(path), flush=True)
    # Meta construction avoids a second, unused FP32 copy of the XXL encoder.
    with torch.device("meta"):
        encoder = WanTextEncoder()
    weights_dict = load_file(str(weights), device="cpu")
    encoder.load_state_dict(weights_dict, strict=True, assign=True)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(str(tokenizer_root), seq_len=128, clean="whitespace", local_files_only=True)
    prompt = DEFAULT_PROMPT.format(task=instruction)
    # Invoke the original prompt encoder, including zero padding and all-visible
    # attention keys; no substitute/synthetic text embeddings.
    context, mask = WAM.encode_prompt(SimpleNamespace(text_encoder=encoder, tokenizer=tokenizer, device=device), prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"context": context[0].cpu(), "mask": mask[0].cpu(), "prompt": prompt}, temporary)
    temporary.replace(path)
    del context, mask, encoder, tokenizer, weights_dict
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return path


class PiperPolicy:
    def __init__(self, root, run_dir, instruction, *, device="cuda:0", seed=0, rtc=False,
                 inference_steps=None, checkpoint=None, decode_video=None,
                 attention_backend=None):
        if decode_video is not None and not isinstance(decode_video, bool):
            raise ValueError("decode_video must be bool or None")
        if attention_backend is not None:
            attention_backend = str(attention_backend).strip().lower()
            if attention_backend not in {"sdpa", "flex"}:
                raise ValueError("attention_backend must be 'sdpa', 'flex', or None")
            if rtc and attention_backend != "sdpa":
                raise ValueError("RTC guidance requires attention_backend='sdpa'")
        install_import_guard()
        self.root = Path(root).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.device = device
        self.instruction = instruction
        self.seed = int(seed)
        self.rtc = bool(rtc)
        # None preserves the original entrypoint's decode behavior. Omitting
        # RGB reconstruction still runs the original future-video denoising.
        self.decode_video = not self.rtc if decode_video is None else decode_video
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        self.cfg = load_config(self.root, self.run_dir)
        saved_inference_steps = int(self.cfg.eval_num_inference_steps)
        if inference_steps is not None:
            if int(inference_steps) < 1:
                raise ValueError("inference_steps must be positive")
            self.cfg.eval_num_inference_steps = int(inference_steps)
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            if not checkpoint.is_absolute():
                checkpoint = self.root / checkpoint
        checkpoint, expected_step = select_checkpoint(self.run_dir, checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        self.identity = validate_checkpoint(self.cfg, payload, expected_step)
        step = payload["step"]
        cache = prepare_text_cache(self.root, instruction, device)
        text = torch.load(cache, map_location="cpu", weights_only=True)
        context, mask = official_text_padding(text["context"], text["mask"])
        self.context = context.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        self.context_mask = mask.unsqueeze(0).to(device=device)
        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(self.run_dir / "dataset_stats.json")))
        self.action_pad = _static_dimension_is_pad(self.processor, "action")
        self.state_pad = _static_dimension_is_pad(self.processor, "state")
        for pad in (self.action_pad, self.state_pad):
            if (~pad).nonzero().flatten().tolist() != VALID_DIMS:
                raise ValueError("Piper canonical80 mapping mismatch")
        self.camera_is_pad = torch.zeros((1, 3), dtype=torch.bool, device=device)
        self.action_dim_is_pad = self.action_pad.unsqueeze(0).to(device=device)
        self.proprio_dim_is_pad = self.state_pad.unsqueeze(0).to(device=device)
        self.embodiment_ids = torch.tensor([14], dtype=torch.long, device=device)
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(self.root / "checkpoints")
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        if self.cfg.model.load_text_encoder:
            raise ValueError("Expected the saved cached-T5 Piper training configuration")
        training_backend = str(self.cfg.model.mot_attention_backend)
        if self.rtc or attention_backend is not None:
            # Official RTC guidance differentiates attention w.r.t. noisy
            # actions. This is an inference backend choice, not new weights.
            self.cfg.model.mot_attention_backend = "sdpa" if self.rtc else attention_backend
        if str(self.cfg.model.mot_attention_backend).strip().lower() == "sdpa":
            self.cfg.model.mot_flex_masked_batches = False
        print("PIPER_MODEL_BUILD", str(checkpoint), "step", step, flush=True)
        self.model = instantiate_piper_model(self.cfg, model_dtype=torch.bfloat16, device=device)
        _load_joint96_checkpoint(self.model, payload)
        counts = {}
        for name in ("mot", "proprio_encoder", "action_proprio_encoder"):
            module = getattr(self.model, name, None)
            if module is None:
                continue
            state = module.state_dict()
            for key, value in state.items():
                if not torch.equal(value, payload[name][key].to(device=value.device, dtype=value.dtype)):
                    raise ValueError("Strict tensor restore mismatch: " + name + "." + key)
            counts[name] = len(state)
        self.model.eval().requires_grad_(False)
        # Same default current-frame adapter as the project's Robot37 Joint96
        # evaluation entrypoint. The underlying sampler is not wrapped/replaced.
        from eval.robotwin.current_visual_adapter import configure_current_visual
        current_visual = configure_current_visual(self.model, "auto")
        self.audit = {"checkpoint": str(checkpoint), "step": step, "embodiment_id": 14,
                      "strict_tensor_counts": counts, "normalization": str(self.run_dir / "dataset_stats.json"),
                      "sampler": "wam.inference.frame_joint.infer_frame_joint_window",
                      "action_horizon": 32, "num_video_frames": 9,
                      "num_inference_steps": int(self.cfg.eval_num_inference_steps),
                      "saved_num_inference_steps": saved_inference_steps,
                      "decode_video": self.decode_video,
                      "training_attention_backend": training_backend,
                      "inference_attention_backend": self.model.mot.attention_backend,
                      "current_visual": current_visual, "text_cache": str(cache)}
        self.audit["rtc"] = {"enabled": self.rtc, "mode": "inference_time_guidance" if self.rtc else None,
                             "training_attention_backend": training_backend,
                             "inference_attention_backend": self.model.mot.attention_backend,
                             "decode_video": self.decode_video}
        print("PIPER_MODEL_READY", json.dumps(self.audit), flush=True)
        del payload, state
        gc.collect()

    @torch.no_grad()
    def infer(self, observation, *, rtc_guidance=None):
        if rtc_guidance is not None and not self.rtc:
            raise ValueError("Construct PiperPolicy with rtc=True for RTC guidance")
        obs = validate_observation(observation)
        if obs["prompt"] != self.instruction:
            raise ValueError("Task differs from the loaded T5 context; load a matching policy context")
        original_obs = {"observation": {camera: {"rgb": obs[key]} for camera, key in
                        zip(("head_camera", "left_camera", "right_camera"), IMAGE_KEYS)}}
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        rgb = current_rgb(self.processor, original_obs).to(device=self.device, dtype=torch.bfloat16)
        proprio = RobotWinPolicySession._normalize_state(self, obs["state"]).to(device=self.device, dtype=torch.bfloat16)
        result = infer_frame_joint_window(
            self.model, None, rgb, num_frames=9, action_horizon=32, proprio=proprio,
            context=self.context, context_mask=self.context_mask,
            num_inference_steps=int(self.cfg.eval_num_inference_steps), seed=self.seed,
            rand_device="cpu", tiled=False, video_layout="latent_horizontal", video_view_names="main|left|right",
            camera_is_pad=self.camera_is_pad,
            action_dim_is_pad=self.action_dim_is_pad,
            proprio_dim_is_pad=self.proprio_dim_is_pad,
            embodiment_ids=self.embodiment_ids,
            rtc_guidance=rtc_guidance, decode_video=self.decode_video)
        normalized = result["action"]
        if normalized.shape != (32, 80) or not torch.isfinite(normalized).all():
            raise ValueError("Invalid normalized action chunk")
        actions = RobotWinPolicySession._denormalize_action(self, normalized)[0].astype(np.float32)
        if actions.shape != (32, 14) or not np.isfinite(actions).all():
            raise ValueError("Invalid native joint14 action chunk")
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        return {"actions": actions, "normalized_actions": normalized.numpy(),
                "latency_seconds": time.perf_counter() - start, "checkpoint": self.audit}
