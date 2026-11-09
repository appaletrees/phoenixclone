"""Read-only server diagnostics; never installed as a training-time patch."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

from packaging.utils import canonicalize_name
from packaging.version import Version

BASE_VERSIONS = {"torch": "2.10.0", "torchvision": "0.25.0", "torchaudio": "2.10.0", "vllm": "0.19.0"}
PROFILE_VERSIONS = {**BASE_VERSIONS, "numpy": "2.2.6", "datasets": "3.6.0", "torchcodec": "0.10.0",
                    "deepspeed": "0.19.8", "accelerate": "1.12.0", "transformers": "4.57.6",
                    "huggingface-hub": "0.36.0", "safetensors": "0.5.3", "packaging": "25.0"}
PROTECTED = {"torch", "torchvision", "torchaudio", "vllm", "triton", "flashinfer-python", "flashinfer-cubin"}


def protected(name):
    name = canonicalize_name(name)
    return name in PROTECTED or name.startswith("nvidia-")


def installed_versions():
    versions = {}
    for distribution in metadata.distributions():
        if distribution.metadata.get("Name"):
            # Match importlib.metadata.version(): the first visible distribution
            # wins when a vendor image retains shadowed site-packages entries.
            versions.setdefault(canonicalize_name(distribution.metadata["Name"]), distribution.version)
    return versions


def version_errors(versions, python_version, cuda_version, base_only=False):
    errors = []
    if tuple(python_version[:2]) != (3, 12):
        errors.append("The server profile requires Python 3.12")
    if cuda_version is None or Version(cuda_version).release[:2] != (13, 0):
        errors.append(f"Expected torch.version.cuda=13.0, got {cuda_version!r}")
    for name, expected in (BASE_VERSIONS if base_only else PROFILE_VERSIONS).items():
        actual = versions.get(name)
        if actual is None or Version(actual).base_version != expected or Version(actual).is_prerelease:
            errors.append(f"{name}: expected {expected}, found {actual!r}")
    return errors


def check_install_plan(path):
    plan = json.loads(Path(path).read_text())
    replacements = [p["metadata"]["name"] for p in plan.get("install", []) if protected(p["metadata"]["name"])]
    if replacements:
        raise ValueError("Installation would replace protected image binaries: " + ", ".join(replacements))
    return {"protected_binary_replacements": [], "planned_packages": len(plan.get("install", []))}


def attention_smoke(device="cpu"):
    """Exercise the project's unchanged Flex wrapper against the same dense mask.

    SDPA is only a numerical test oracle; no training backend is changed.
    """
    import torch
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import create_block_mask
    from wam.model.modules.mot.mixture_of_transformers import _wam_flex_attention
    from wam.model.modules.conditioning.frame_attention import build_joint96_attention_mask

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    torch.manual_seed(42)
    q, k, v = [torch.randn(2, 2, 128, 16, device=device, dtype=dtype, requires_grad=True) for _ in range(3)]
    layout = build_joint96_attention_mask(video_seq_len=96, action_seq_len=32,
                                         video_tokens_per_frame=32, device=torch.device(device))
    def mask_mod(b, h, qi, ki):
        return layout[qi, ki]
    block = create_block_mask(mask_mod, B=None, H=None, Q_LEN=128, KV_LEN=128,
                              device=device, BLOCK_SIZE=128)
    keys = torch.ones(2, 128, device=device, dtype=torch.bool)
    keys[0, 112:] = False
    keys[1] = False
    empty = ~keys.any(-1)
    run = torch.compile(_wam_flex_attention, dynamic=True, fullgraph=True) if device.startswith("cuda") else _wam_flex_attention
    options = {"fwd_BLOCK_M": 64, "fwd_BLOCK_N": 64, "fwd_num_stages": 2,
               "bwd_BLOCK_M1": 64, "bwd_BLOCK_N1": 64, "bwd_BLOCK_M2": 64,
               "bwd_BLOCK_N2": 64, "bwd_num_stages": 2}
    actual = run(q, k, v, block, options, keys, empty)
    diagonal = torch.eye(128, dtype=torch.bool, device=device)
    dense_mask = layout[None, None] & (keys[:, None, None, :] | (empty[:, None, None, None] & diagonal[None, None]))
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)
    # The two forwards share only leaf inputs, so neither backward needs graph
    # retention. Compiled PyTorch 2.10 backward also requires these flags false
    # when AOTAutograd donates intermediate buffers.
    grads = torch.autograd.grad(actual.float().square().mean(), (q, k, v),
                                retain_graph=False, create_graph=False)
    expected_grads = torch.autograd.grad(expected.float().square().mean(), (q, k, v),
                                         retain_graph=False, create_graph=False)
    atol, rtol = ((.02, .05) if dtype == torch.bfloat16 else (2e-5, 2e-4))
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    grad_atol, grad_rtol = ((2e-5, .08) if dtype == torch.bfloat16 else (1e-7, 5e-4))
    for grad, expected_grad in zip(grads, expected_grads):
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, expected_grad, atol=grad_atol, rtol=grad_rtol)
    return {"device": device, "dtype": str(dtype), "masked_flex_forward_backward": "passed",
            "max_forward_error": float((actual - expected).abs().max().detach().cpu())}


def zero1_smoke():
    """One tiny synthetic BF16/AdamW/ZeRO-1 step, not a Phoenix training run."""
    import os
    import tempfile
    import torch
    import torch.distributed as dist
    import deepspeed

    if dist.is_initialized():
        raise RuntimeError("Run the smoke check in a separate, non-distributed process")
    old_rank = os.environ.get("LOCAL_RANK")
    os.environ["LOCAL_RANK"] = "0"
    torch.cuda.set_device(0)
    try:
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("nccl", init_method="file://" + str(Path(directory) / "rendezvous"), rank=0, world_size=1)
            config = json.loads(Path("scripts/ds_configs/ds_joint96_posttrain_zero1.json").read_text())
            # Resolve only the original 'auto' fields for this synthetic probe.
            config.update(train_batch_size=2, train_micro_batch_size_per_gpu=2, gradient_accumulation_steps=1)
            model = torch.nn.Linear(8, 8).to(device="cuda", dtype=torch.bfloat16)
            # Make one small update observable even after BF16 rounding.
            torch.nn.init.zeros_(model.weight)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config, dist_init_required=False)
            before = model.weight.detach().float().clone()
            loss = engine(torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)).float().square().mean()
            engine.backward(loss)
            engine.step()
            assert engine.zero_optimization_stage() == 1
            assert torch.isfinite(model.weight).all() and not torch.equal(before, model.weight.detach().float())
            return {"zero_stage": 1, "bf16_adamw_step": "passed", "optimizer_updates": engine.global_steps}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if old_rank is None:
            os.environ.pop("LOCAL_RANK", None)
        else:
            os.environ["LOCAL_RANK"] = old_rank
