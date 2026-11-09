"""Validate the target server image and optional tiny CUDA probes, without weights."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wam.utils.server_compat import attention_smoke, check_install_plan, installed_versions, protected, version_errors, zero1_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--write-image-constraints")
    parser.add_argument("--check-plan")
    parser.add_argument("--cpu-smoke", action="store_true", help="API test only; does not certify the CUDA image")
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--output", default="resources/compat/server-runtime.json")
    args = parser.parse_args()
    report = {"status": "failed", "training_started": False}
    try:
        if args.check_plan:
            report["phase"] = "installation_plan"
            report.update(check_install_plan(args.check_plan), status="passed", mode="installation_plan")
        elif args.cpu_smoke:
            report["phase"] = "cpu_attention"
            import torch
            report.update(mode="cpu_api_only", torch=str(torch.__version__), checks=attention_smoke(), status="passed")
        else:
            report["phase"] = "import:torch"
            import torch
            report["phase"] = "import:torchvision"
            import torchvision
            versions = installed_versions()
            report.update(mode="base" if args.base_only else "runtime", packages=versions, torch_cuda=torch.version.cuda)
            report["phase"] = "version_check"
            errors = version_errors(versions, sys.version_info, torch.version.cuda, args.base_only)
            if errors:
                raise RuntimeError("; ".join(errors))
            report["phase"] = "torchvision_operators"
            if not torchvision.extension._has_ops():
                raise RuntimeError("torchvision native operators failed to load")
            if args.write_image_constraints:
                p = Path(args.write_image_constraints)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("# Preserve this image's compiled binaries\n" + "".join(f"{name}=={version}\n" for name, version in sorted(versions.items()) if protected(name)))
            if not args.base_only:
                modules = ["accelerate", "deepspeed", "transformers", "datasets", "av", "wam.model.factory", "wam.runtime"]
                for name in modules:
                    report["phase"] = "import:" + name
                    importlib.import_module(name)
                report["phase"] = "video_backend"
                from wam.datasets.lerobot.lerobot.datasets.video_utils import get_safe_default_codec
                report["imports"] = modules
                report["video_backend"] = get_safe_default_codec()
            if args.cuda_smoke:
                report["phase"] = "cuda_availability"
                if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                    raise RuntimeError("CUDA smoke checks require an available BF16-capable server GPU")
                report["phase"] = "cuda_attention"
                report["attention"] = attention_smoke("cuda:0")
                report["phase"] = "cuda_deepspeed_zero1"
                report["deepspeed"] = zero1_smoke()
            report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    if report["status"] == "passed":
        report["phase"] = "complete"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
