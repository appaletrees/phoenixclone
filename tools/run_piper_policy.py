"""Independent Phoenix Piper chunk runner; old run_piper_inference.py is unchanged.

Default is recorded-data replay. --live reads cameras/CAN; only --live --execute
enables motion. All resource paths are relative to --project-root (default .).
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from uuid import uuid4

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE_ROOT / "src"), str(SOURCE_ROOT)]

from wam.inference.piper_control import add_control_arguments, settings_from_args, with_control_settings


def rtc_steps(value):
    if value == "auto":
        return None
    try:
        steps = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use auto or an integer in 0..32") from exc
    if not 0 <= steps <= 32:
        raise argparse.ArgumentTypeError("Use auto or an integer in 0..32")
    return steps


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--run-dir", type=Path, default=Path("postcheckpoints"))
    p.add_argument("--checkpoint", type=Path, help="Explicit trained weight file; otherwise use manifest latest")
    p.add_argument("--list-checkpoints", action="store_true")
    p.add_argument("--prompt", default="fold the towel and put it into the plate")
    p.add_argument("--input-npz", type=Path, help="Replay a recorded RGB/state observation")
    p.add_argument("--live", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--return-on-interrupt", choices=("initial", "zero", "none"), default="initial",
                   help="Ctrl+C during live execution: return arm joints to pre-enable pose or zero, then disable")
    p.add_argument("--return-duration", type=float, default=5.,
                   help="Seconds for the joint return ramp; gripper targets stay unchanged; Ctrl+C again aborts")
    p.add_argument("--mode", choices=("sync", "async", "rtc"), default="sync")
    p.add_argument("--chunks", type=int, default=2, help="Predicted chunks after warmup; 0 runs until Ctrl+C")
    p.add_argument("--execute-steps", type=int, default=32, help="Sync: execute first K actions, 1..32")
    p.add_argument("--replan-steps", type=int, default=10, help="Async/RTC: request interval in controller ticks, 1..32")
    p.add_argument("--inference-steps", type=int, help="Override saved denoising count for inference only")
    p.add_argument("--attention-backend", choices=("saved", "sdpa", "flex"), default="saved",
                   help="Saved backend by default; RTC requires SDPA and selects it automatically")
    p.add_argument("--rtc-delay-margin", type=int, default=2)
    p.add_argument("--rtc-guidance-weight", type=float, default=5.)
    p.add_argument("--rtc-inference-delay", type=rtc_steps, default=None, metavar="auto|0..32",
                   help="Soft-guidance delay steps; auto uses rolling P95 plus delay margin; fixed values exclude margin")
    p.add_argument("--rtc-prefix-horizon", type=rtc_steps, default=None, metavar="auto|0..32",
                   help="Maximum old-plan steps to guide; auto uses all real remaining actions, never padded targets")
    p.add_argument("--rtc-prefix-schedule", choices=("exp", "linear", "ones", "zeros"), default="exp",
                   help="Old-plan guidance weights: exp/linear taper, ones full prefix, zeros delay region only")
    p.add_argument("--record", choices=("none", "actions", "full"), default="none")
    p.add_argument("--record-queue-size", type=int, default=4)
    p.add_argument("--record-fps", type=float, default=30., help="LeRobot v2.1 timeline FPS; original sample times are preserved")
    p.add_argument("--output-dir", type=Path, default=Path("runs/piper-policy"),
                   help="Each run creates its own timestamped subdirectory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interface-root", type=Path, default=Path("../Desktop/piper_tele/sink-bot-control_bk/example_agilex"))
    p.add_argument("--robot-config", type=Path, default=Path("configs/inference/piper_robot.json"))
    add_control_arguments(p)
    return p


def parse_args(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        settings_from_args(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.execute and not args.live:
        p.error("--execute requires --live")
    if not math.isfinite(args.return_duration) or args.return_duration <= 0:
        p.error("return-duration must be finite and positive")
    if args.input_npz and args.live:
        p.error("Recorded observations cannot be sent to hardware")
    if args.chunks < 0 or not 1 <= args.execute_steps <= 32 or not 1 <= args.replan_steps <= 32:
        p.error("Use chunks >= 0 and execution/replanning windows in 1..32")
    if args.record_queue_size < 1:
        p.error("record-queue-size must be positive")
    if not math.isfinite(args.record_fps) or not 1 <= args.record_fps <= 100:
        p.error("record-fps must be in 1..100")
    if args.inference_steps is not None and args.inference_steps < 1:
        p.error("inference-steps must be positive")
    if args.rtc_delay_margin < 0 or not math.isfinite(args.rtc_guidance_weight) or args.rtc_guidance_weight <= 0:
        p.error("RTC requires non-negative delay margin and finite positive guidance weight")
    if (args.rtc_inference_delay is not None and args.rtc_prefix_horizon is not None
            and args.rtc_inference_delay > args.rtc_prefix_horizon):
        p.error("rtc-inference-delay must not exceed an explicitly set rtc-prefix-horizon")
    if args.mode == "rtc" and args.attention_backend == "flex":
        p.error("RTC requires SDPA; choose --attention-backend saved or sdpa")
    if not args.prompt.strip():
        p.error("prompt must not be empty")
    return args


def resolve(root, path):
    return (root / Path(path).expanduser()).resolve()


def preflight(root, run_dir, checkpoint=None):
    """Fail for missing/deleted trained resources before importing any driver."""
    required = [run_dir / "config.yaml", run_dir / "dataset_stats.json"]
    if checkpoint is not None:
        required.append(resolve(root, checkpoint))
    else:
        manifest = run_dir / "checkpoints/weights/weights_manifest.json"
        required.append(manifest)
        if manifest.is_file():
            path = (manifest.parent / json.loads(manifest.read_text())["latest"]["path"]).resolve()
            if not path.is_relative_to(manifest.parent.resolve()):
                raise ValueError("Invalid weights manifest path")
            required.append(path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Restore the trained run (config, dataset_stats and weights) before inference. Missing:\n"
                                + "\n".join(missing))


def create_output(root, output_dir):
    parent = resolve(root, output_dir)
    path = parent / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def main(argv=None):
    args = parse_args(argv)
    root = args.project_root.resolve()
    run_dir = resolve(root, args.run_dir)
    if args.list_checkpoints:
        from wam.inference.piper import available_checkpoints
        print(json.dumps(available_checkpoints(root, run_dir), indent=2))
        return
    preflight(root, run_dir, args.checkpoint)
    os.chdir(root)
    if args.live:
        from wam.inference.piper_can import resolve_can_config
        config = resolve_can_config(json.loads(resolve(root, args.robot_config).read_text()))
        config = with_control_settings(config, settings_from_args(args))
        print("PIPER_CONTROL", json.dumps(settings_from_args(args).audit()), flush=True)
        if not resolve(root, args.interface_root).is_dir():
            raise FileNotFoundError(resolve(root, args.interface_root))
        print("PIPER_CAN_READY", json.dumps({key: config.get(key) for key in
              ("left_can", "right_can", "can_usb_serials", "feedback_id_offsets")}), flush=True)
    import numpy as np
    from wam.inference.piper import PiperPolicy, IMAGE_KEYS, validate_observation
    from wam.inference.piper_chunk_process import PiperChunkProcess
    from wam.inference.piper_policy_runtime import RecordedObservationRobot, run_policy
    if args.live:
        source = {"mode": "live_execute" if args.execute else "live_read_only"}
    else:
        if args.input_npz:
            with np.load(resolve(root, args.input_npz), allow_pickle=False) as saved:
                obs = {key: saved[key].copy() for key in (*IMAGE_KEYS, "state")}
                obs["prompt"] = str(saved["prompt"].item()) if "prompt" in saved else args.prompt
            source = {"mode": "recorded_replay", "input_npz": str(args.input_npz)}
        else:
            from tools.run_piper_inference import recorded_observation
            obs, _, source = recorded_observation(root, run_dir, args.prompt)
            source["mode"] = "recorded_replay"
        obs = validate_observation(obs)
        if obs["prompt"] != args.prompt:
            raise ValueError("Recorded prompt differs from requested task")
        config = {"observation": obs}
    # All model loading, strict restoration and cache preparation precede any
    # hardware connection. No synthetic/replacement model is silently loaded.
    policy = PiperPolicy(root, run_dir, args.prompt, device=args.device, seed=args.seed,
                         rtc=args.mode == "rtc", inference_steps=args.inference_steps,
                         checkpoint=args.checkpoint, decode_video=False,
                         attention_backend=None if args.attention_backend == "saved" else args.attention_backend)
    out = create_output(root, args.output_dir)
    receipt = {"options": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
               "source": source, "model": policy.audit,
               "robot_control": settings_from_args(args).audit() if args.live else None}
    (out / "run.json").write_text(json.dumps(receipt, indent=2))
    print("PIPER_POLICY_OUTPUT", str(out), flush=True)
    robot = None
    try:
        robot = PiperChunkProcess(resolve(root, args.interface_root), config,
                                  prompt=args.prompt, execute_steps=args.execute_steps,
                                  hardware_factory=None if args.live else RecordedObservationRobot)
        return run_policy(policy, robot, out, mode=args.mode, chunks=args.chunks,
                          execute_steps=args.execute_steps, replan_steps=args.replan_steps,
                          hardware_execute=args.execute, record=args.record,
                          record_queue_size=args.record_queue_size,
                          record_fps=args.record_fps,
                          delay_margin=args.rtc_delay_margin, guidance_weight=args.rtc_guidance_weight,
                          rtc_inference_delay=args.rtc_inference_delay,
                          rtc_prefix_horizon=args.rtc_prefix_horizon,
                          rtc_prefix_schedule=args.rtc_prefix_schedule,
                          return_on_interrupt=args.return_on_interrupt, return_duration=args.return_duration)
    except BaseException as exc:
        (out / "failure.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2))
        raise
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    main()
