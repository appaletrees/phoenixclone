"""Original Phoenix Joint96 inference with an independent Piper transport.

Default: recorded-data inference only. --live reads hardware; --live --execute
is required to command it. Other active controller locks are always respected.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE_ROOT / "src"), str(SOURCE_ROOT)]

from wam.inference.piper_control import add_control_arguments, settings_from_args, with_control_settings


def recorded_observation(root, run_dir, prompt):
    import numpy as np
    import pyarrow.parquet as pq
    from wam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
    dataset = root / "resources/piper/dataset"
    info = json.loads((dataset / "meta/info.json").read_text())
    episode = next(row for row in (json.loads(x) for x in (dataset / "meta/episodes.jsonl").read_text().splitlines())
                   if prompt in row["tasks"])
    index = int(episode["episode_index"])
    fields = dict(episode_chunk=index // info["chunks_size"], episode_index=index)
    table = pq.read_table(dataset / info["data_path"].format(**fields))
    obs = {"state": np.asarray(table["observation.state"][0].as_py(), dtype=np.float32), "prompt": prompt}
    for camera, key in zip(("cam_high", "cam_left_wrist", "cam_right_wrist"), ("image", "wrist_image_l", "wrist_image_r")):
        path = dataset / info["video_path"].format(**fields, video_key="observation.images." + camera)
        frame = decode_video_frames(str(path), [float(table["timestamp"][0].as_py())], 1e-4)[0]
        obs[key] = frame.mul(255).round().clamp(0, 255).to(__import__("torch").uint8).permute(1, 2, 0).numpy()
    reference = np.asarray(table["action"].slice(0, 32).to_pylist(), dtype=np.float32)
    return obs, reference, {"episode_index": index, "frame_index": 0}


def finish_chunk(out, index, obs, result, metrics, robot=None, captured_at=None):
    """Execute before compressing images, preserving diagnostics on failure."""
    import numpy as np
    try:
        if robot is not None:
            metrics["observation_age_before_execution_seconds"] = time.monotonic() - captured_at
            robot.execute(result["actions"])
            metrics["hardware_commands_sent"] = True
    except Exception as exc:
        if metrics["hardware_commands_sent"] is False:
            # Some targets may have been sent before an execution failure.
            metrics["hardware_commands_sent"] = None
        metrics["execution_error"] = str(exc)
        raise
    finally:
        if robot is not None:
            metrics["observation_age_after_execution_seconds"] = time.monotonic() - captured_at
        np.savez_compressed(out / f"chunk_{index:04d}.npz", **obs,
                            actions=result["actions"], normalized_actions=result["normalized_actions"])
        (out / f"chunk_{index:04d}.json").write_text(json.dumps(metrics, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--run-dir", type=Path, default=Path("postcheckpoints"))
    p.add_argument("--checkpoint", type=Path, help="Explicit weight file, relative to project root; default is weights_manifest.json latest")
    p.add_argument("--list-checkpoints", action="store_true", help="List weight files and their actual steps without loading a GPU model or hardware")
    p.add_argument("--prompt", default="fold the towel and put it into the plate")
    p.add_argument("--input-npz", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("runs/piper-inference"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunks", type=int, default=2)
    p.add_argument("--live", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--execute-steps", type=int, default=10)
    p.add_argument("--rtc", action="store_true", help="Phoenix inference-time RTC plus asynchronous execution/replay")
    p.add_argument("--rtc-delay-margin", type=int, default=2, help="Extra committed prefix ticks beyond measured inference delay")
    p.add_argument("--rtc-guidance-weight", type=float, default=5.)
    p.add_argument("--inference-steps", type=int, help="Explicit inference-only override; default is saved training configuration (10)")
    p.add_argument("--interface-root", type=Path, default=Path("../Desktop/piper_tele/sink-bot-control_bk/example_agilex"))
    p.add_argument("--robot-config", type=Path, default=Path("configs/inference/piper_robot.json"))
    add_control_arguments(p)
    args = p.parse_args()
    try:
        control = settings_from_args(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.execute and not args.live:
        p.error("--execute requires --live")
    if args.input_npz and args.live:
        p.error("Recorded observations cannot be sent to hardware")
    if args.chunks < 1 or not 1 <= args.execute_steps <= 32:
        p.error("Use positive chunks and execute-steps in 1..32")
    if args.rtc_delay_margin < 0 or not __import__("math").isfinite(args.rtc_guidance_weight) or args.rtc_guidance_weight <= 0:
        p.error("RTC requires non-negative delay margin and finite positive guidance weight")
    if args.inference_steps is not None and args.inference_steps < 1:
        p.error("inference-steps must be positive")
    root = args.project_root.resolve()
    os.chdir(root)
    run_dir = args.run_dir.resolve()
    out = args.output_dir.resolve()
    import numpy as np
    from wam.inference.piper import PiperPolicy, IMAGE_KEYS, validate_observation, available_checkpoints
    if args.list_checkpoints:
        print(json.dumps(available_checkpoints(root, run_dir), indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    robot = None
    try:
        if args.live:
            from wam.inference.piper_robot import PhoenixPiperRobot
            if args.rtc:
                from wam.inference.piper_robot_process import PiperRobotProcess
                robot_class = PiperRobotProcess
            else:
                robot_class = PhoenixPiperRobot
            config = with_control_settings(json.loads(args.robot_config.read_text()), control)
            print("PIPER_CONTROL", json.dumps(control.audit()), flush=True)
            (out / "robot_control.json").write_text(json.dumps(control.audit(), indent=2))
            robot = robot_class(args.interface_root, config,
                                prompt=args.prompt, execute_steps=args.execute_steps)
            obs, reference, source = robot.observe(), None, {
                "mode": "live_execute" if args.execute else "live_read_only",
                "transport": robot_class.__module__ + "." + robot_class.__name__,
                "action_hz": robot.action_hz, "execute_steps": args.execute_steps}
        elif args.input_npz:
            with np.load(args.input_npz, allow_pickle=False) as saved:
                obs = {key: saved[key].copy() for key in (*IMAGE_KEYS, "state")}
                obs["prompt"] = str(saved["prompt"].item()) if "prompt" in saved else args.prompt
                reference = saved["reference_actions"].copy() if "reference_actions" in saved else None
            source = {"input_npz": str(args.input_npz)}
        else:
            obs, reference, source = recorded_observation(root, run_dir, args.prompt)
        obs = validate_observation(obs)
        if obs["prompt"] != args.prompt:
            raise ValueError("Recorded prompt differs from requested task")
        np.savez_compressed(out / "observation.npz", **obs,
                            **({"reference_actions": reference} if reference is not None else {}))
        policy = PiperPolicy(root, run_dir, args.prompt, device=args.device, seed=args.seed, rtc=args.rtc,
                             inference_steps=args.inference_steps, checkpoint=args.checkpoint)
        (out / "model_loaded.json").write_text(json.dumps(policy.audit, indent=2))
        if args.rtc:
            from wam.inference.piper_rtc import run_async_rtc
            run_async_rtc(
                policy, robot.observe if robot else lambda: obs,
                (robot.send_action if args.execute else robot.preview_action)
                if robot else lambda action: None, out,
                chunks=args.chunks, replan_steps=args.execute_steps,
                hardware_execute=args.execute, enable=robot.enable if args.execute else None,
                delay_margin=args.rtc_delay_margin, guidance_weight=args.rtc_guidance_weight)
            return
        for index in range(args.chunks):
            if robot:
                obs = robot.observe()
            captured_at = time.monotonic()
            result = policy.infer(obs)
            if args.execute and index == 0:
                # Warm model/compiler before enabling. First actual command is
                # based on a new observation after warmup, never this old one.
                robot.enable()
                obs = robot.observe()
                captured_at = time.monotonic()
                result = policy.infer(obs)
            actions = result["actions"]
            metrics = {"chunk": index, "source": source, "latency_seconds": result["latency_seconds"],
                       "action_shape": list(actions.shape), "joint_unit": "radians",
                       "joint_min": float(actions[:, [0,1,2,3,4,5,7,8,9,10,11,12]].min()),
                       "joint_max": float(actions[:, [0,1,2,3,4,5,7,8,9,10,11,12]].max()),
                       "gripper_min": float(actions[:, [6,13]].min()), "gripper_max": float(actions[:, [6,13]].max()),
                       "hardware_execution_requested": bool(args.execute),
                       "hardware_commands_sent": False}
            if reference is not None:
                joints = [0,1,2,3,4,5,7,8,9,10,11,12]
                metrics["reference_joint_mae_rad"] = float(np.abs(actions[:, joints]-reference[:, joints]).mean())
                metrics["reference_gripper_mae"] = float(np.abs(actions[:, [6,13]]-reference[:, [6,13]]).mean())
            finish_chunk(out, index, obs, result, metrics,
                         robot=robot if args.execute else None, captured_at=captured_at)
            print("PIPER_INFERENCE_RESULT", json.dumps(metrics), flush=True)
    finally:
        if robot:
            robot.close()


if __name__ == "__main__":
    main()
