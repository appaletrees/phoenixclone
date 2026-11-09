"""Chunk orchestration independent of the Phoenix model and hardware drivers.

The model lives in the parent. A transport owns observation acquisition and the
30 Hz action clock in a spawned process. No pi policy/client is imported here.
"""
from collections import deque
import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np


COUNTERS = ("controller_ticks", "sent_target_ticks", "queue_underrun_ticks", "missed_control_ticks")


class RecordedObservationRobot:
    """Spawn-picklable replay source; it can never enable or command hardware."""
    action_hz = 30.

    def __init__(self, root, config, **kwargs):
        self.observation = config["observation"]

    def observe(self):
        return {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self.observation.items()}

    def enable(self):
        raise RuntimeError("Recorded observations cannot enable hardware")

    def send_action(self, action):
        raise RuntimeError("Recorded observations cannot command hardware")

    def close(self):
        pass


class ChunkRecorder:
    """Optional bounded recording; a slow disk drops records, not control ticks.

    The queue contains at most max_pending records plus the worker's current
    record. Full images are copied only when requested and a queue slot exists.
    A writer failure is surfaced to the run loop and in the final summary.
    """
    def __init__(self, directory, mode="none", max_pending=4, lerobot_fps=30.):
        if mode not in {"none", "actions", "full"} or max_pending < 1:
            raise ValueError("Expected record mode none/actions/full and a positive queue size")
        self.directory = Path(directory)
        self.lerobot_fps = float(lerobot_fps)
        if not math.isfinite(self.lerobot_fps) or not 1 <= self.lerobot_fps <= 100:
            raise ValueError('LeRobot recording FPS must be in 1..100')
        self.lerobot_export = None
        self.mode = mode
        self.pending = queue.Queue(maxsize=max_pending)
        self.closed = False
        self.error = None
        self.submitted = self.written = self.dropped = 0
        self.thread = None
        if mode != "none":
            self.directory.mkdir(parents=True, exist_ok=True)
            self.thread = threading.Thread(target=self._run, name="phoenix-policy-recorder", daemon=True)
            self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Piper recording failed") from self.error

    def submit(self, index, observation, result, metrics):
        self.check()
        if self.closed:
            raise RuntimeError("Recorder is closed")
        if self.mode == "none":
            return False
        if self.pending.full():
            self.dropped += 1
            return False
        arrays = {key: np.array(result[key], copy=True) for key in ("actions", "normalized_actions")}
        arrays.update(state=np.array(observation["state"], copy=True), prompt=observation["prompt"])
        if self.mode == "full":
            arrays.update({key: np.array(observation[key], copy=True)
                           for key in ("image", "wrist_image_l", "wrist_image_r")})
        # One producer; the worker only removes entries, so the free slot stays
        # available during copying. No unbounded Future list is retained.
        self.pending.put_nowait((index, arrays, dict(metrics)))
        self.submitted += 1
        return True

    def _write(self, index, arrays, metrics):
        stem = self.directory / f"chunk_{index:06d}"
        save = np.savez_compressed if self.mode == "full" else np.savez
        save(str(stem) + ".npz", **arrays)
        Path(str(stem) + ".json").write_text(json.dumps(metrics, indent=2))

    def _run(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    try:
                        self._write(*item)
                        self.written += 1
                    except BaseException as exc:
                        self.error = exc
            finally:
                self.pending.task_done()

    def stats(self):
        return {"mode": self.mode, "submitted": self.submitted, "written": self.written,
                "dropped": self.dropped, "queue_capacity": self.pending.maxsize,
                "format": "lerobot_v2.1", "lerobot": self.lerobot_export}

    def close(self):
        if not self.closed:
            self.closed = True
            if self.thread is not None:
                self.pending.put(None)
                self.thread.join()
                if self.error is None and self.written:
                    try:
                        from wam.inference.piper_lerobot_export import export_run
                        self.lerobot_export = export_run(self.directory, kind='policy', mode=self.mode, fps=self.lerobot_fps)
                    except Exception as exc:
                        self.error = exc
                        self.lerobot_export = {'status':'error', 'error':f'{type(exc).__name__}: {exc}'}
                elif not self.written:
                    self.lerobot_export = {'status':'empty', 'reason':'No successfully written observations'}
        self.check()


def _wait(robot, *, until_tick=None):
    while True:
        status = robot.status()
        if (until_tick is None and status["finished"]) or (
                until_tick is not None and status["controller_ticks"] >= until_tick):
            return status
        time.sleep(min(.01, .5 / robot.action_hz))


def _distribution(samples):
    if not samples:
        return None
    return {"sample_count": len(samples), "p50": float(np.quantile(samples, .5)),
            "p95": float(np.quantile(samples, .95)), "max": float(max(samples))}


def run_policy(policy, robot, output_dir, *, mode="sync", chunks=2, execute_steps=32,
               replan_steps=10, hardware_execute=False, record="none", record_queue_size=4,
               delay_margin=2, guidance_weight=5., record_fps=30., rtc_inference_delay=None,
               rtc_prefix_horizon=None, rtc_prefix_schedule="exp",
               return_on_interrupt="none", return_duration=5.):
    """Run normal Phoenix sampling with sync, async, or inference-time VJP RTC.

    Owns transport cleanup, including on warmup/inference/recording failures.
    chunks=0 runs until interrupted. Finite completion means the requested
    number of chunks ran; it does not claim the robot task succeeded.
    """
    if mode not in {"sync", "async", "rtc"} or chunks < 0:
        raise ValueError("Expected mode sync/async/rtc and chunks >= 0")
    if not 1 <= execute_steps <= 32 or not 1 <= replan_steps <= 32:
        raise ValueError("Execution and replanning windows must lie within the trained horizon 1..32")
    if delay_margin < 0 or not math.isfinite(guidance_weight) or guidance_weight <= 0:
        raise ValueError("Invalid RTC delay margin or guidance weight")
    for name, value in (("rtc_inference_delay", rtc_inference_delay), ("rtc_prefix_horizon", rtc_prefix_horizon)):
        if value is not None and (type(value) is not int or not 0 <= value <= 32):
            raise ValueError(f"{name} must be None or an integer in 0..32")
    if rtc_inference_delay is not None and rtc_prefix_horizon is not None and rtc_inference_delay > rtc_prefix_horizon:
        raise ValueError("rtc_inference_delay must not exceed explicit rtc_prefix_horizon")
    if rtc_prefix_schedule not in ("exp", "linear", "ones", "zeros"):
        raise ValueError("Invalid RTC prefix schedule")
    guidance_options = dict(inference_delay=rtc_inference_delay, prefix_horizon=rtc_prefix_horizon,
                            schedule=rtc_prefix_schedule)
    from wam.inference.piper_return import validate_return_options
    validate_return_options(return_on_interrupt, return_duration)
    out = Path(output_dir)
    recorder = None
    total = dict.fromkeys(COUNTERS, 0)
    latencies, end_to_end = deque(maxlen=256), deque(maxlen=256)
    delay_samples = deque(maxlen=32)
    summary = {"mode": mode, "rtc_algorithm": "inference_time_vjp" if mode == "rtc" else None,
               "rtc_config": {"inference_delay": rtc_inference_delay, "prefix_horizon": rtc_prefix_horizon,
                              "prefix_schedule": rtc_prefix_schedule, "delay_margin": delay_margin,
                              "guidance_weight": guidance_weight} if mode == "rtc" else None,
               "hardware_execution_requested": hardware_execute, "hardware_commands_sent": False,
               "action_hz": robot.action_hz, "action_horizon": 32, "chunks_requested": chunks,
               "chunks_completed": 0, "accepted_plans": 0, "rejected_plans": 0,
               "rtc_prefix_shortfall_chunks": 0,
               "execute_steps": execute_steps if mode == "sync" else None,
               "replan_steps": replan_steps if mode != "sync" else None,
               "completed": False, "interrupted": False}
    summary["return_on_interrupt"] = {"target": return_on_interrupt, "duration": return_duration,
                                      "status": "not_requested"}
    running = False
    execution_attempted = False
    stats_uncertain = False
    started = time.monotonic()

    def stop_clock():
        nonlocal running, stats_uncertain
        if running:
            try:
                status = robot.stop()
                for key in COUNTERS:
                    total[key] += status[key]
                running = False
            except BaseException:
                stats_uncertain = True
                raise

    try:
        out.mkdir(parents=True, exist_ok=True)
        recorder = ChunkRecorder(out, record, record_queue_size, lerobot_fps=record_fps)
        # Compile/warm both relevant inference paths before any enable request.
        packet = robot.observe()
        warm = policy.infer(packet["observation"])
        delay_samples.append(time.monotonic() - packet["request_monotonic"])
        if mode == "rtc":
            from wam.inference.piper_rtc import estimate_delay_ticks, make_guidance
            guidance = make_guidance(warm["normalized_actions"], delay_samples[-1],
                                     action_hz=robot.action_hz, margin=delay_margin, weight=guidance_weight,
                                     **guidance_options)
            packet = robot.observe()
            policy.infer(packet["observation"], rtc_guidance=guidance)
            delay_samples.append(time.monotonic() - packet["request_monotonic"])
        summary["warmup_seconds"] = time.monotonic() - started
        if hardware_execute:
            execution_attempted = True
            robot.enable()
        index = 0
        next_request = last_publish = 0
        while chunks == 0 or index < chunks:
            recorder.check()
            if running:
                _wait(robot, until_tick=next_request)
            packet = robot.observe()
            obs = packet["observation"]
            origin = packet["request_tick"]
            guidance = None
            estimated_delay = None
            requested_delay = None
            prefix_covers_delay = None
            if mode == "rtc" and running:
                # Includes capture/IPC/model/publication latency, not merely
                # CUDA time. A bounded rolling P95 accounts for recent jitter.
                estimated_latency = float(np.quantile(delay_samples, .95))
                estimated_delay = estimate_delay_ticks(
                    estimated_latency, action_hz=robot.action_hz, margin=delay_margin)
                available = packet["previous_available_actions"]
                requested_delay = estimated_delay if rtc_inference_delay is None else rtc_inference_delay
                effective_prefix = min(available, 32 if rtc_prefix_horizon is None else rtc_prefix_horizon)
                prefix_covers_delay = effective_prefix >= estimated_delay
                summary["rtc_prefix_shortfall_chunks"] += int(not prefix_covers_delay)
                if available > 0:
                    guidance = make_guidance(packet["previous_normalized_actions"], estimated_latency,
                                             action_hz=robot.action_hz, margin=delay_margin,
                                             weight=guidance_weight, available_actions=available,
                                             **guidance_options)
            result = (policy.infer(obs, rtc_guidance=guidance)
                      if guidance is not None else policy.infer(obs))
            if not running:
                # A stopped robot holds position during the initial/sync
                # inference, so the new plan starts from index zero.
                running = True  # Even a partially failed start needs cleanup.
                robot.start(result, execute=hardware_execute,
                            limit_steps=execute_steps if mode == "sync" else None)
                publication = {"publish_tick": 0, "discarded_elapsed_actions": 0,
                               "accepted_plan": True, "remaining_actions": 32}
            else:
                publication = robot.publish(result, origin)
            elapsed = time.monotonic() - packet["request_monotonic"]
            latency = float(result["latency_seconds"])
            delay_samples.append(elapsed)
            latencies.append(latency)
            end_to_end.append(elapsed)
            summary["accepted_plans" if publication["accepted_plan"] else "rejected_plans"] += 1
            metrics = {"chunk": index, "mode": mode, "request_tick": origin,
                       "observation_monotonic": packet["observed_monotonic"],
                       "request_monotonic": packet["request_monotonic"],
                       "latency_seconds": latency, "observation_to_publish_seconds": elapsed,
                       "capture_seconds": packet["observed_monotonic"] - packet["request_monotonic"],
                       "rtc_guidance": guidance is not None,
                       "estimated_inference_delay_ticks": estimated_delay,
                       "rtc_requested_delay_ticks": requested_delay,
                       "rtc_guidance_delay_ticks": guidance.inference_delay if guidance is not None else None,
                       "rtc_prefix_schedule": rtc_prefix_schedule if mode == "rtc" else None,
                       "rtc_guidance_weight": guidance_weight if mode == "rtc" else None,
                       "prefix_attention_horizon": guidance.prefix_attention_horizon if guidance is not None else 0,
                       "rtc_prefix_covers_delay": prefix_covers_delay,
                       "previous_available_actions": packet["previous_available_actions"],
                       "sensor_timing": packet.get("sensor_timing"),
                       "hardware_execution_requested": hardware_execute,
                       "joint_unit": "radians", "action_shape": [32, 14], **publication}
            if mode == "sync":
                _wait(robot)
                stop_clock()
                metrics.update(total)
            else:
                next_request = origin + replan_steps
                last_publish = publication["publish_tick"]
                metrics.update(robot.status())
            recorder.submit(index, obs, result, metrics)
            print("PIPER_POLICY_RESULT", json.dumps(metrics), flush=True)
            index += 1
            summary["chunks_completed"] = index
        if running:
            _wait(robot, until_tick=last_publish + replan_steps)
        summary["completed"] = True
    except KeyboardInterrupt:
        summary["interrupted"] = True
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Stop and close the hardware BEFORE waiting on optional disk writes.
        cleanup_errors = []
        try:
            stop_clock()
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            if (summary["interrupted"] and hardware_execute and execution_attempted
                    and return_on_interrupt != "none" and not cleanup_errors):
                print("PIPER_RETURN_START", json.dumps({"target": return_on_interrupt,
                      "duration": return_duration, "gripper_targets": "unchanged",
                      "cancel": "Press Ctrl+C again to abort return and disable"}), flush=True)
                summary["return_on_interrupt"].update(robot.return_to_pose(return_on_interrupt, return_duration))
        except KeyboardInterrupt:
            summary["return_on_interrupt"].update(status="aborted")
        except BaseException as exc:
            summary["return_on_interrupt"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
            cleanup_errors.append(exc)
        finally:
            # Even an interrupt at the return announcement/RPC boundary must
            # disable the motors; no logging or disk I/O precedes this cleanup.
            try:
                robot.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if summary["return_on_interrupt"]["status"] != "not_requested":
            print("PIPER_RETURN_RESULT", json.dumps(summary["return_on_interrupt"]), flush=True)
        try:
            if recorder is not None:
                recorder.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            summary["completed"] = False
            summary["cleanup_errors"] = [f"{type(exc).__name__}: {exc}" for exc in cleanup_errors]
        summary.update(total)
        summary["recording"] = recorder.stats() if recorder is not None else {"mode": record, "initialized": False}
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["model_latency_seconds_last_256"] = _distribution(latencies)
        summary["observation_to_publish_seconds_last_256"] = _distribution(end_to_end)
        summary["hardware_commands_sent"] = (
            None if hardware_execute and execution_attempted and (
                stats_uncertain or summary["return_on_interrupt"]["status"] == "failed")
            else bool(hardware_execute and (total["sent_target_ticks"]
                or summary["return_on_interrupt"].get("sent_targets", 0))))
        summary["continuous_supply_observed"] = bool(mode != "sync" and summary["chunks_completed"] > 1
            and summary["completed"] and not summary["rejected_plans"]
            and not total["queue_underrun_ticks"] and not total["missed_control_ticks"])
        try:
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
        except OSError as exc:
            # A full/unwritable disk must not hide the originating failure or
            # prevent the console from reporting that hardware has stopped.
            summary["completed"] = False
            summary["summary_write_error"] = f"{type(exc).__name__}: {exc}"
            cleanup_errors.append(exc)
        print("PIPER_POLICY_SUMMARY", json.dumps(summary), flush=True)
        if cleanup_errors and "error" not in summary:
            raise RuntimeError("Piper policy cleanup failed") from cleanup_errors[0]
    return summary
