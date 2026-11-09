"""Asynchronous Piper execution using Phoenix's inference-time RTC guidance."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import torch

from wam.inference.online_action_policy import RTCActionGuidance

HORIZON = 32


class ActionTimeline:
    """All plan indices refer to controller ticks at observation/request time."""
    def __init__(self, actions, normalized, origin=0):
        self.replace(actions, normalized, origin)

    def replace(self, actions, normalized, origin):
        actions = np.asarray(actions, dtype=np.float32)
        normalized = np.asarray(normalized, dtype=np.float32)
        if actions.shape != (HORIZON, 14) or normalized.shape != (HORIZON, 80):
            raise ValueError("Expected native32x14 and normalized32x80 plans")
        if not np.isfinite(actions).all() or not np.isfinite(normalized).all():
            raise ValueError("Non-finite RTC plan")
        self.actions, self.normalized = actions.copy(), normalized.copy()
        self.origin = int(origin)

    def action_at(self, tick):
        offset = int(tick) - self.origin
        return self.actions[offset].copy() if 0 <= offset < HORIZON else None

    def aligned_prefix(self, tick):
        offset = int(tick) - self.origin
        if offset < 0:
            raise ValueError("Cannot align a future plan to a past observation")
        indices = np.minimum(np.arange(HORIZON) + offset, HORIZON - 1)
        return self.normalized[indices].copy(), max(0, HORIZON - offset)


class AsyncActionExecutor:
    def __init__(self, send_action, initial, *, action_hz=30.):
        self.send_action = send_action
        self.timeline = ActionTimeline(initial["actions"], initial["normalized_actions"])
        self.action_hz = float(action_hz)
        self.tick = 0
        self.sent_ticks = 0
        self.underrun_ticks = 0
        self.missed_ticks = 0
        self.fault = None
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="phoenix-action-clock", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        start = time.monotonic()
        try:
            while not self.stop_event.is_set():
                with self.condition:
                    due = max(self.tick, int((time.monotonic() - start) * self.action_hz))
                    self.missed_ticks += due - self.tick
                    action = self.timeline.action_at(due)
                    if action is None:
                        # No invented/extrapolated joint motion. The hardware
                        # driver continues its last target until a valid plan.
                        self.underrun_ticks += 1
                    else:
                        self.send_action(action)
                        self.sent_ticks += 1
                    self.tick = due + 1
                    self.condition.notify_all()
                    deadline = start + self.tick / self.action_hz
                self.stop_event.wait(max(0., deadline - time.monotonic()))
        except BaseException as exc:
            with self.condition:
                self.fault = exc
                self.stop_event.set()
                self.condition.notify_all()

    def _check(self):
        if self.fault is not None:
            raise RuntimeError("Asynchronous Piper driver failed") from self.fault
        if self.stop_event.is_set():
            raise RuntimeError("Asynchronous Piper execution stopped")

    def wait_until(self, tick):
        with self.condition:
            while self.tick < tick:
                self._check()
                self.condition.wait(.1)
            self._check()

    def snapshot(self):
        with self.condition:
            self._check()
            previous, available = self.timeline.aligned_prefix(self.tick)
            return self.tick, previous, available

    def publish(self, result, origin):
        with self.condition:
            self._check()
            skipped = max(0, self.tick - int(origin))
            accepted = skipped < HORIZON
            if accepted:
                self.timeline.replace(result["actions"], result["normalized_actions"], origin)
            return {"publish_tick": self.tick, "discarded_elapsed_actions": min(skipped, HORIZON),
                    "accepted_plan": accepted, "remaining_actions": max(0, HORIZON-skipped)}

    def stats(self):
        with self.condition:
            return {"controller_ticks": self.tick, "sent_target_ticks": self.sent_ticks,
                    "queue_underrun_ticks": self.underrun_ticks, "missed_control_ticks": self.missed_ticks}

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("Asynchronous Piper command thread did not exit")
        if self.fault is not None:
            raise RuntimeError("Asynchronous Piper driver failed") from self.fault


def estimate_delay_ticks(latency, *, action_hz=30., margin=2):
    """Keep the full estimate so logs expose delays beyond the usable prefix."""
    return max(0, math.ceil(float(latency) * action_hz) + margin)


def make_guidance(previous, latency, *, action_hz=30., margin=2, weight=5.,
                  available_actions=HORIZON, inference_delay=None, prefix_horizon=None,
                  schedule="exp"):
    """Guide only real remaining predictions, never the right-padded tail.

    Padding preserves the fixed tensor shape but is not a predicted hold plan.
    When latency exceeds coverage, guide the available prefix only; the caller
    must still report the full delay estimate and any queue underruns.
    """
    if not 0 <= available_actions <= HORIZON:
        raise ValueError("Available RTC actions must lie within 0..32")
    for name, value in (("inference_delay", inference_delay), ("prefix_horizon", prefix_horizon)):
        if value is not None and (type(value) is not int or not 0 <= value <= HORIZON):
            raise ValueError(f"{name} must be None or an integer in 0..32")
    if inference_delay is not None and prefix_horizon is not None and inference_delay > prefix_horizon:
        raise ValueError("inference_delay must not exceed explicit prefix_horizon")
    if schedule not in ("exp", "linear", "ones", "zeros"):
        raise ValueError("Invalid RTC prefix schedule")
    prefix_horizon = min(int(available_actions), HORIZON if prefix_horizon is None else prefix_horizon)
    requested_delay = (estimate_delay_ticks(latency, action_hz=action_hz, margin=margin)
                       if inference_delay is None else inference_delay)
    delay = min(prefix_horizon, requested_delay)
    return RTCActionGuidance(
        prev_action_chunk=torch.from_numpy(np.asarray(previous, dtype=np.float32)).unsqueeze(0),
        inference_delay=delay, prefix_attention_horizon=prefix_horizon,
        max_guidance_weight=weight, schedule=schedule)


def save_rtc_chunk(out, index, obs, result, metrics):
    np.savez_compressed(Path(out) / f"chunk_{index:04d}.npz", **obs,
                        actions=result["actions"], normalized_actions=result["normalized_actions"])
    (Path(out) / f"chunk_{index:04d}.json").write_text(json.dumps(metrics, indent=2))


def run_async_rtc(policy, observe, send_action, out, *, chunks, replan_steps=10,
                  action_hz=30., hardware_execute=False, enable=None, delay_margin=2,
                  guidance_weight=5.):
    """The calling thread owns GPU inference; one thread owns control ticks.

    With a no-op sender this is a wall-clock offline replay, not hardware motion.
    RTC timing uses measured inference delay and drops actions consumed while
    inference was in flight. Image compression runs on a separate worker.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Warm both normal and VJP paths while hardware remains disabled.
    warm = policy.infer(observe())
    warm_guidance = make_guidance(warm["normalized_actions"], warm["latency_seconds"],
                                 action_hz=action_hz, margin=delay_margin, weight=guidance_weight)
    warm_rtc = policy.infer(observe(), rtc_guidance=warm_guidance)
    latency = warm_rtc["latency_seconds"]
    if enable is not None:
        enable()
    obs = observe()
    initial = policy.infer(obs)
    executor = AsyncActionExecutor(send_action, initial, action_hz=action_hz)
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phoenix-rtc-recorder")
    writes = []
    final = {"mode": "rtc_async_execute" if hardware_execute else "rtc_async_replay",
             "hardware_execution_requested": hardware_execute, "action_hz": action_hz,
             "replan_steps": replan_steps, "action_horizon": HORIZON,
             "warmup_rtc_seconds": latency, "completed": False}
    try:
        executor.start()
        metrics = {"chunk": 0, "latency_seconds": initial["latency_seconds"],
                   "action_shape": [HORIZON, 14], "joint_unit": "radians", "rtc_guidance": False,
                   "request_tick": 0, "accepted_plan": True,
                   "hardware_commands_sent": None if hardware_execute else False}
        writes.append(writer.submit(save_rtc_chunk, out, 0, obs, initial, metrics))
        next_request = replan_steps
        last_publish = 0
        for index in range(1, chunks):
            executor.wait_until(next_request)
            obs = observe()
            origin, previous, available = executor.snapshot()
            estimated_delay = estimate_delay_ticks(latency, action_hz=action_hz, margin=delay_margin)
            guidance = make_guidance(previous, latency, action_hz=action_hz,
                                     margin=delay_margin, weight=guidance_weight,
                                     available_actions=available) if available else None
            result = policy.infer(obs, rtc_guidance=guidance)
            publication = executor.publish(result, origin)
            latency = result["latency_seconds"]
            metrics = {"chunk": index, "request_tick": origin,
                       "previous_available_actions": available,
                       "rtc_guidance": guidance is not None,
                       "estimated_inference_delay_ticks": estimated_delay,
                       "rtc_prefix_covers_delay": available >= estimated_delay,
                       "rtc_guidance_delay_ticks": guidance.inference_delay if guidance else None,
                       "prefix_attention_horizon": guidance.prefix_attention_horizon if guidance else 0,
                       "guidance_weight": guidance_weight,
                       "latency_seconds": latency, "action_shape": [HORIZON,14], "joint_unit": "radians",
                       "hardware_commands_sent": None if hardware_execute else False,
                       **publication, **executor.stats()}
            writes.append(writer.submit(save_rtc_chunk, out, index, obs, result, metrics))
            print("PIPER_RTC_RESULT", json.dumps(metrics), flush=True)
            next_request = origin + replan_steps
            last_publish = publication["publish_tick"]
            # Surface recording failures without delaying the action thread.
            for future in writes:
                if future.done():
                    future.result()
        executor.wait_until(last_publish + replan_steps)
        final["completed"] = True
    except BaseException as exc:
        final["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup_error = None
        try:
            executor.close()
        except BaseException as exc:
            cleanup_error = exc
            final["completed"] = False
            final["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            writer.shutdown(wait=True)
            for future in writes:
                try:
                    future.result()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                    final["completed"] = False
                    final["recording_error"] = f"{type(exc).__name__}: {exc}"
            final.update(executor.stats())
            final["continuous_supply_observed"] = bool(chunks > 1 and final["completed"]
                and final["queue_underrun_ticks"] == 0 and final["missed_control_ticks"] == 0)
            final["hardware_commands_sent"] = bool(hardware_execute and final["sent_target_ticks"])
            (out / "rtc_summary.json").write_text(json.dumps(final, indent=2))
        if cleanup_error is not None:
            raise cleanup_error
    print("PIPER_RTC_SUMMARY", json.dumps(final), flush=True)
    return final
