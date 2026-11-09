"""Run the Piper target clock beside its drivers, outside the model process.

This module does not construct a policy or CUDA context. Only the worker's
default hardware factory imports the existing Phoenix Piper transport. A
supplied spawn-picklable factory supports offline tests without any drivers.
"""
import logging
import multiprocessing as mp
import queue
import signal
import threading
import time
import traceback

import numpy as np

LOG = logging.getLogger(__name__)
HORIZON = 32


def _plan(result):
    actions = np.array(result["actions"], dtype=np.float32, copy=True)
    normalized = np.array(result["normalized_actions"], dtype=np.float32, copy=True)
    if actions.shape != (HORIZON, 14) or normalized.shape != (HORIZON, 80):
        raise ValueError("Expected native actions[32,14] and normalized_actions[32,80]")
    if not np.isfinite(actions).all() or not np.isfinite(normalized).all():
        raise ValueError("Action plans must contain finite values")
    return actions, normalized


def _idle_stats():
    return dict(controller_ticks=0, sent_target_ticks=0, queue_underrun_ticks=0,
                missed_control_ticks=0, finished=True)


class _ChunkClock:
    """A local clock whose lock is never held while capturing observations."""
    def __init__(self, sender, result, action_hz, limit_steps=None):
        self.actions, self.normalized = _plan(result)
        if not np.isfinite(action_hz) or action_hz <= 0:
            raise ValueError("Expected a finite positive action_hz")
        if limit_steps is not None and (type(limit_steps) is not int or not 1 <= limit_steps <= HORIZON):
            raise ValueError("limit_steps must be None or an integer in 1..32")
        self.sender, self.action_hz, self.limit_steps = sender, float(action_hz), limit_steps
        self.origin = self.tick = self.sent = self.underruns = self.missed = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.finished = False
        self.fault = None
        self.thread = threading.Thread(target=self._run, name="phoenix-piper-chunk-clock", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        start = deadline = time.monotonic()
        period = 1.0 / self.action_hz
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                with self.lock:
                    if self.limit_steps is None:
                        due = max(self.tick, int((now - start) * self.action_hz))
                        self.missed += due - self.tick
                    else:
                        # A synchronous chunk sends exactly its first K targets.
                        # A late loop stretches cadence instead of omitting targets
                        # or issuing a burst of catch-up commands.
                        due = self.tick
                        self.missed += max(0, int((now - deadline) * self.action_hz))
                    offset = due - self.origin
                    if 0 <= offset < HORIZON:
                        self.sender(self.actions[offset].copy())
                        self.sent += 1
                    else:
                        self.underruns += 1
                    self.tick = due + 1
                    if self.limit_steps is not None and self.tick >= self.limit_steps:
                        # Keep the last target active for its complete control
                        # interval before the next synchronous observation.
                        final_tick = True
                    else:
                        final_tick = False
                    deadline = (start + self.tick * period if self.limit_steps is None
                                else max(deadline + period, now + period))
                self.stop_event.wait(max(0.0, deadline - time.monotonic()))
                if final_tick:
                    break
        except BaseException:
            with self.lock:
                self.fault = traceback.format_exc()
            self.stop_event.set()
        finally:
            with self.lock:
                self.finished = True

    def check(self):
        with self.lock:
            fault = self.fault
        if fault is not None:
            raise RuntimeError("Piper target clock failed:\n" + fault)

    def snapshot(self):
        self.check()
        with self.lock:
            offset = max(0, self.tick - self.origin)
            indices = np.minimum(np.arange(HORIZON) + offset, HORIZON - 1)
            return self.tick, self.normalized[indices].copy(), max(0, HORIZON - offset), time.monotonic()

    def publish(self, result, origin):
        actions, normalized = _plan(result)
        if type(origin) is not int or origin < 0:
            raise ValueError("Plan origin must be a non-negative integer tick")
        self.check()
        with self.lock:
            if self.limit_steps is not None or self.finished:
                raise RuntimeError("publish requires an active asynchronous clock")
            if origin > self.tick:
                raise ValueError("Cannot publish a plan from a future tick")
            skipped = self.tick - origin
            accepted = skipped < HORIZON
            if accepted:
                self.actions, self.normalized, self.origin = actions, normalized, origin
            return dict(publish_tick=self.tick, discarded_elapsed_actions=min(skipped, HORIZON),
                        accepted_plan=accepted, remaining_actions=max(0, HORIZON - skipped))

    def stats(self):
        self.check()
        with self.lock:
            return dict(controller_ticks=self.tick, sent_target_ticks=self.sent,
                        queue_underrun_ticks=self.underruns, missed_control_ticks=self.missed,
                        finished=self.finished)

    def stop(self, *, raise_fault=True):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("Piper target clock did not stop")
        if raise_fault:
            return self.stats()
        with self.lock:
            return dict(controller_ticks=self.tick, sent_target_ticks=self.sent,
                        queue_underrun_ticks=self.underruns, missed_control_ticks=self.missed,
                        finished=self.finished)


def _preview(action):
    if np.asarray(action).shape != (14,) or not np.isfinite(action).all():
        raise ValueError("Expected one finite preview target[14]")


def _hardware_factory(root, config, **kwargs):
    from wam.inference.piper_robot import PhoenixPiperRobot, build_robot_observation
    from wam.inference.piper_feedback import gripper_diagnostics

    class TimestampedPiperRobot(PhoenixPiperRobot):
        def observe_with_metadata(self):
            frames = self.cameras.capture_frames()
            snapshots = [controller.get_feedback_snapshot() for controller in self.controllers]
            observation = build_robot_observation(frames, snapshots, self.config["camera_mapping"], self.prompt)
            mono, wall = time.monotonic(), time.time()
            cameras = {}
            for camera in self.config["camera_mapping"]:
                frame = frames[camera]
                cameras[camera] = {key: frame.get(key) for key in
                                   ("timestamp", "frame_id", "device_timestamp_ms", "device_frame_number")}
                timestamp = frame.get("timestamp")
                cameras[camera]["age_seconds"] = None if timestamp is None else mono - float(timestamp)
            feedback = {}
            for arm, snapshot, controller in zip(("left", "right"), snapshots, self.controllers):
                feedback[arm] = {key: snapshot.get(key) for key in
                                 ("joint_timestamp", "gripper_timestamp", "joint_hz", "gripper_hz", "read_monotonic")}
                for field in ("joint", "gripper"):
                    feedback[arm][field + "_age_seconds"] = wall - float(snapshot[field + "_timestamp"])
                feedback[arm]["gripper_diagnostics"] = gripper_diagnostics(controller)
            return observation, dict(cameras=cameras, feedback=feedback,
                                     read_monotonic=mono, read_wall_time=wall)

    return TimestampedPiperRobot(root, config, **kwargs)


def _worker(connection, root, config, kwargs, factory, return_cancel):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    robot = clock = None
    enabled = False
    last_stats = _idle_stats()
    sequence = 0
    try:
        robot = (factory or _hardware_factory)(root, config, **kwargs)
        connection.send(dict(sequence=0, ok=True, value=robot.action_hz))
        while True:
            sequence = None
            if clock is not None:
                clock.check()
            if not connection.poll(.05):
                continue
            sequence, operation, payload = connection.recv()
            if operation == "observe":
                if clock is None:
                    tick, previous, available, requested_at = 0, None, 0, time.monotonic()
                else:
                    tick, previous, available, requested_at = clock.snapshot()
                if callable(getattr(robot, "observe_with_metadata", None)):
                    observation, timing = robot.observe_with_metadata()
                else:
                    observation, timing = robot.observe(), None
                value = dict(observation=observation, request_tick=tick,
                             request_monotonic=requested_at, observed_monotonic=time.monotonic(),
                             previous_normalized_actions=previous, previous_available_actions=available)
                if timing is not None:
                    value["sensor_timing"] = timing
            elif operation == "enable":
                value = robot.enable()
                enabled = True
            elif operation == "start":
                if clock is not None:
                    raise RuntimeError("Stop the previous clock before starting another chunk")
                if payload["execute"] and not enabled:
                    raise RuntimeError("Enable the robot before starting execution")
                clock = _ChunkClock(robot.send_action if payload["execute"] else _preview,
                                    payload["result"], robot.action_hz, payload["limit_steps"])
                clock.start()
                value = clock.stats()
            elif operation == "publish":
                if clock is None:
                    raise RuntimeError("Start an asynchronous clock before publishing")
                value = clock.publish(payload["result"], payload["origin"])
            elif operation == "status":
                value = clock.stats() if clock is not None else last_stats
            elif operation == "stop":
                if clock is not None:
                    last_stats = clock.stop()
                    clock = None
                value = last_stats
            elif operation == "return_to_pose":
                if clock is not None:
                    last_stats = clock.stop()
                    clock = None
                if not enabled:
                    value = {"status": "skipped", "reason": "not_enabled"}
                else:
                    value = robot.return_to_pose(**payload, cancel=return_cancel)
            elif operation == "close":
                if clock is not None:
                    last_stats = clock.stop(raise_fault=False)
                    clock = None
                robot.close()
                robot = None
                connection.send(dict(sequence=sequence, ok=True, value=last_stats))
                break
            else:
                raise ValueError(f"Unknown Piper chunk operation: {operation}")
            connection.send(dict(sequence=sequence, ok=True, value=value))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send(dict(sequence=sequence, ok=False, error=traceback.format_exc()))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        try:
            if clock is not None:
                clock.stop(raise_fault=False)
        finally:
            try:
                if robot is not None:
                    robot.close()
            finally:
                connection.close()


class PiperChunkProcess:
    """Synchronous control RPC with a continuously running child target clock.

    ``execute=False`` validates preview targets only. ``enable`` is explicit;
    neither construction nor ``start`` enables hardware. Responses carry IDs
    so an interrupted request cannot be mistaken for a later close response.
    """
    def __init__(self, interface_root, config, *, prompt, execute_steps=32, hardware_factory=None):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.lock = threading.Lock()
        self.sequence = 0
        self.closed = False
        self.worker_failed = False
        self.replies = queue.Queue()
        self.return_cancel = context.Event()
        self.process = context.Process(target=_worker,
            args=(child, interface_root, config, dict(prompt=prompt, execute_steps=execute_steps), hardware_factory,
                  self.return_cancel),
            name="phoenix-piper-chunk-hardware", daemon=True)
        self.process.start()
        child.close()
        # recv must not be interrupted halfway through a large image payload:
        # that would lose the pipe frame boundary. Python delivers SIGINT to
        # the main thread, leaving this dedicated reader's recv uninterrupted.
        self.receiver = threading.Thread(target=self._read_replies,
                                         name="phoenix-piper-replies", daemon=True)
        self.receiver.start()
        try:
            self.action_hz = self._receive(0)
        except BaseException:
            self.close()
            raise

    def _read_replies(self):
        try:
            while True:
                self.replies.put((True, self.connection.recv()))
        except (EOFError, BrokenPipeError, OSError) as exc:
            self.replies.put((False, exc))

    def _receive(self, sequence, *, deadline=None, closing=False):
        while True:
            try:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Piper hardware did not confirm shutdown")
                try:
                    received, message = self.replies.get(timeout=.1)
                except queue.Empty:
                    if not self.process.is_alive():
                        raise RuntimeError("Piper hardware process exited without a response")
                    continue
                if not received:
                    raise message
            except KeyboardInterrupt:
                if closing:
                    continue
                raise
            actual = message.get("sequence")
            if actual is None:
                if closing:
                    continue
                self.worker_failed = True
                raise RuntimeError("Piper hardware process failed:\n" + message.get("error", "unknown worker failure"))
            if actual < sequence:
                # Drain replies belonging to a request interrupted by Ctrl+C.
                continue
            if actual != sequence:
                raise RuntimeError("Piper hardware response sequence mismatch")
            if not message["ok"]:
                self.worker_failed = True
                raise RuntimeError("Piper hardware process failed:\n" + message["error"])
            return message["value"]

    def _request(self, operation, payload=None):
        with self.lock:
            if self.closed:
                raise RuntimeError("Piper hardware process is closed")
            self.sequence += 1
            self.connection.send((self.sequence, operation, payload))
            return self._receive(self.sequence)

    def enable(self):
        return self._request("enable")

    def observe(self):
        return self._request("observe")

    def start(self, result, *, execute, limit_steps=None):
        if type(execute) is not bool:
            raise TypeError("execute must be a bool")
        _plan(result)
        return self._request("start", dict(result=result, execute=execute, limit_steps=limit_steps))

    def publish(self, result, origin):
        _plan(result)
        return self._request("publish", dict(result=result, origin=origin))

    def status(self):
        return self._request("status")

    def stop(self):
        return self._request("stop")

    def return_to_pose(self, target="initial", duration=5.):
        from wam.inference.piper_return import validate_return_options
        validate_return_options(target, duration)
        with self.lock:
            if self.closed:
                raise RuntimeError("Piper hardware process is closed")
            self.return_cancel.clear()
            self.sequence += 1
            try:
                self.connection.send((self.sequence, "return_to_pose", dict(target=target, duration=duration)))
                return self._receive(self.sequence, deadline=time.monotonic() + duration + 8.)
            except KeyboardInterrupt:
                # The worker ignores terminal SIGINT so the first Ctrl+C can
                # return home. A second Ctrl+C cancels through shared memory,
                # without waiting for its synchronous RPC loop to receive.
                self.return_cancel.set()
                print("PIPER_RETURN_ABORT requested; stopping return and disabling", flush=True)
                return self._receive(self.sequence, deadline=time.monotonic() + 5., closing=True)
            except BaseException:
                self.return_cancel.set()
                raise

    def close(self):
        self.return_cancel.set()
        with self.lock:
            if self.closed:
                return
            deadline = time.monotonic() + 15
            try:
                if self.process.is_alive():
                    self.sequence += 1
                    self.connection.send((self.sequence, "close", None))
                    self._receive(self.sequence, deadline=deadline, closing=True)
            except (EOFError, BrokenPipeError, OSError, RuntimeError, TimeoutError) as exc:
                if not self.worker_failed:
                    LOG.warning("Piper chunk worker shutdown was not acknowledged: %s", exc)
            finally:
                self.closed = True
                self.connection.close()
                while self.process.is_alive() and time.monotonic() < deadline:
                    try:
                        self.process.join(timeout=.1)
                    except KeyboardInterrupt:
                        continue
                if self.process.is_alive():
                    # Never terminate drivers in the middle of their cleanup.
                    raise RuntimeError("Piper hardware process is still shutting down")
                self.receiver.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
