"""Phoenix-native Piper transport: observations in, absolute joint targets out.

Only hardware drivers are imported from the robot repository. Policy sampling,
normalization and action cadence belong to Phoenix; no pi0.5 client is used.
"""
import fcntl
import logging
from pathlib import Path
import re
import sys
import time

import numpy as np

from wam.inference.piper import IMAGE_KEYS, validate_observation
from wam.inference.piper_feedback import FeedbackIdAdapter, feedback_id_offsets
from wam.inference.piper_can import resolve_can_config
from wam.inference.piper_control import ControlSettings, controller_class

LOG = logging.getLogger(__name__)


def build_robot_observation(frames, snapshots, mapping, prompt):
    if len(snapshots) != 2:
        raise ValueError("Expected left and right arm feedback")
    state = []
    for arm, snapshot in zip(("left", "right"), snapshots):
        if snapshot is None:
            raise RuntimeError(f"Missing Piper {arm} arm feedback")
        feedback = {field: snapshot[field] for field in
                    ("joint_timestamp", "gripper_timestamp", "joint_hz", "gripper_hz")}
        if any(not np.isfinite(value) or value <= 0 for value in feedback.values()):
            raise RuntimeError(f"Piper {arm} arm has not received joint/gripper feedback yet: {feedback}")
        joints = np.asarray(snapshot["joints_deg"], dtype=np.float32)
        if joints.shape != (6,):
            raise ValueError("Expected six joints per arm")
        state.extend(np.deg2rad(joints))
        state.append(float(snapshot["gripper"]))
    if len(mapping) != 3 or set(mapping.values()) != set(IMAGE_KEYS):
        raise ValueError("Expected exactly three model cameras")
    observation = {"state": np.asarray(state, dtype=np.float32), "prompt": prompt}
    for camera, key in mapping.items():
        bgr = frames.get(camera, {}).get("rgb")
        if bgr is None:
            raise RuntimeError(f"Missing camera frame: {camera}")
        # The existing RealSense driver provides bgr8; Phoenix expects RGB.
        observation[key] = np.ascontiguousarray(np.asarray(bgr)[..., ::-1])
    return validate_observation(observation)


class PhoenixPiperRobot:
    def __init__(self, interface_root, config, *, prompt, execute_steps=10, action_hz=30.):
        if not 1 <= execute_steps <= 32 or not np.isfinite(action_hz) or action_hz <= 0:
            raise ValueError("Expected execute_steps in 1..32 and positive action_hz")
        self.config = config
        self.control = ControlSettings.from_dict(config.get("robot_control"))
        self.prompt = prompt
        self.execute_steps = execute_steps
        self.action_hz = action_hz
        self.controllers = []
        self.feedback_bindings = []
        self.cameras = None
        self.locks = []
        self.enable_attempted = False
        self.enabled = False
        self.initial_joints_deg = None
        try:
            self._connect(Path(interface_root).resolve())
        except BaseException:
            self.close()
            raise

    def _connect(self, root):
        # Recheck after model loading in case USB names changed since preflight.
        self.config = resolve_can_config(self.config)
        left, right = self.config["left_can"], self.config["right_can"]
        if left == right or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in (left, right)):
            raise ValueError("Expected distinct CAN interface names")
        offsets = feedback_id_offsets(self.config, (left, right))
        for name in (left, right):
            # Use the hardware's existing lock namespace so other controllers
            # cannot command the same CAN bus. This imports no policy client.
            handle = open(f"/tmp/pi05_checkpoint_{name}.lock", "a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise RuntimeError(f"Another controller is using {name}")
            self.locks.append(handle)
        sys.path[:0] = [str(root), str(root / "examples"), str(root / "piper_sdk")]
        from piper_controller import PiperController
        from piper_sdk import C_PiperInterface_V2
        from common.multi_camera import MultiCameraManager
        controller_type = controller_class(PiperController, C_PiperInterface_V2, self.control)
        for name in (left, right):
            interface = None
            if offsets[name]:
                # The SDK caches one interface per CAN name. Prepare its receive
                # callback before PiperController starts the SDK reader thread.
                interface = C_PiperInterface_V2(
                    name, start_sdk_joint_limit=True, start_sdk_gripper_limit=True)
                if interface.get_connect_status():
                    raise RuntimeError(f"Cannot install feedback adapter on an already connected SDK: {name}")
                bus = interface.GetCanBus()
                adapter = FeedbackIdAdapter(bus.callback_function, offsets[name])
                self.feedback_bindings.append((interface, bus, adapter))
                bus.callback_function = adapter
                LOG.warning("%s: receive-only feedback IDs 0x%03X..0x%03X -> 0x2A1..0x2A8; transmit IDs unchanged",
                            name, 0x2A1 + offsets[name], 0x2A8 + offsets[name])
            self.controllers.append(controller_type(
                can_interface=name, robot_rate=self.control.controller_hz,
                control_mode=PiperController.ControlMode.JOINT_SPACE, auto_enable=False))
            if interface is not None and self.controllers[-1].piper is not interface:
                raise RuntimeError("Piper SDK no longer shares interfaces by CAN name; feedback adapter was not installed")
        self.cameras = MultiCameraManager(self.config["cameras"], fps=30,
                                         resolution=(640, 480), record_depth=False)
        started = self.cameras.start()
        if not all(started.get(key) for key in self.config["camera_mapping"]):
            raise RuntimeError(f"Camera startup failed: {started}")
        # Wait for the first real frames/feedback. No inference-age cutoff.
        deadline = time.monotonic() + 10
        while True:
            try:
                self.observe()
                break
            except RuntimeError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{exc}; left={left}, right={right}, feedback_id_offsets={offsets}") from exc
                time.sleep(.05)

    def observe(self):
        return build_robot_observation(
            self.cameras.capture_frames(),
            [controller.get_feedback_snapshot() for controller in self.controllers],
            self.config["camera_mapping"], self.prompt)

    def enable(self):
        # Capture immediately before enabling, after model warmup. Returning
        # later needs joint feedback only, even if a camera has disconnected.
        initial_joints = self.read_joint_positions()
        self.enable_attempted = True
        for controller in self.controllers:
            if not controller.enable_robot(sync_targets=True):
                raise RuntimeError("Could not enable Piper arms")
            controller.start_control_loop()
        self.enabled = True
        self.initial_joints_deg = initial_joints

    def read_joint_positions(self):
        joints = []
        for arm, controller in zip(("left", "right"), self.controllers):
            snapshot = controller.get_feedback_snapshot()
            if snapshot is None:
                raise RuntimeError(f"Missing {arm} joint feedback for return")
            values = np.asarray(snapshot["joints_deg"], dtype=np.float64)
            timestamp = float(snapshot["joint_timestamp"])
            hz = float(snapshot["joint_hz"])
            age = time.time() - timestamp
            if (values.shape != (6,) or not np.isfinite(values).all()
                    or not np.isfinite([timestamp, hz]).all() or hz <= 0
                    or timestamp <= 0 or not -.1 <= age <= 1.):
                raise RuntimeError(f"Invalid/stale {arm} joint feedback for return")
            joints.extend(values)
        if len(joints) != 12:
            raise RuntimeError("Return requires feedback from both arms")
        return np.asarray(joints, dtype=np.float64)

    def return_to_pose(self, target="initial", duration=5., *, cancel=None):
        from wam.inference.piper_return import return_joints, validate_return_options
        validate_return_options(target, duration)
        if not self.enabled or target == "none":
            return {"status": "skipped", "reason": "not_enabled" if not self.enabled else "disabled"}
        destination = self.initial_joints_deg if target == "initial" else np.zeros(12)
        if destination is None:
            raise RuntimeError("Initial joint pose was not captured before enable")
        # Reject an unreachable destination instead of silently accepting SDK
        # clamping and reporting that the requested pose was reached.
        for controller, offset in zip(self.controllers, (0, 6)):
            limits = np.asarray(controller.JOINT_LIMITS)
            if np.any(destination[offset:offset+6] < limits[:, 0]) or np.any(destination[offset:offset+6] > limits[:, 1]):
                raise ValueError("Return pose lies outside the controller joint limits")

        def send_joints(joints):
            if not self.enabled:
                raise RuntimeError("Robot became disabled during return")
            for controller, offset in zip(self.controllers, (0, 6)):
                controller.set_target_joint_angles(joints[offset:offset+6])

        result = return_joints(self.read_joint_positions, send_joints, destination,
                               duration=duration, hz=self.action_hz, cancel=cancel)
        return {**result, "target": target, "gripper_targets": "unchanged"}

    def execute(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (32, 14) or not np.isfinite(actions).all():
            raise ValueError("Expected finite Phoenix actions (32,14)")
        if not self.enabled:
            raise RuntimeError("Enable the arms before execution")
        for action in actions[:self.execute_steps]:
            start = time.monotonic()
            self.send_action(action)
            time.sleep(max(0., 1 / self.action_hz - (time.monotonic() - start)))

    def send_action(self, action):
        action = np.asarray(action)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected one finite joint14 action")
        if not self.enabled:
            raise RuntimeError("Enable the arms before execution")
        for controller, offset in zip(self.controllers, (0, 7)):
            controller.set_target_joint_angles(np.rad2deg(action[offset:offset+6]))
            controller.set_gripper_open_value(float(action[offset+6]))

    def close(self):
        self.enabled = False
        if self.enable_attempted:
            # On exit, stop both arms before waiting for motor-disable replies.
            for controller in self.controllers:
                try:
                    controller.latch_emergency_stop()
                except Exception:
                    LOG.exception("Piper quick stop failed")
            for controller in self.controllers:
                try:
                    if not controller.finish_emergency_disable():
                        LOG.error("Piper motor disable was not confirmed")
                except Exception:
                    LOG.exception("Piper disable failed")
        if self.cameras is not None:
            try:
                self.cameras.stop()
            except Exception:
                LOG.exception("Camera shutdown failed")
        for controller in self.controllers:
            try:
                if not controller.disconnect_without_homing(
                        require_disabled=self.enable_attempted, feedback_timeout_s=1):
                    LOG.error("Piper disconnect was not confirmed")
            except Exception:
                LOG.exception("Piper disconnect failed")
        for interface, bus, adapter in getattr(self, "feedback_bindings", []):
            try:
                # Also close a prepared SDK if controller construction failed.
                owned = any(getattr(controller, "piper", None) is interface
                            for controller in self.controllers)
                if not self.enable_attempted and (interface.get_connect_status() or not owned):
                    interface.DisconnectPort()
                if not interface.get_connect_status() and bus.callback_function is adapter:
                    bus.callback_function = adapter.callback
            except Exception:
                LOG.exception("Piper feedback adapter cleanup failed")
        for handle in self.locks:
            handle.close()
        self.locks.clear()
