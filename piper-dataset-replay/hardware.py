"""Hardware-only adapter. No Phoenix or pi policy is imported or executed."""
import fcntl
from dataclasses import asdict
from pathlib import Path
import sys
import threading
import time

import numpy as np
from paths import default_project_root


class Cancelled(RuntimeError):
    pass


class ControlSelection:
    """Share only lightweight control configuration/transport, never a policy."""
    def _init_control(self, project, values=None):
        sys.path.insert(0, str(Path(project) / "src"))
        from wam.inference.piper_control import ControlSettings, controller_class
        self.control_type, self.controller_class = ControlSettings, controller_class
        self.control = ControlSettings.from_dict(values)

    def validate_control(self, values):
        return self.control_type.from_dict(values)

    def configure_control(self, values):
        if self.connected or getattr(self, "controllers", []):
            raise ValueError("请先断开设备，再修改控制模式或 MIT 参数")
        self.control = self.validate_control(values)

    @property
    def control_settings(self):
        return asdict(self.control)


class DemoRobot(ControlSelection):
    kind = "demo"

    def __init__(self, project=None):
        self._init_control(project or default_project_root())
        self.connected = self.enabled = False
        self._state = np.zeros(14, dtype=float)
        self._target = self._state.copy()
        self._last = time.monotonic()
        self.camera_status = {}
        self.lock = threading.Lock()
        self.last_feedback = {"simulated": True}

    def connect(self):
        self.connected = True

    def enable(self, cancel):
        if cancel.is_set():
            raise Cancelled("已取消启动")
        self.enabled = True

    def state(self):
        if not self.connected:
            return None
        with self.lock:
            now = time.monotonic()
            self._state += (self._target-self._state) * (1-np.exp(-(now-self._last)/.12))
            self._last = now
            return self._state.copy()

    def send(self, target):
        if not self.enabled:
            raise RuntimeError("演示设备未启动")
        with self.lock:
            self._target = np.asarray(target, dtype=float).copy()
        return np.asarray(target).copy()

    def hold(self):
        value = self.state()
        if value is not None:
            with self.lock:
                self._target = value.copy()
        return value

    def disable(self):
        self.hold()
        self.enabled = False

    def capture_frames(self):
        return {}

    def close(self):
        self.disable()
        self.connected = False


class PiperRobot(ControlSelection):
    kind = "hardware"

    def __init__(self, project, driver_root, config):
        self._init_control(project, config.get("robot_control"))
        # Share CAN helpers and the optional MIT sender; no policy/model import.
        sys.path.insert(0, str(Path(project) / "src"))
        from wam.inference.piper_can import resolve_can_config
        from wam.inference.piper_feedback import FeedbackIdAdapter, feedback_id_offsets, gripper_diagnostics
        self.resolve = resolve_can_config
        self.adapter_class = FeedbackIdAdapter
        self.offsets = feedback_id_offsets
        self.gripper_diagnostics = gripper_diagnostics
        self.last_feedback = {}
        self.root, self.config = Path(driver_root).resolve(), config
        self.controllers, self.bindings, self.locks = [], [], []
        self.cameras = None
        self.connected = self.enabled = self.enable_attempted = False
        self.camera_status = {}

    def connect(self):
        if self.connected:
            return
        cfg = self.resolve(self.config)
        self.config = cfg
        names = (cfg["left_can"], cfg["right_can"])
        offsets = self.offsets(cfg, names)
        try:
            for name in names:
                handle = open(f"/tmp/pi05_checkpoint_{name}.lock", "a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    raise RuntimeError(f"{name} 正由其他控制程序使用，请先从原终端正常停止")
                self.locks.append(handle)
            sys.path[:0] = [str(self.root), str(self.root / "examples"), str(self.root / "piper_sdk")]
            from piper_controller import PiperController
            from piper_sdk import C_PiperInterface_V2
            from common.multi_camera import MultiCameraManager
            controller_type = self.controller_class(PiperController, C_PiperInterface_V2, self.control)
            for name in names:
                interface = None
                if offsets[name]:
                    interface = C_PiperInterface_V2(name, start_sdk_joint_limit=True, start_sdk_gripper_limit=True)
                    if interface.get_connect_status():
                        raise RuntimeError(f"{name} SDK 已连接，无法安装接收 ID 适配")
                    bus = interface.GetCanBus()
                    adapter = self.adapter_class(bus.callback_function, offsets[name])
                    bus.callback_function = adapter
                    self.bindings.append((interface, bus, adapter))
                controller = controller_type(can_interface=name, robot_rate=self.control.controller_hz,
                                             control_mode=PiperController.ControlMode.JOINT_SPACE, auto_enable=False)
                self.controllers.append(controller)
                if interface is not None and controller.piper is not interface:
                    raise RuntimeError("SDK 实例不匹配")
            deadline = time.monotonic() + 10
            while True:
                try:
                    self.state()
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.05)
            self.cameras = MultiCameraManager(cfg["cameras"], fps=30, resolution=(640, 480), record_depth=False)
            self.camera_status = self.cameras.start()
            self.connected = True
        except BaseException:
            self.close()
            raise

    def state(self):
        if len(self.controllers) != 2:
            return None
        values, feedback = [], {}
        for arm, c in zip(("left", "right"), self.controllers):
            snapshot = c.get_feedback_snapshot()
            if snapshot is None or any(not np.isfinite(snapshot[k]) or snapshot[k] <= 0 for k in
                                       ("joint_timestamp", "gripper_timestamp", "joint_hz", "gripper_hz")):
                raise RuntimeError(f"{c.can_interface} 尚无有效关节/夹爪反馈")
            if any(time.time()-snapshot[k] > 1 for k in ("joint_timestamp", "gripper_timestamp")):
                raise RuntimeError(f"{c.can_interface} 关节/夹爪反馈已中断")
            values.extend(np.deg2rad(snapshot["joints_deg"]))
            values.append(float(snapshot["gripper"]))
            feedback[arm] = {key: snapshot.get(key) for key in
                             ("joint_timestamp", "gripper_timestamp", "joint_hz", "gripper_hz", "read_monotonic")}
            feedback[arm]["gripper_diagnostics"] = self.gripper_diagnostics(c)
        array = np.asarray(values, dtype=float)
        if array.shape != (14,) or not np.isfinite(array).all():
            raise RuntimeError("关节反馈无效")
        self.last_feedback = feedback
        return array

    def enable(self, cancel):
        def check():
            if cancel.is_set():
                raise Cancelled("已取消启动")
        self.enable_attempted = True
        try:
            for c in self.controllers:
                check()
                if not c.enable_robot(sync_targets=True, progress_check=check):
                    raise RuntimeError(f"{c.can_interface} 使能失败")
                c.start_control_loop()
            check()
            self.enabled = True
        except BaseException:
            self.disable()
            raise

    def send(self, target):
        target = np.asarray(target, dtype=float)
        if not self.enabled or target.shape != (14,) or not np.isfinite(target).all():
            raise RuntimeError("设备未使能或回放目标无效")
        applied = []
        for controller, offset in zip(self.controllers, (0, 7)):
            controller.set_target_joint_angles(np.rad2deg(target[offset:offset+6]))
            controller.set_gripper_open_value(float(target[offset+6]))
            applied.extend(np.deg2rad(controller.get_target_joint_angles()))
            applied.append(float(controller.get_gripper_open_value()))
        return np.asarray(applied)

    def hold(self):
        for controller in self.controllers:
            if self.enabled:
                controller.hold_position()
        return self.state()

    def disable(self):
        # Latch both arms before waiting for either arm's disable replies.
        if not self.enable_attempted:
            return
        errors = []
        for controller in self.controllers:
            try:
                if not controller.latch_emergency_stop():
                    errors.append(controller.can_interface + " quick-stop 未确认")
            except Exception as exc:
                errors.append(str(exc))
        for controller in self.controllers:
            try:
                if not controller.finish_emergency_disable():
                    errors.append(controller.can_interface + " 失能未确认")
            except Exception as exc:
                errors.append(str(exc))
        self.enabled = False
        if errors:
            raise RuntimeError("; ".join(errors))

    def capture_frames(self):
        return self.cameras.capture_frames() if self.cameras is not None else {}

    def close(self):
        errors = []
        try:
            self.disable()
        except Exception as exc:
            errors.append(str(exc))
        if self.cameras is not None:
            try:
                self.cameras.stop()
            except Exception as exc:
                errors.append(str(exc))
            self.cameras = None
        self.camera_status = {}
        for controller in self.controllers:
            if not controller.disconnect_without_homing(require_disabled=self.enable_attempted, feedback_timeout_s=1):
                errors.append("断开未确认: " + controller.can_interface)
        for interface, bus, adapter in self.bindings:
            if not self.enable_attempted and interface.get_connect_status():
                interface.DisconnectPort()
            if not interface.get_connect_status() and bus.callback_function is adapter:
                bus.callback_function = adapter.callback
        # Retain ownership if a live enabled/unknown controller could not close.
        if errors:
            raise RuntimeError("; ".join(errors))
        for handle in self.locks:
            handle.close()
        self.locks.clear()
        self.controllers.clear()
        self.bindings.clear()
        self.connected = self.enabled = self.enable_attempted = False
