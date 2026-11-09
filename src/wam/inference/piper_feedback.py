"""Receive-only normalization of explicitly configured Piper feedback IDs."""
from copy import copy
import math


def gripper_diagnostics(controller):
    """Copy cached SDK feedback only; never query CAN or change gripper state.

    This has its own timestamp: it is not atomic with the joint snapshot.
    Missing diagnostics must not prevent otherwise valid robot observations.
    """
    try:
        message = controller.piper.GetArmGripperMsgs()
        state = message.gripper_state
        code = int(state.status_code)
        def finite(value):
            value = float(value)
            return value if math.isfinite(value) else None
        flags = ("voltage_low", "motor_overheating", "driver_overcurrent",
                 "driver_overheating", "sensor_abnormal", "driver_error",
                 "driver_enabled", "homed")
        return dict(available=True, timestamp=finite(message.time_stamp),
                    hz=finite(message.Hz), stroke_mm=finite(state.grippers_angle / 1000),
                    effort_nm=finite(state.grippers_effort / 1000), status_code=code,
                    **{flag: bool(code & (1 << bit)) for bit, flag in enumerate(flags)})
    except Exception as exc:
        return dict(available=False, error=f"{type(exc).__name__}: {exc}")


def feedback_id_offsets(config, interfaces):
    offsets = config.get("feedback_id_offsets", {})
    if not isinstance(offsets, dict) or set(offsets) - set(interfaces):
        raise ValueError("feedback_id_offsets must map configured CAN interfaces to integer offsets")
    if any(type(value) is not int or not 0 <= value <= 255 for value in offsets.values()):
        raise ValueError("Feedback ID offsets must be integers in 0..255")
    return {name: offsets.get(name, 0) for name in interfaces}


class FeedbackIdAdapter:
    """Translate only the eight state/pose/joint/gripper receive frames.

    The driver callback still performs all normal decoding and validation.
    This adapter has no CAN send interface and never changes the wire frame.
    """
    def __init__(self, callback, offset):
        if type(offset) is not int or not 0 <= offset <= 255:
            raise ValueError("Feedback ID offset must be an integer in 0..255")
        self.callback = callback
        self.offset = offset
        self.mapped_frames = 0

    def __call__(self, frame):
        if (self.offset and not frame.is_extended_id and not frame.is_remote_frame
                and not frame.is_error_frame and len(frame.data) == 8
                and 0x2A1 + self.offset <= frame.arbitration_id <= 0x2A8 + self.offset):
            frame = copy(frame)
            frame.arbitration_id -= self.offset
            self.mapped_frames += 1
        return self.callback(frame)
