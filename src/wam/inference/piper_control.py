"""Optional Piper MIT transport, shared by inference and dataset replay.

Ported from sink-bot-control_bk mit_dev (3058b10f6aa0), without switching
that checkout or replacing its feedback, enable, stop or gripper handling.
This module imports no model, CUDA context, SDK or hardware driver.
MIT parameter defaults are synced from teleop_inference in the shared JSON profile.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


_MIT_DEFAULTS_PATH = Path(__file__).resolve().parents[3] / "configs/inference/piper_mit_defaults.json"
_MIT_DEFAULTS = json.loads(_MIT_DEFAULTS_PATH.read_text())["mit"]


@dataclass(frozen=True)
class ControlSettings:
    mode: str = "joint"
    kp: float = _MIT_DEFAULTS["kp"]
    kd: float = _MIT_DEFAULTS["kd"]
    robot_rate: float = _MIT_DEFAULTS["robot_rate"]  # MIT driver loop only
    vel_ref: float = _MIT_DEFAULTS["vel_ref"]
    t_ref: float = _MIT_DEFAULTS["t_ref"]
    speed_limit: int = _MIT_DEFAULTS["speed_limit"]

    def __post_init__(self):
        if self.mode not in ("joint", "mit"):
            raise ValueError("control mode must be joint or mit")
        for name, lo, hi in (("kp", 0, 500), ("kd", -5, 5), ("robot_rate", 1, 1000),
                             ("vel_ref", -45, 45), ("t_ref", -8, 8)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"MIT {name} must be finite in [{lo}, {hi}]")
        if type(self.speed_limit) is not int or not 0 <= self.speed_limit <= 100:
            raise ValueError("MIT speed_limit must be an integer in [0, 100]")

    @classmethod
    def from_dict(cls, values=None):
        if values is None:
            return cls()
        if not isinstance(values, dict) or set(values) - set(cls.__dataclass_fields__):
            raise ValueError("robot_control expects mode, kp, kd, robot_rate, vel_ref, t_ref, speed_limit")
        return cls(**values)

    @property
    def controller_hz(self):
        return self.robot_rate if self.mode == "mit" else 100.0

    def audit(self):
        return {"mode": self.mode, "controller_hz_requested": self.controller_hz,
                "joint_api": "JointMitCtrl" if self.mode == "mit" else "JointCtrl",
                "mit": {"kp": self.kp, "kd": self.kd, "vel_ref": self.vel_ref,
                        "t_ref": self.t_ref, "speed_limit": self.speed_limit} if self.mode == "mit" else None}


def add_control_arguments(parser):
    defaults = ControlSettings()
    parser.add_argument("--control-mode", choices=("joint", "mit"), default=defaults.mode,
                        help="Hardware joint control; independent of sync/async/rtc scheduling")
    parser.add_argument("--mit-kp", type=float, default=defaults.kp, help="MIT stiffness, 0..500")
    parser.add_argument("--mit-kd", type=float, default=defaults.kd, help="MIT damping, SDK range -5..5")
    parser.add_argument("--mit-robot-rate", type=float, default=defaults.robot_rate,
                        help="MIT driver loop requested Hz, 1..1000; action clock unchanged")
    parser.add_argument("--mit-vel-ref", type=float, default=defaults.vel_ref, help="MIT velocity reference rad/s, -45..45")
    parser.add_argument("--mit-t-ref", type=float, default=defaults.t_ref, help="MIT torque reference, SDK encoding range -8..8")
    parser.add_argument("--mit-speed-limit", type=int, default=defaults.speed_limit, help="MotionCtrl_2 speed percentage, 0..100")


def settings_from_args(args):
    return ControlSettings(mode=args.control_mode, kp=args.mit_kp, kd=args.mit_kd,
                           robot_rate=args.mit_robot_rate, vel_ref=args.mit_vel_ref,
                           t_ref=args.mit_t_ref, speed_limit=args.mit_speed_limit)


def with_control_settings(config, settings):
    return {**config, "robot_control": asdict(settings)}


def controller_class(base, sdk_class, settings):
    """Return the original class for joint mode, an isolated subclass for MIT.

    Keep the driver's JOINT_SPACE target bookkeeping (degrees, hold, limits,
    synchronization). Override only its gated joint sender with mit_dev's
    radians/MOVEM/0xAD packets. Creating/selecting this class sends nothing.
    """
    if settings.mode == "joint":
        return base
    if not callable(getattr(sdk_class, "JointMitCtrl", None)):
        raise RuntimeError("Selected Piper SDK has no JointMitCtrl; MIT mode is unavailable")

    class MitPiperController(base):
        def _send_joint_command(self, joint_angles):
            positions = [math.radians(float(value)) for value in joint_angles]
            if len(positions) != 6 or any(not math.isfinite(v) or abs(v) > 12.5 for v in positions):
                raise ValueError("MIT requires six finite joint positions within SDK radian range")
            self.piper.MotionCtrl_2(0x01, 0x04, settings.speed_limit, 0xAD)
            for index, position in enumerate(positions, 1):
                self.piper.JointMitCtrl(index, position, settings.vel_ref, settings.kp, settings.kd, settings.t_ref)

    return MitPiperController
