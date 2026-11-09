"""MIT routing and CLI validation. Never connects a CAN device or moves a robot."""
from contextlib import redirect_stderr
from dataclasses import asdict
import io
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from wam.inference.piper_control import (
    ControlSettings, controller_class, settings_from_args, with_control_settings,
)
from tools.run_piper_policy import parse_args


class ControlTests(unittest.TestCase):
    def test_joint_returns_unmodified_driver(self):
        class Driver:
            pass
        self.assertIs(controller_class(Driver, object, ControlSettings()), Driver)
        self.assertEqual(ControlSettings().controller_hz, 100)

    def test_mit_exact_packets_radians_joint_order_and_custom_parameters(self):
        class Driver:
            pass
        sdk = Mock(JointMitCtrl=Mock())
        settings = ControlSettings(mode="mit", kp=21, kd=1.3, robot_rate=220,
                                   vel_ref=.2, t_ref=-.15, speed_limit=55)
        adapted = controller_class(Driver, sdk, settings)
        robot = adapted()
        robot.piper = sdk
        sdk.assert_not_called()
        robot._send_joint_command([0, 30, -90, 45, -45, 180])
        sdk.MotionCtrl_2.assert_called_once_with(1, 4, 55, 0xAD)
        self.assertEqual(sdk.JointMitCtrl.call_count, 6)
        for index, (call, expected) in enumerate(zip(sdk.JointMitCtrl.call_args_list,
                                                  [0, math.pi/6, -math.pi/2, math.pi/4, -math.pi/4, math.pi]), 1):
            joint, position, velocity, kp, kd, torque = call.args
            self.assertEqual((joint, velocity, kp, kd, torque), (index, .2, 21, 1.3, -.15))
            self.assertAlmostEqual(position, expected)
        sdk.JointCtrl.assert_not_called()

    def test_invalid_parameters_and_targets_do_not_send_partial_commands(self):
        for values in ({"mode": "typo"}, {"kp": float("nan")}, {"kp": 501}, {"kd": 5.1},
                       {"robot_rate": 0}, {"vel_ref": 46}, {"t_ref": 8.1},
                       {"speed_limit": 50.5}, {"speed_limit": True}, {"kp": "15"}, {"extra": 2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ControlSettings.from_dict(values)
        robot = controller_class(object, Mock(JointMitCtrl=Mock()), ControlSettings(mode="mit"))()
        robot.piper = Mock()
        for values in ([0]*5, [0]*7, [0]*5+[float("nan")], [0]*5+[800]):
            with self.assertRaises(ValueError):
                robot._send_joint_command(values)
        self.assertEqual(robot.piper.mock_calls, [])
        with self.assertRaisesRegex(RuntimeError, "JointMitCtrl"):
            controller_class(object, object, ControlSettings(mode="mit"))

    def test_selection_is_independent_of_scheduling_and_serializable(self):
        self.assertEqual(settings_from_args(parse_args([])).mode, "joint")
        for mode in ("sync", "async", "rtc"):
            args = parse_args(["--mode", mode, "--control-mode", "mit", "--mit-kp", "18",
                               "--mit-kd", "1.2", "--mit-robot-rate", "240", "--mit-vel-ref", ".1",
                               "--mit-t-ref", ".05", "--mit-speed-limit", "50"])
            settings = settings_from_args(args)
            config = with_control_settings({"left_can": "left"}, settings)
            self.assertEqual(ControlSettings.from_dict(config["robot_control"]), settings)
            self.assertEqual(asdict(settings), dict(mode="mit", kp=18., kd=1.2, robot_rate=240.,
                                                  vel_ref=.1, t_ref=.05, speed_limit=50))
            self.assertEqual(args.replan_steps, 10)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--mit-kp", "nan"])


class InstalledDriverTests(unittest.TestCase):
    def test_real_driver_gate_hold_and_gripper_are_inherited(self):
        root = Path("/home/midea/Desktop/piper_tele/sink-bot-control_bk/example_agilex")
        if not (root / "piper_controller.py").is_file():
            self.skipTest("Optional local hardware driver source is unavailable")
        with patch.object(sys, "path", [str(root), str(root / "piper_sdk"), *sys.path]):
            from piper_controller import PiperController
            from piper_sdk import C_PiperInterface_V2
        adapted = controller_class(PiperController, C_PiperInterface_V2, ControlSettings(mode="mit"))
        sdk = Mock()
        # Block SDK construction even if a future driver change bypasses initialization.
        with patch("piper_controller.C_PiperInterface_V2", side_effect=AssertionError("No real SDK construction")), \
             patch.object(PiperController, "_initialize_robot", lambda obj: setattr(obj, "piper", sdk)), \
             patch.object(PiperController, "cleanup"):
            robot = adapted(can_interface="fake", robot_rate=210, auto_enable=False)
            robot.JOINT_LIMITS = np.array([[-180., 180.]] * 6)  # fixture replaces SDK range query
            robot.cleanup = Mock()  # also prevent destructor work after patch scope ends
            self.assertEqual(sdk.mock_calls, [])
            self.assertIs(adapted.enable_robot, PiperController.enable_robot)
            self.assertIs(adapted.hold_position, PiperController.hold_position)
            self.assertIs(adapted.latch_emergency_stop, PiperController.latch_emergency_stop)
            self.assertIs(adapted.finish_emergency_disable, PiperController.finish_emergency_disable)
            with patch("piper_controller.time.sleep", side_effect=lambda _: robot.running.clear()):
                robot.control_loop()
            self.assertEqual(sdk.mock_calls, [])  # read-only gate remains closed
            robot.set_target_joint_angles([10, 20, -30, 40, 50, 60])
            robot.set_gripper_open_value(.5)
            robot._set_command_stream_enabled(True)
            robot.running.set()
            with patch("piper_controller.time.sleep", side_effect=lambda _: robot.running.clear()):
                robot.control_loop()
            sdk.MotionCtrl_2.assert_called_once_with(1, 4, 60, 0xAD)
            self.assertEqual(sdk.JointMitCtrl.call_count, 6)
            self.assertAlmostEqual(sdk.JointMitCtrl.call_args_list[0].args[1], math.radians(10))
            sdk.GripperCtrl.assert_called_once_with(47500, 1000, 1, 0)
            sdk.JointCtrl.assert_not_called()
            robot._set_command_stream_enabled(False)
            del robot


if __name__ == "__main__":
    unittest.main()
