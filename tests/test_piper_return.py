"""Return-on-interrupt checks with synthetic feedback; no hardware access."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from wam.inference.piper_return import return_joints


class VirtualClock:
    def __init__(self):
        self.now = 0.
        self.cancelled = False

    def monotonic(self):
        return self.now

    def is_set(self):
        return self.cancelled

    def wait(self, seconds):
        self.now += seconds
        return self.cancelled


class ReturnTrajectoryTests(unittest.TestCase):
    def test_return_starts_at_feedback_is_smooth_and_confirms_arrival(self):
        clock = VirtualClock()
        state = np.linspace(-40., 50., 12)
        initial = state.copy()
        destination = -initial
        sent = []
        def send(value):
            sent.append((clock.now, value.copy()))
            state[:] = value
        with patch('wam.inference.piper_return.time.monotonic', clock.monotonic):
            result = return_joints(lambda: state.copy(), send, destination, duration=2., hz=30., cancel=clock)
        self.assertEqual(result['status'], 'completed')
        np.testing.assert_array_equal(sent[0][1], initial)
        np.testing.assert_array_equal(sent[-1][1], destination)
        self.assertGreaterEqual(result['elapsed_seconds'], 2.2)
        trajectory = np.stack([value for _, value in sent])
        self.assertTrue(np.all(trajectory >= np.minimum(initial, destination) - 1e-8))
        self.assertTrue(np.all(trajectory <= np.maximum(initial, destination) + 1e-8))
        self.assertLess(np.max(np.abs(trajectory[1] - trajectory[0])), .01)

    def test_cancellation_stops_target_generation(self):
        clock = VirtualClock()
        sent = []
        def send(value):
            sent.append(value.copy())
            if len(sent) == 4:
                clock.cancelled = True
        with patch('wam.inference.piper_return.time.monotonic', clock.monotonic):
            result = return_joints(lambda: np.ones(12), send, np.zeros(12), cancel=clock)
        self.assertEqual(result['status'], 'aborted')
        self.assertEqual(len(sent), 4)

    def test_stationary_feedback_cannot_be_reported_as_success(self):
        clock = VirtualClock()
        with patch('wam.inference.piper_return.time.monotonic', clock.monotonic):
            with self.assertRaisesRegex(TimeoutError, 'not reached'):
                return_joints(lambda: np.full(12, 30.), lambda value: None, np.zeros(12),
                              duration=.1, settle_timeout=.2, cancel=clock)

    def test_feedback_failure_and_invalid_options_send_no_new_target(self):
        sent = []
        def missing():
            raise RuntimeError('feedback lost')
        with self.assertRaisesRegex(RuntimeError, 'feedback lost'):
            return_joints(missing, sent.append, np.zeros(12))
        for duration in (0, -1, float('nan'), float('inf')):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                return_joints(lambda: np.zeros(12), sent.append, np.ones(12), duration=duration)
        self.assertEqual(sent, [])


class RobotReturnTests(unittest.TestCase):
    def robot(self, mode='joint'):
        from wam.inference.piper_robot import PhoenixPiperRobot
        import time
        state = [np.arange(6, dtype=float) + 10, np.arange(6, dtype=float) - 20]
        sent = [[], []]
        robot = PhoenixPiperRobot.__new__(PhoenixPiperRobot)
        robot.enabled = True
        robot.action_hz = 30.
        robot.control = SimpleNamespace(mode=mode)
        robot.initial_joints_deg = np.zeros(12)
        def update(index, values):
            state[index][:] = values
            sent[index].append(np.array(values, copy=True))
        robot.controllers = [SimpleNamespace(
            JOINT_LIMITS=np.tile([-180., 180.], (6, 1)),
            get_feedback_snapshot=lambda i=i: dict(joints_deg=state[i].copy(),
                joint_timestamp=time.time(), joint_hz=100.),
            set_target_joint_angles=lambda values, i=i: update(i, values),
            set_gripper_open_value=lambda _: self.fail('Return must not change the gripper target'))
            for i in range(2)]
        return robot, state, sent

    def test_joint_and_mit_use_same_return_path_without_gripper_writes(self):
        for mode in ('joint', 'mit'):
            with self.subTest(mode=mode):
                robot, state, sent = self.robot(mode)
                clock = VirtualClock()
                robot.initial_joints_deg = np.arange(12, dtype=float)
                with patch('wam.inference.piper_return.time.monotonic', clock.monotonic):
                    result = robot.return_to_pose('initial', .2, cancel=clock)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['gripper_targets'], 'unchanged')
                np.testing.assert_array_equal(np.concatenate(state), np.arange(12))
                self.assertEqual(len(sent[0]), len(sent[1]))

    def test_zero_destination_and_disabled_robot(self):
        robot, state, sent = self.robot()
        clock = VirtualClock()
        with patch('wam.inference.piper_return.time.monotonic', clock.monotonic):
            result = robot.return_to_pose('zero', .2, cancel=clock)
        self.assertEqual(result['status'], 'completed')
        np.testing.assert_array_equal(np.concatenate(state), np.zeros(12))
        robot.enabled = False
        count = len(sent[0])
        self.assertEqual(robot.return_to_pose()['status'], 'skipped')
        self.assertEqual(len(sent[0]), count)

    def test_missing_initial_pose_and_stale_feedback_do_not_move(self):
        robot, _, sent = self.robot()
        robot.initial_joints_deg = None
        with self.assertRaisesRegex(RuntimeError, 'not captured'):
            robot.return_to_pose()
        robot.controllers[0].get_feedback_snapshot = lambda: dict(
            joints_deg=np.zeros(6), joint_timestamp=1., joint_hz=30.)
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            robot.return_to_pose('zero')
        self.assertEqual(sent, [[], []])

    def test_pre_enable_pose_is_captured_after_warmup(self):
        robot, state, _ = self.robot()
        robot.enabled = False
        expected = np.concatenate(state)
        for controller in robot.controllers:
            controller.enable_robot = lambda **kw: True
            controller.start_control_loop = lambda: None
        robot.enable()
        self.assertTrue(robot.enabled)
        np.testing.assert_array_equal(robot.initial_joints_deg, expected)
        state[0][:] = 90.
        np.testing.assert_array_equal(robot.initial_joints_deg, expected)


if __name__ == '__main__':
    unittest.main()
