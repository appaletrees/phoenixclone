"""Spawned chunk scheduling with fake hardware only; no CUDA or robot I/O."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from wam.inference.piper_chunk_process import PiperChunkProcess, _ChunkClock


class ChunkHardwareFixture:
    def __init__(self, root, config, **kwargs):
        self.root = Path(root)
        self.config = config
        self.action_hz = float(config.get("action_hz", 100.))
        self.enabled = False
        self.actions = []

    def enable(self):
        self.enabled = True

    def observe(self):
        before = len(self.actions)
        time.sleep(float(self.config.get("observe_seconds", 0)))
        if self.config.get("observe_failure"):
            raise RuntimeError("fixture observation failed")
        torch_module = __import__("sys").modules.get("torch")
        return dict(pid=os.getpid(), enabled=self.enabled, commands=len(self.actions),
                    commands_during_observe=len(self.actions) - before,
                    state=np.zeros(14, dtype=np.float32),
                    cuda_initialized=torch_module is not None and torch_module.cuda.is_initialized())

    def send_action(self, action):
        if not self.enabled:
            raise RuntimeError("fixture is not enabled")
        if self.config.get("send_failure"):
            raise RuntimeError("fixture send failed")
        self.actions.append(np.asarray(action).tolist())

    def close(self):
        (self.root / "closed.json").write_text(json.dumps(dict(enabled=self.enabled, actions=self.actions)))

    def return_to_pose(self, target, duration, *, cancel):
        before = len(self.actions)
        (self.root / "return_started").touch()
        aborted = cancel.wait(duration)
        return dict(status="aborted" if aborted else "completed", target=target,
                    commands_before=before, commands_after=len(self.actions))


def result(base=0):
    return dict(actions=np.repeat((np.arange(32, dtype=np.float32) + base)[:, None], 14, axis=1),
                normalized_actions=np.repeat((np.arange(32, dtype=np.float32) + base)[:, None], 80, axis=1))


def wait_finished(robot):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        stats = robot.status()
        if stats["finished"]:
            return stats
        time.sleep(.005)
    raise AssertionError("Fixture clock failed to finish")


class ChunkProcessTests(unittest.TestCase):
    def test_return_stops_policy_clock_before_moving_and_drains_old_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            with PiperChunkProcess(directory, {}, prompt="test", hardware_factory=ChunkHardwareFixture) as robot:
                robot.enable()
                robot.start(result(), execute=True)
                time.sleep(.04)
                robot.sequence += 1
                robot.connection.send((robot.sequence, "observe", None))
                returned = robot.return_to_pose('initial', .08)
                self.assertEqual(returned['status'], 'completed')
                self.assertEqual(returned['commands_before'], returned['commands_after'])
                self.assertGreater(returned['commands_before'], 0)
                self.assertTrue(robot.status()['finished'])

    def test_second_interrupt_cancels_return_and_closes_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = PiperChunkProcess(directory, {}, prompt="test", hardware_factory=ChunkHardwareFixture)
            try:
                robot.enable()
                def interrupt_when_returning():
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        if (Path(directory) / 'return_started').exists():
                            __import__('_thread').interrupt_main()
                            return
                        time.sleep(.005)
                timer = threading.Thread(target=interrupt_when_returning)
                timer.start()
                started = time.monotonic()
                returned = robot.return_to_pose('initial', 10.)
                self.assertEqual(returned['status'], 'aborted')
                self.assertLess(time.monotonic() - started, 3.)
                timer.join(3)
            finally:
                robot.close()
            self.assertFalse(robot.process.is_alive())

    def test_return_never_enables_an_idle_read_only_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            with PiperChunkProcess(directory, {}, prompt="test", hardware_factory=ChunkHardwareFixture) as robot:
                self.assertEqual(robot.return_to_pose()['reason'], 'not_enabled')
                self.assertFalse((Path(directory) / 'return_started').exists())

    def test_spawn_isolation_and_preview_never_enable_or_send(self):
        with tempfile.TemporaryDirectory() as directory:
            with PiperChunkProcess(directory, {}, prompt="test", hardware_factory=ChunkHardwareFixture) as robot:
                self.assertTrue(robot.status()["finished"])
                self.assertEqual(robot.stop()["controller_ticks"], 0)
                packet = robot.observe()
                self.assertNotEqual(packet["observation"]["pid"], os.getpid())
                self.assertFalse(packet["observation"]["cuda_initialized"])
                self.assertFalse(packet["observation"]["enabled"])
                self.assertEqual(packet["request_tick"], 0)
                self.assertIsNone(packet["previous_normalized_actions"])
                self.assertLessEqual(packet["request_monotonic"], packet["observed_monotonic"])
                robot.start(result(), execute=False, limit_steps=4)
                self.assertEqual(wait_finished(robot)["sent_target_ticks"], 4)
                robot.stop()
                self.assertEqual(robot.observe()["observation"]["commands"], 0)
            self.assertFalse(robot.process.is_alive())
            robot.close()
            saved = json.loads((Path(directory) / "closed.json").read_text())
            self.assertFalse(saved["enabled"])
            self.assertEqual(saved["actions"], [])

    def test_synchronous_chunks_send_exactly_k_and_restart_from_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            with PiperChunkProcess(directory, {}, prompt="test", hardware_factory=ChunkHardwareFixture) as robot:
                robot.enable()
                for base, steps in ((0, 5), (100, 3)):
                    robot.start(result(base), execute=True, limit_steps=steps)
                    stats = wait_finished(robot)
                    self.assertEqual(stats["controller_ticks"], steps)
                    self.assertEqual(stats["sent_target_ticks"], steps)
                    self.assertEqual(stats["queue_underrun_ticks"], 0)
                    time.sleep(.04)
                    self.assertEqual(robot.status()["sent_target_ticks"], steps)
                    self.assertTrue(robot.stop()["finished"])
                    self.assertEqual(robot.observe()["request_tick"], 0)
            saved = json.loads((Path(directory) / "closed.json").read_text())
            self.assertEqual([row[0] for row in saved["actions"]], [0, 1, 2, 3, 4, 100, 101, 102])

    def test_slow_observe_does_not_block_child_clock_and_captures_origin_first(self):
        with tempfile.TemporaryDirectory() as directory:
            with PiperChunkProcess(directory, {"observe_seconds": .09}, prompt="test",
                                   hardware_factory=ChunkHardwareFixture) as robot:
                robot.enable()
                robot.start(result(), execute=True)
                packet = robot.observe()
                self.assertGreaterEqual(packet["observation"]["commands_during_observe"], 3)
                self.assertGreaterEqual(packet["observed_monotonic"] - packet["request_monotonic"], .08)
                self.assertGreater(robot.status()["controller_ticks"], packet["request_tick"])
                self.assertEqual(packet["previous_available_actions"], 32 - packet["request_tick"])
                publication = robot.publish(result(100), packet["request_tick"])
                self.assertTrue(publication["accepted_plan"])
                self.assertGreater(publication["discarded_elapsed_actions"], 0)
                time.sleep(.03)
                robot.stop()
            saved = json.loads((Path(directory) / "closed.json").read_text())
            self.assertTrue(any(row[0] >= 100 for row in saved["actions"]))

    def test_fully_elapsed_plan_is_rejected_and_underrun_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            with PiperChunkProcess(directory, {"action_hz": 200.}, prompt="test",
                                   hardware_factory=ChunkHardwareFixture) as robot:
                robot.start(result(), execute=False)
                time.sleep(.23)
                publication = robot.publish(result(100), 0)
                self.assertFalse(publication["accepted_plan"])
                self.assertEqual(publication["discarded_elapsed_actions"], 32)
                self.assertEqual(publication["remaining_actions"], 0)
                stats = robot.stop()
                self.assertGreater(stats["queue_underrun_ticks"], 0)
                self.assertTrue(stats["finished"])

    def test_worker_rpc_error_closes_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = PiperChunkProcess(directory, {"observe_failure": True}, prompt="test",
                                      hardware_factory=ChunkHardwareFixture)
            try:
                with self.assertRaisesRegex(RuntimeError, "fixture observation failed"):
                    robot.observe()
            finally:
                robot.close()
            self.assertFalse(robot.process.is_alive())
            self.assertTrue((Path(directory) / "closed.json").is_file())

    def test_driver_thread_fault_propagates_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = PiperChunkProcess(directory, {"send_failure": True}, prompt="test",
                                      hardware_factory=ChunkHardwareFixture)
            try:
                robot.enable()
                with self.assertRaisesRegex(RuntimeError, "fixture send failed"):
                    robot.start(result(), execute=True)
                    time.sleep(.08)
                    robot.status()
            finally:
                robot.close()
            self.assertFalse(robot.process.is_alive())
            self.assertTrue((Path(directory) / "closed.json").is_file())

    def test_close_drains_unread_reply_from_interrupted_request(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = PiperChunkProcess(directory, {"observe_seconds": .06}, prompt="test",
                                      hardware_factory=ChunkHardwareFixture)
            # An interrupted _receive leaves the preceding response unread.
            # Submit exactly that state without delivering SIGINT to the tests.
            robot.sequence += 1
            robot.connection.send((robot.sequence, "observe", None))
            robot.close()
            self.assertFalse(robot.process.is_alive())
            self.assertTrue((Path(directory) / "closed.json").is_file())

    def test_keyboard_interrupt_during_request_still_closes_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = PiperChunkProcess(directory, {"observe_seconds": .2}, prompt="test",
                                      hardware_factory=ChunkHardwareFixture)
            timer = threading.Timer(.03, __import__("_thread").interrupt_main)
            timer.start()
            try:
                with self.assertRaises(KeyboardInterrupt):
                    robot.observe()
            finally:
                timer.join()
                robot.close()
            self.assertFalse(robot.process.is_alive())
            self.assertFalse(robot.receiver.is_alive())
            self.assertTrue((Path(directory) / "closed.json").is_file())

    def test_stop_idle_is_idempotent_and_execute_requires_enable(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = PiperChunkProcess(directory, {}, prompt="test", hardware_factory=ChunkHardwareFixture)
            try:
                self.assertEqual(robot.stop(), robot.stop())
                with self.assertRaisesRegex(RuntimeError, "Enable the robot"):
                    robot.start(result(), execute=True)
            finally:
                robot.close()
            self.assertFalse(json.loads((Path(directory) / "closed.json").read_text())["enabled"])

    def test_sync_late_sender_preserves_each_target_without_catch_up_burst(self):
        sent = []
        def sender(action):
            sent.append(float(action[0]))
            time.sleep(.025)
        clock = _ChunkClock(sender, result(), action_hz=100., limit_steps=4)
        clock.start()
        clock.thread.join(timeout=2)
        self.assertEqual(sent, [0, 1, 2, 3])
        self.assertGreater(clock.stats()["missed_control_ticks"], 0)
        self.assertEqual(clock.stop()["sent_target_ticks"], 4)


if __name__ == "__main__":
    unittest.main()
