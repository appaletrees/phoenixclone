"""RTC alignment, concurrency and failure propagation without hardware."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from wam.inference.piper_rtc import (
    ActionTimeline, AsyncActionExecutor, make_guidance, run_async_rtc,
)
from wam.inference.online_action_policy import (
    _prepare_rtc_action_guidance, _apply_rtc_guidance_to_action_velocity,
)
from wam.inference.piper_robot_process import PiperRobotProcess


class ProcessHardwareFixture:
    """Spawn-picklable hardware substitute; never imports or connects drivers."""
    action_hz = 30.
    def __init__(self, root, config, **kwargs):
        self.root = Path(root)
        self.enabled = False
        self.actions = []
    def observe(self):
        return {'state':np.arange(14,dtype=np.float32), 'pid':os.getpid(),
                'enabled':self.enabled, 'commands':len(self.actions)}
    def enable(self):
        self.enabled = True
    def send_action(self, action):
        if not self.enabled:
            raise RuntimeError('fixture not enabled')
        self.actions.append(np.asarray(action).tolist())
    def close(self):
        (self.root/'closed.json').write_text(json.dumps({'enabled':self.enabled,'actions':self.actions}))


def plan():
    return {"actions": np.repeat(np.arange(32, dtype=np.float32)[:,None], 14, axis=1),
            "normalized_actions": np.repeat(np.arange(32, dtype=np.float32)[:,None], 80, axis=1),
            "latency_seconds": .01}


class RTCExecutorTests(unittest.TestCase):
    def test_hardware_rpc_is_a_separate_process_and_propagates_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            robot=PiperRobotProcess(directory,{},prompt='test',_factory=ProcessHardwareFixture)
            try:
                observation=robot.observe()
                self.assertNotEqual(observation['pid'],os.getpid())
                np.testing.assert_array_equal(observation['state'],np.arange(14,dtype=np.float32))
                self.assertFalse(observation['enabled'])
                robot.preview_action(np.zeros(14,np.float32))
                self.assertEqual(robot.observe()['commands'],0)
                self.assertFalse(robot.observe()['enabled'])
                with self.assertRaisesRegex(RuntimeError,'fixture not enabled'):
                    robot.send_action(np.zeros(14,np.float32))
                robot.enable()
                robot.send_action(np.arange(14,dtype=np.float32))
                self.assertEqual(robot.observe()['commands'],1)
            finally:
                robot.close()
            self.assertFalse(robot.process.is_alive())
            robot.close()
            report=json.loads((Path(directory)/'closed.json').read_text())
            self.assertEqual(report['actions'],[list(range(14))])

    def test_previous_plan_is_aligned_and_right_padded(self):
        result = plan()
        timeline = ActionTimeline(result['actions'], result['normalized_actions'], origin=10)
        prefix, available = timeline.aligned_prefix(13)
        np.testing.assert_array_equal(prefix[:,0], np.minimum(np.arange(32)+3,31))
        self.assertEqual(available, 29)
        self.assertIsNone(timeline.action_at(42))
        prefix, available = timeline.aligned_prefix(50)
        self.assertEqual(available, 0)
        np.testing.assert_array_equal(prefix, np.full((32,80),31))

    def test_publish_discards_actions_elapsed_during_inference(self):
        executor = AsyncActionExecutor(lambda action: None, plan())
        executor.tick = 18
        replacement = plan()
        replacement['actions'] += 100
        publication = executor.publish(replacement, origin=10)
        self.assertEqual(publication['discarded_elapsed_actions'], 8)
        self.assertEqual(publication['remaining_actions'], 24)
        np.testing.assert_array_equal(executor.timeline.action_at(18), np.full(14,108))
        executor.tick = 50
        self.assertFalse(executor.publish(plan(), origin=10)['accepted_plan'])
        self.assertEqual(executor.timeline.origin, 10)

    def test_delay_is_measured_in_control_ticks(self):
        guidance = make_guidance(plan()['normalized_actions'], .51, action_hz=30, margin=2)
        self.assertEqual(guidance.inference_delay, 18)
        self.assertEqual(guidance.prefix_attention_horizon, 32)
        self.assertEqual(guidance.schedule, 'exp')

    def test_explicit_delay_schedule_and_prefix_cap_do_not_guide_outside_real_overlap(self):
        for available, expected_delay, expected_prefix in ((22, 8, 12), (5, 5, 5)):
            guidance = make_guidance(plan()['normalized_actions'], .9, margin=10,
                                     inference_delay=8, prefix_horizon=12, schedule='linear',
                                     available_actions=available, weight=7.)
            self.assertEqual(guidance.inference_delay, expected_delay)
            self.assertEqual(guidance.prefix_attention_horizon, expected_prefix)
            self.assertEqual(guidance.max_guidance_weight, 7.)
            prepared = _prepare_rtc_action_guidance(
                guidance, action_horizon=32, action_dim=80, action_dim_is_pad=None,
                device=torch.device('cpu'), dtype=torch.float32)
            weights = prepared.prefix_weights.flatten()
            self.assertTrue(torch.equal(weights[:expected_delay], torch.ones(expected_delay)))
            self.assertEqual(torch.count_nonzero(weights[expected_prefix:]).item(), 0)
            if available == 22:
                torch.testing.assert_close(weights[8:12], torch.tensor([.8, .6, .4, .2]))

    def test_auto_delay_with_explicit_cap_and_invalid_settings(self):
        g = make_guidance(plan()['normalized_actions'], .51, margin=2, prefix_horizon=12)
        self.assertEqual((g.inference_delay, g.prefix_attention_horizon), (12, 12))
        for kwargs in ({'inference_delay': -1}, {'prefix_horizon': 33}, {'schedule': 'invalid'},
                       {'inference_delay': 9, 'prefix_horizon': 8}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_guidance(plan()['normalized_actions'], .1, **kwargs)

    def test_author_guidance_changes_velocity_and_preserves_padding(self):
        pad = torch.tensor([False, True])
        guidance = make_guidance(np.ones((32,2),np.float32), .1)
        prepared = _prepare_rtc_action_guidance(
            guidance, action_horizon=32, action_dim=2, action_dim_is_pad=pad,
            device=torch.device('cpu'), dtype=torch.float32)
        noisy = torch.zeros((1,32,2))
        guided = _apply_rtc_guidance_to_action_velocity(
            latents_action=noisy, timestep_action=torch.tensor([500.]), num_train_timesteps=1000,
            predict_velocity=lambda x: x*.1, guidance=prepared, action_dim_is_pad=pad)
        self.assertTrue(torch.isfinite(guided).all())
        self.assertGreater(float(guided[:,:,0].abs().sum()), 0)
        self.assertEqual(float(guided[:,:,1].abs().sum()), 0)

    def test_padded_old_targets_cannot_affect_guidance_even_with_token_mixing(self):
        # After 14 ticks only 18 old predictions remain. The tensor padding
        # must not constrain the new future, regardless of its stored values.
        previous = np.ones((32, 2), np.float32)
        perturbed = previous.copy()
        perturbed[18:] = 100
        pad = torch.tensor([False, True])

        def velocity(values, available=32):
            guidance = make_guidance(values, .47, available_actions=available)
            prepared = _prepare_rtc_action_guidance(
                guidance, action_horizon=32, action_dim=2, action_dim_is_pad=pad,
                device=torch.device('cpu'), dtype=torch.float32)
            return _apply_rtc_guidance_to_action_velocity(
                latents_action=torch.zeros((1, 32, 2)),
                timestep_action=torch.tensor([500.]), num_train_timesteps=1000,
                predict_velocity=lambda x: x * .1 + x.mean(dim=1, keepdim=True) * .2,
                guidance=prepared, action_dim_is_pad=pad)

        # The old full-horizon guide contaminates even the real prefix via VJP.
        self.assertGreater(float((velocity(previous) - velocity(perturbed))[:, :18].abs().max()), 0)
        torch.testing.assert_close(velocity(previous, 18), velocity(perturbed, 18), rtol=0, atol=0)

    def test_guidance_never_extends_beyond_valid_previous_predictions(self):
        for available in (0, 2, 18, 32):
            with self.subTest(available=available):
                guidance = make_guidance(plan()['normalized_actions'], .99, available_actions=available)
                self.assertEqual(guidance.prefix_attention_horizon, available)
                self.assertEqual(guidance.inference_delay, min(32, available))
                _prepare_rtc_action_guidance(
                    guidance, action_horizon=32, action_dim=80, action_dim_is_pad=None,
                    device=torch.device('cpu'), dtype=torch.float32)

    def test_controller_continues_while_gpu_worker_is_busy(self):
        sent = []
        overlap = []
        class Policy:
            def infer(self, obs, **kwargs):
                before = len(sent)
                time.sleep(.04)
                overlap.append(len(sent)-before)
                result = plan()
                result['latency_seconds'] = .04
                return result
        with tempfile.TemporaryDirectory() as directory:
            summary = run_async_rtc(
                Policy(), lambda: {'state':np.zeros(14)}, lambda action:sent.append(action.copy()),
                directory, chunks=4, replan_steps=2, action_hz=100.)
            self.assertTrue(summary['completed'])
            self.assertFalse(summary['hardware_commands_sent'])
            self.assertTrue(all(value > 0 for value in overlap[3:]))
            self.assertEqual(summary['queue_underrun_ticks'], 0)
            for index in range(1,4):
                report=json.loads((Path(directory)/f'chunk_{index:04d}.json').read_text())
                self.assertTrue(report['rtc_guidance'])
                self.assertGreater(report['discarded_elapsed_actions'], 0)
            count = len(sent)
            time.sleep(.02)
            self.assertEqual(len(sent), count)

    def test_inference_failure_stops_executor_and_records_failure(self):
        sent = []
        class Policy:
            calls = 0
            def infer(self, obs, **kwargs):
                self.calls += 1
                if self.calls == 4:
                    raise RuntimeError('GPU failed')
                return plan()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'GPU failed'):
                run_async_rtc(Policy(), lambda: {}, lambda action:sent.append(action),
                              directory, chunks=3, replan_steps=1, action_hz=100.)
            summary=json.loads((Path(directory)/'rtc_summary.json').read_text())
            self.assertFalse(summary['completed'])
            self.assertIn('GPU failed', summary['error'])
            count=len(sent)
            time.sleep(.02)
            self.assertEqual(len(sent),count)
            self.assertFalse(any(t.name=='phoenix-action-clock' and t.is_alive() for t in threading.enumerate()))

    def test_driver_failure_reaches_planner(self):
        def fail(action):
            raise RuntimeError('CAN failed')
        executor=AsyncActionExecutor(fail, plan())
        executor.start()
        with self.assertRaisesRegex(RuntimeError, 'driver failed'):
            executor.wait_until(2)
        with self.assertRaisesRegex(RuntimeError, 'driver failed'):
            executor.close()
        self.assertFalse(executor.thread.is_alive())

    def test_slow_inference_reports_underrun_and_rejects_fully_elapsed_plan(self):
        class Policy:
            calls = 0
            def infer(self, obs, **kwargs):
                self.calls += 1
                if self.calls == 4:
                    time.sleep(.05)
                return plan()
        with tempfile.TemporaryDirectory() as directory:
            summary=run_async_rtc(Policy(), lambda: {}, lambda action:None, directory,
                                  chunks=2, replan_steps=1, action_hz=1000.)
            self.assertTrue(summary['completed'])
            self.assertGreater(summary['queue_underrun_ticks'],0)
            self.assertFalse(summary['continuous_supply_observed'])
            row=json.loads((Path(directory)/'chunk_0001.json').read_text())
            self.assertFalse(row['accepted_plan'])
            self.assertEqual(row['remaining_actions'],0)

    def test_recording_failure_is_reported_and_command_thread_exits(self):
        class Policy:
            def infer(self, obs, **kwargs):
                return plan()
        with tempfile.TemporaryDirectory() as directory:
            with patch('wam.inference.piper_rtc.save_rtc_chunk', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError,'disk full'):
                    run_async_rtc(Policy(), lambda:{}, lambda action:None, directory,
                                  chunks=1,replan_steps=1,action_hz=100.)
            report=json.loads((Path(directory)/'rtc_summary.json').read_text())
            self.assertFalse(report['completed'])
            self.assertIn('disk full',report['recording_error'])
            self.assertFalse(any(t.name=='phoenix-action-clock' and t.is_alive() for t in threading.enumerate()))


if __name__ == '__main__':
    unittest.main()
