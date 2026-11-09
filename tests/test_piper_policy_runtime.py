"""Spawned policy orchestration and CLI preflight; fixtures never import hardware."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from wam.inference.piper_chunk_process import PiperChunkProcess
from wam.inference import piper_policy_runtime as runtime
from tools import run_piper_policy as cli


def observation():
    return {'state': np.zeros(14, np.float32), 'prompt': 'fixture task',
            **{name: np.zeros((16, 16, 3), np.uint8)
               for name in ('image', 'wrist_image_l', 'wrist_image_r')}}


class FullRecordingTests(unittest.TestCase):
    def test_saved_observation_prediction_and_sensor_metadata_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            obs = observation()
            obs['image'][0, 0] = [11, 22, 33]
            obs['state'][6] = .73
            result = dict(actions=np.ones((32, 14)), normalized_actions=np.ones((32, 80)))
            metrics = dict(request_tick=10, publish_tick=20, accepted_plan=True, observation_monotonic=100., sensor_timing={
                'feedback': {'left': {'gripper_diagnostics': {'status_code': 192, 'stroke_mm': 69.1}}}})
            recorder = runtime.ChunkRecorder(directory, 'full')
            try:
                self.assertTrue(recorder.submit(0, obs, result, metrics))
                obs['image'][:] = 0
                obs['state'][:] = 0
                result['actions'][:] = 0
            finally:
                recorder.close()
            with np.load(Path(directory) / 'chunk_000000.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['image'][0, 0], [11, 22, 33])
                self.assertAlmostEqual(float(saved['state'][6]), .73, places=6)
                np.testing.assert_array_equal(saved['actions'], np.ones((32, 14)))
                self.assertEqual(saved['wrist_image_r'].shape, (16, 16, 3))
            self.assertEqual(json.loads((Path(directory) / 'chunk_000000.json').read_text()), metrics)


class RuntimeHardwareFixture:
    """Only records fake commands to a temporary file on close."""
    def __init__(self, root, config, **kwargs):
        self.root, self.config = Path(root), config
        self.action_hz = float(config.get('action_hz', 100))
        self.enabled = False
        self.actions = []

    def observe(self):
        return {**observation(), 'enabled': self.enabled, 'commands': len(self.actions),
                'pid': os.getpid()}

    def enable(self):
        self.enabled = True

    def send_action(self, action):
        if not self.enabled:
            raise AssertionError('Cannot send before fake enable')
        self.actions.append(float(action[0]))

    def close(self):
        (self.root / 'closed.json').write_text(json.dumps({
            'enabled': self.enabled, 'actions': self.actions, 'pid': os.getpid()}))

    def return_to_pose(self, target, duration, *, cancel):
        if not self.enabled:
            raise AssertionError('Cannot return before enable')
        result = dict(status='completed', target=target, duration=duration)
        (self.root / 'returned.json').write_text(json.dumps(result))
        return result


class RuntimePolicyFixture:
    def __init__(self, *, infer_seconds=.035, interrupt_on=None, fail_on=None, probe=None):
        self.calls = []
        self.infer_seconds = infer_seconds
        self.interrupt_on, self.fail_on, self.probe = interrupt_on, fail_on, probe
        self.tick_advances = []

    def infer(self, obs, *, rtc_guidance=None):
        self.calls.append({'enabled': obs.get('enabled', False),
                           'guidance': rtc_guidance, 'pid': obs.get('pid')})
        index = len(self.calls)
        if index == self.interrupt_on:
            raise KeyboardInterrupt()
        if index == self.fail_on:
            raise RuntimeError('fixture inference failed')
        before = self.probe()['controller_ticks'] if self.probe else 0
        start = time.monotonic()
        # Exercise a busy parent rather than sleeping/releasing the Python GIL.
        while time.monotonic() - start < self.infer_seconds:
            pass
        if self.probe:
            self.tick_advances.append(self.probe()['controller_ticks'] - before)
        native = np.repeat((index * 100 + np.arange(32, dtype=np.float32))[:, None], 14, axis=1)
        normalized = np.repeat((index * 100 + np.arange(32, dtype=np.float32))[:, None], 80, axis=1)
        return {'actions': native, 'normalized_actions': normalized,
                'latency_seconds': time.monotonic() - start}


class PolicyRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'run'
        self.output = io.StringIO()
        self.quiet = redirect_stdout(self.output)
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def robot(self, *, recorded=False):
        factory = runtime.RecordedObservationRobot if recorded else RuntimeHardwareFixture
        config = {'observation': observation()} if recorded else {}
        robot = PiperChunkProcess(self.root, config, prompt='fixture task', hardware_factory=factory)
        self.addCleanup(robot.close)
        return robot

    def closed(self):
        return json.loads((self.root / 'closed.json').read_text())

    def test_sync_exact_first_k_each_chunk_and_warmup_precedes_enable(self):
        robot = self.robot()
        policy = RuntimePolicyFixture()
        summary = runtime.run_policy(policy, robot, self.out, mode='sync', chunks=3,
                                     execute_steps=3, hardware_execute=True, record='actions')
        self.assertEqual(self.closed()['actions'], [200, 201, 202, 300, 301, 302, 400, 401, 402])
        self.assertEqual(summary['sent_target_ticks'], 9)
        self.assertEqual(summary['controller_ticks'], 9)
        self.assertTrue(summary['completed'])
        self.assertFalse(summary['continuous_supply_observed'])
        self.assertEqual([c['enabled'] for c in policy.calls], [False, True, True, True])
        self.assertTrue(all(c['guidance'] is None for c in policy.calls))
        self.assertFalse(robot.process.is_alive())
        for path in sorted(self.out.glob('chunk_*.json')):
            self.assertEqual(json.loads(path.read_text())['request_tick'], 0)

    def test_async_child_continues_during_busy_model_and_skips_elapsed_targets(self):
        robot = self.robot()
        policy = RuntimePolicyFixture(infer_seconds=.06, probe=robot.status)
        summary = runtime.run_policy(policy, robot, self.out, mode='async', chunks=3,
                                     replan_steps=3, hardware_execute=True, record='actions')
        self.assertTrue(all(c['guidance'] is None for c in policy.calls))
        self.assertTrue(any(advance >= 3 for advance in policy.tick_advances[2:]))
        self.assertEqual(summary['chunks_completed'], 3)
        self.assertTrue(summary['completed'])
        records = [json.loads(p.read_text()) for p in sorted(self.out.glob('chunk_*.json'))]
        for index, record in enumerate(records[1:], 1):
            self.assertGreater(record['discarded_elapsed_actions'], 0)
            self.assertEqual(record['discarded_elapsed_actions'], record['publish_tick'] - record['request_tick'])
            base = (index + 2) * 100
            sent = [value for value in self.closed()['actions'] if base <= value < base + 32]
            self.assertTrue(sent)
            self.assertGreaterEqual(min(sent) - base, record['discarded_elapsed_actions'])
        self.assertFalse(robot.process.is_alive())
        self.assertTrue(all(c['pid'] != os.getpid() for c in policy.calls))

    def test_rtc_warms_both_paths_before_enable_and_only_then_guides_queued_plans(self):
        robot = self.robot()
        policy = RuntimePolicyFixture()
        summary = runtime.run_policy(policy, robot, self.out, mode='rtc', chunks=3,
                                     replan_steps=3, hardware_execute=True)
        self.assertEqual([c['enabled'] for c in policy.calls], [False, False, True, True, True])
        self.assertIsNone(policy.calls[0]['guidance'])
        self.assertIsNotNone(policy.calls[1]['guidance'])
        self.assertIsNone(policy.calls[2]['guidance'])
        self.assertTrue(all(c['guidance'] is not None for c in policy.calls[3:]))
        self.assertEqual(summary['rtc_algorithm'], 'inference_time_vjp')
        self.assertTrue(summary['completed'])
        self.assertFalse(robot.process.is_alive())
        rows = [json.loads(line.split(' ', 1)[1]) for line in self.output.getvalue().splitlines()
                if line.startswith('PIPER_POLICY_RESULT ')]
        for call, row in zip(policy.calls[3:], rows[1:]):
            self.assertLess(row['previous_available_actions'], 32)
            self.assertEqual(call['guidance'].prefix_attention_horizon, row['previous_available_actions'])
            self.assertEqual(row['prefix_attention_horizon'], row['previous_available_actions'])
            self.assertLessEqual(row['rtc_guidance_delay_ticks'], row['previous_available_actions'])

    def test_rtc_overrides_reach_warmup_and_live_guidance_and_keep_measured_delay(self):
        robot = self.robot()
        policy = RuntimePolicyFixture()
        summary = runtime.run_policy(policy, robot, self.out, mode='rtc', chunks=3,
                                     replan_steps=3, rtc_inference_delay=3, rtc_prefix_horizon=12,
                                     rtc_prefix_schedule='linear', guidance_weight=7.)
        self.assertEqual(summary['rtc_config']['inference_delay'], 3)
        self.assertEqual(summary['rtc_config']['prefix_horizon'], 12)
        self.assertFalse(self.closed()['enabled'])
        self.assertEqual(self.closed()['actions'], [])
        guided = [c['guidance'] for c in policy.calls if c['guidance'] is not None]
        self.assertGreaterEqual(len(guided), 3)
        for g in guided:
            self.assertEqual((g.inference_delay, g.prefix_attention_horizon, g.schedule,
                              g.max_guidance_weight), (3, 12, 'linear', 7.))
        rows = [json.loads(line.split(' ', 1)[1]) for line in self.output.getvalue().splitlines()
                if line.startswith('PIPER_POLICY_RESULT ')][1:]
        for row in rows:
            self.assertEqual(row['rtc_requested_delay_ticks'], 3)
            self.assertEqual(row['rtc_guidance_delay_ticks'], 3)
            self.assertGreater(row['estimated_inference_delay_ticks'], 3)
            self.assertEqual(row['rtc_prefix_schedule'], 'linear')
            self.assertEqual(row['prefix_attention_horizon'], 12)

    def test_rtc_shortfall_is_reported_without_guiding_padded_holds(self):
        robot = self.robot()
        policy = RuntimePolicyFixture(infer_seconds=.18)
        summary = runtime.run_policy(policy, robot, self.out, mode='rtc', chunks=4,
                                     replan_steps=3, record='none')
        rows = [json.loads(line.split(' ', 1)[1]) for line in self.output.getvalue().splitlines()
                if line.startswith('PIPER_POLICY_RESULT ')]
        shortfalls = [row for row in rows if row['rtc_prefix_covers_delay'] is False]
        self.assertTrue(shortfalls)
        self.assertEqual(summary['rtc_prefix_shortfall_chunks'], len(shortfalls))
        self.assertGreater(summary['queue_underrun_ticks'], 0)
        for row in shortfalls:
            self.assertGreater(row['estimated_inference_delay_ticks'], row['previous_available_actions'])
            if row['rtc_guidance']:
                self.assertEqual(row['rtc_guidance_delay_ticks'], row['previous_available_actions'])

    def test_recorded_preview_never_enables_or_sends_hardware(self):
        robot = self.robot(recorded=True)
        summary = runtime.run_policy(RuntimePolicyFixture(infer_seconds=.01), robot, self.out,
                                     chunks=2, execute_steps=2, hardware_execute=False)
        self.assertTrue(summary['completed'])
        self.assertEqual(summary['sent_target_ticks'], 4)
        self.assertFalse(summary['hardware_commands_sent'])
        self.assertFalse(robot.process.is_alive())

    def test_infinite_async_keyboard_interrupt_stops_and_closes_child(self):
        robot = self.robot()
        policy = RuntimePolicyFixture(interrupt_on=4)
        summary = runtime.run_policy(policy, robot, self.out, mode='async', chunks=0,
                                     replan_steps=2, hardware_execute=True)
        self.assertTrue(summary['interrupted'])
        self.assertFalse(summary['completed'])
        self.assertEqual(summary['chunks_completed'], 2)
        self.assertTrue(self.closed()['enabled'])
        self.assertFalse(robot.process.is_alive())
        self.assertTrue((self.out / 'summary.json').is_file())

    def test_interrupt_returns_only_after_enable_in_each_scheduling_mode(self):
        for mode, interrupt_on in (('sync', 3), ('async', 3), ('rtc', 4)):
            with self.subTest(mode=mode):
                returned = self.root / 'returned.json'
                returned.unlink(missing_ok=True)
                robot = self.robot()
                summary = runtime.run_policy(RuntimePolicyFixture(interrupt_on=interrupt_on), robot,
                    self.out / mode, mode=mode, chunks=0, replan_steps=2, execute_steps=2,
                    hardware_execute=True, return_on_interrupt='initial', return_duration=3.)
                self.assertTrue(summary['interrupted'])
                self.assertEqual(summary['return_on_interrupt']['status'], 'completed')
                self.assertEqual(json.loads(returned.read_text())['duration'], 3.)
                self.assertFalse(robot.process.is_alive())

    def test_no_return_on_normal_finish_fault_preview_or_warmup_interrupt(self):
        cases = [dict(chunks=1), dict(interrupt_on=1), dict(fail_on=3),
                 dict(interrupt_on=3, hardware_execute=False),
                 dict(interrupt_on=3, return_on_interrupt='none')]
        for case in cases:
            with self.subTest(case=case):
                returned = self.root / 'returned.json'
                returned.unlink(missing_ok=True)
                robot = self.robot()
                policy = RuntimePolicyFixture(interrupt_on=case.get('interrupt_on'), fail_on=case.get('fail_on'))
                options = dict(chunks=case.get('chunks', 2), execute_steps=2,
                               hardware_execute=case.get('hardware_execute', True),
                               return_on_interrupt=case.get('return_on_interrupt', 'initial'))
                if case.get('fail_on'):
                    with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                        runtime.run_policy(policy, robot, self.out, **options)
                else:
                    runtime.run_policy(policy, robot, self.out, **options)
                self.assertFalse(returned.exists())
                self.assertFalse(robot.process.is_alive())

    def test_return_failure_still_closes_hardware_and_records_failure(self):
        robot = self.robot()
        with patch.object(robot, 'return_to_pose', side_effect=RuntimeError('return feedback lost')):
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                runtime.run_policy(RuntimePolicyFixture(interrupt_on=3), robot, self.out,
                    chunks=0, execute_steps=2, hardware_execute=True, return_on_interrupt='zero')
        self.assertFalse(robot.process.is_alive())
        saved = json.loads((self.out / 'summary.json').read_text())
        self.assertEqual(saved['return_on_interrupt']['status'], 'failed')
        self.assertIn('feedback lost', saved['return_on_interrupt']['error'])

    def test_second_interrupt_before_return_rpc_still_disables(self):
        robot = self.robot()
        original_print = print
        def interrupt_at_announcement(*args, **kwargs):
            if args and args[0] == 'PIPER_RETURN_START':
                raise KeyboardInterrupt()
            return original_print(*args, **kwargs)
        with patch('builtins.print', new=interrupt_at_announcement):
            summary = runtime.run_policy(RuntimePolicyFixture(interrupt_on=3), robot, self.out,
                chunks=0, execute_steps=2, hardware_execute=True, return_on_interrupt='initial')
        self.assertEqual(summary['return_on_interrupt']['status'], 'aborted')
        self.assertFalse((self.root / 'returned.json').exists())
        self.assertFalse(robot.process.is_alive())

    def test_inference_failures_close_child_before_and_after_enable(self):
        for fail_on in (1, 3):
            with self.subTest(fail_on=fail_on):
                robot = self.robot()
                with self.assertRaisesRegex(RuntimeError, 'fixture inference failed'):
                    runtime.run_policy(RuntimePolicyFixture(fail_on=fail_on), robot,
                                       self.out / str(fail_on), mode='async', chunks=3,
                                       replan_steps=2, hardware_execute=True)
                self.assertFalse(robot.process.is_alive())
                self.assertEqual(self.closed()['enabled'], fail_on > 1)

    def test_recorder_disk_failure_propagates_after_hardware_cleanup(self):
        robot = self.robot()
        with patch.object(runtime.ChunkRecorder, '_write', side_effect=OSError('fixture disk full')):
            with self.assertRaisesRegex(RuntimeError, 'recording failed|cleanup failed'):
                runtime.run_policy(RuntimePolicyFixture(), robot, self.out, mode='async',
                                   chunks=4, replan_steps=2, hardware_execute=True, record='actions')
        self.assertFalse(robot.process.is_alive())
        self.assertTrue((self.root / 'closed.json').is_file())
        summary = json.loads((self.out / 'summary.json').read_text())
        self.assertFalse(summary['completed'])
        self.assertTrue(summary.get('error') or summary.get('cleanup_errors'))

    def test_slow_disk_is_bounded_and_hardware_closes_before_recording_finishes(self):
        robot = self.robot()
        started, release = threading.Event(), threading.Event()
        results, errors = [], []
        original_write = runtime.ChunkRecorder._write
        def delayed_write(recorder, *args):
            started.set()
            if not release.wait(5):
                raise TimeoutError('test writer was not released')
            return original_write(recorder, *args)
        def run():
            try:
                results.append(runtime.run_policy(RuntimePolicyFixture(infer_seconds=.005), robot,
                    self.out, mode='sync', chunks=6, execute_steps=2, hardware_execute=True,
                    record='full', record_queue_size=1))
            except BaseException as exc:
                errors.append(exc)
        with patch.object(runtime.ChunkRecorder, '_write', new=delayed_write):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(started.wait(3))
                deadline = time.monotonic() + 3
                while not (self.root / 'closed.json').exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((self.root / 'closed.json').is_file())
                self.assertTrue(worker.is_alive())
                self.assertEqual(len(self.closed()['actions']), 12)
            finally:
                release.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        stats = results[0]['recording']
        self.assertEqual(stats['queue_capacity'], 1)
        self.assertLessEqual(stats['submitted'], 2)
        self.assertGreaterEqual(stats['dropped'], 4)
        self.assertEqual(stats['written'], stats['submitted'])
        self.assertFalse(robot.process.is_alive())

    def test_output_setup_failure_still_closes_owned_transport(self):
        robot = self.robot()
        blocked = self.root / 'not-a-directory'
        blocked.write_text('file')
        with self.assertRaises(OSError):
            runtime.run_policy(RuntimePolicyFixture(), robot, blocked / 'output')
        self.assertFalse(robot.process.is_alive())
        self.assertTrue((self.root / 'closed.json').is_file())


class PolicyCliTests(unittest.TestCase):
    def test_down_can_fails_before_loading_the_model_or_creating_output(self):
        import builtins
        original_import = builtins.__import__
        def no_model_import(name, *args, **kwargs):
            if name == 'wam.inference.piper':
                raise AssertionError('CAN preflight must precede model imports')
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'trained'
            run.mkdir()
            for name in ('config.yaml', 'dataset_stats.json', 'chosen.pt'):
                (run / name).touch()
            (root / 'robot.json').write_text('{}')
            with patch('wam.inference.piper_can.resolve_can_config', side_effect=RuntimeError('CAN DOWN')):
                with patch.object(cli.os, 'chdir'), patch.object(cli, 'create_output') as output:
                    with patch.object(builtins, '__import__', side_effect=no_model_import):
                        with self.assertRaisesRegex(RuntimeError, 'CAN DOWN'):
                            cli.main(['--project-root', str(root), '--run-dir', 'trained',
                                      '--checkpoint', 'trained/chosen.pt', '--robot-config', 'robot.json',
                                      '--live'])
                    output.assert_not_called()

    def test_parse_defaults_and_reject_incompatible_or_invalid_options(self):
        args = cli.parse_args([])
        self.assertEqual((args.return_on_interrupt, args.return_duration), ('initial', 5.))
        self.assertEqual(cli.parse_args(['--return-on-interrupt', 'zero']).return_on_interrupt, 'zero')
        self.assertEqual(cli.parse_args(['--return-on-interrupt', 'none']).return_on_interrupt, 'none')
        self.assertEqual((args.mode, args.chunks, args.execute_steps, args.attention_backend),
                         ('sync', 2, 32, 'saved'))
        self.assertEqual(cli.parse_args(['--chunks', '0', '--mode', 'async']).chunks, 0)
        self.assertEqual((args.rtc_inference_delay, args.rtc_prefix_horizon, args.rtc_prefix_schedule),
                         (None, None, 'exp'))
        custom = cli.parse_args(['--mode', 'rtc', '--rtc-inference-delay', '8',
                                 '--rtc-prefix-horizon', '16', '--rtc-prefix-schedule', 'linear'])
        self.assertEqual((custom.rtc_inference_delay, custom.rtc_prefix_horizon, custom.rtc_prefix_schedule),
                         (8, 16, 'linear'))
        self.assertIsNone(cli.parse_args(['--rtc-inference-delay', 'auto']).rtc_inference_delay)
        self.assertIsNone(cli.parse_args(['--rtc-prefix-horizon', 'auto']).rtc_prefix_horizon)
        for argv in (['--execute'], ['--live', '--input-npz', 'record.npz'],
                     ['--return-duration', '0'], ['--return-duration', 'nan'],
                     ['--return-duration', 'inf'], ['--return-on-interrupt', 'bad'],
                     ['--chunks', '-1'], ['--execute-steps', '33'], ['--replan-steps', '0'],
                     ['--inference-steps', '0'], ['--record-queue-size', '0'],
                     ['--rtc-guidance-weight', 'nan'], ['--rtc-delay-margin', '-1'],
                     ['--rtc-inference-delay', '-1'], ['--rtc-prefix-horizon', '33'],
                     ['--rtc-inference-delay', '1.5'], ['--rtc-inference-delay', 'invalid'],
                     ['--rtc-prefix-schedule', 'invalid'],
                     ['--rtc-inference-delay', '16', '--rtc-prefix-horizon', '8'],
                     ['--mode', 'rtc', '--attention-backend', 'flex'], ['--prompt', ' ']):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(argv)

    def test_preflight_resolves_explicit_relative_weight_and_checks_manifest_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'trained'
            weights = run / 'checkpoints' / 'weights'
            weights.mkdir(parents=True)
            (run / 'config.yaml').touch()
            (run / 'dataset_stats.json').touch()
            chosen = root / 'chosen.pt'
            chosen.touch()
            cli.preflight(root, run, Path('chosen.pt'))
            manifest = weights / 'weights_manifest.json'
            manifest.write_text(json.dumps({'latest': {'path': '../../../chosen.pt'}}))
            with self.assertRaisesRegex(ValueError, 'manifest path'):
                cli.preflight(root, run)
            manifest.write_text(json.dumps({'latest': {'path': 'slot_0.pt'}}))
            with self.assertRaisesRegex(FileNotFoundError, 'slot_0.pt'):
                cli.preflight(root, run)
            (weights / 'slot_0.pt').touch()
            cli.preflight(root, run)
            self.assertEqual(cli.resolve(root, Path('trained')), run)

    def test_missing_weights_fail_before_hardware_or_model_import_and_output_creation(self):
        import builtins
        original_import = builtins.__import__
        def no_runtime_imports(name, *args, **kwargs):
            if name in ('wam.inference.piper', 'wam.inference.piper_chunk_process'):
                raise AssertionError('Missing resources must fail before model/hardware imports')
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, 'create_output') as output:
            with patch.object(builtins, '__import__', side_effect=no_runtime_imports):
                with self.assertRaisesRegex(FileNotFoundError, 'Restore the trained run'):
                    cli.main(['--project-root', directory, '--live', '--execute'])
            output.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_output_directories_are_unique_and_relative_to_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = cli.create_output(root, Path('runs/test'))
            second = cli.create_output(root, Path('runs/test'))
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, root / 'runs/test')
            self.assertTrue(first.is_dir() and second.is_dir())


if __name__ == '__main__':
    unittest.main()
