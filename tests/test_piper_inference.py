"""Piper inference boundaries; no hardware is connected or commanded."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hydra.utils import instantiate

from wam.inference.piper import load_config, validate_observation, IMAGE_KEYS, select_checkpoint, validate_checkpoint
from wam.inference.piper_robot import PhoenixPiperRobot, build_robot_observation
from eval.robotwin.policy import RobotWinPolicySession, _static_dimension_is_pad
from eval.robotwin.joint96_robot37_policy import current_rgb, VALID_DIMS
from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from tools.run_piper_inference import finish_chunk

ROOT = Path(os.environ.get('PIPER_TEST_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))


class PiperInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_config(ROOT, ROOT / 'postcheckpoints')
        cls.processor = instantiate(cfg.data.train.processor).eval()
        cls.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(ROOT / 'postcheckpoints/dataset_stats.json')))

    def observation(self):
        state = np.zeros(15, np.float32)
        state[[6, 13]] = .5
        state[14] = 123
        return dict(state=state, prompt='fold the towel and put it into the plate',
                    **{key: np.full((480, 640, 3), i * 80, np.uint8) for i, key in enumerate(IMAGE_KEYS)})

    def test_explicit_checkpoint_can_select_a_file_missing_from_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            weights = run / 'checkpoints/weights'
            weights.mkdir(parents=True)
            for name in ('slot_1.pt','slot_3.pt'):
                (weights/name).touch()
            (weights/'weights_manifest.json').write_text(json.dumps({'latest':{'path':'slot_1.pt','step':10000}}))
            self.assertEqual(select_checkpoint(run), (weights/'slot_1.pt',10000))
            self.assertEqual(select_checkpoint(run,weights/'slot_3.pt'), (weights/'slot_3.pt',None))
            with self.assertRaises(FileNotFoundError): select_checkpoint(run,weights/'missing.pt')

    def test_explicit_selection_keeps_step_and_piper_contract_validation(self):
        cfg = load_config(ROOT,ROOT/'postcheckpoints')
        contract = dict(domain='piper',embodiment_id=14,embodiment_name='openneo_aloha_piper',
                        action_space='absolute_joint14_radians_to_canonical80',cached_visuals=False)
        payload = {'step':17445,'joint96_metadata':{'soft_prompt':{'training_contract':contract}}}
        with patch('wam.inference.piper._joint96_identity',return_value={}):
            validate_checkpoint(cfg,payload,None)
            with self.assertRaisesRegex(ValueError,'manifest'): validate_checkpoint(cfg,payload,10000)
            payload['step'] = -1
            with self.assertRaisesRegex(ValueError,'positive'): validate_checkpoint(cfg,payload,None)
            payload['step'] = 17445
            contract['embodiment_id'] = 2
            with self.assertRaisesRegex(ValueError,'Piper slot14'): validate_checkpoint(cfg,payload,None)

    def test_observation_auxiliary_is_not_a_joint(self):
        original = self.observation()
        clean = validate_observation(original)
        self.assertEqual(clean['state'].shape, (14,))
        clean['state'][0] = 2
        self.assertEqual(original['state'][0], 0)
        self.assertEqual(original['state'][14], 123)

    def test_reject_invalid_sensor_values(self):
        for mutation in ('nan', 'float_rgb', 'gripper', 'prompt'):
            obs = self.observation()
            if mutation == 'nan': obs['state'][0] = np.nan
            elif mutation == 'float_rgb': obs['image'] = obs['image'].astype(np.float32)
            elif mutation == 'gripper': obs['state'][6] = 1.1
            else: obs['prompt'] = None
            with self.assertRaises(ValueError): validate_observation(obs)

    def test_native_radians_roundtrip_and_padding(self):
        state = np.array([-.7, 1.2, -1.1, .1, 1., -.2, .25, .4, 1.7, -1.2, 0, 1.1, .3, .75], np.float32)
        holder = SimpleNamespace(processor=self.processor)
        normalized = RobotWinPolicySession._normalize_state(holder, state)
        batch = {'state': {'default': torch.from_numpy(state.copy()).unsqueeze(0)},
                 'action': {'default': torch.from_numpy(state.copy()).view(1,14)}}
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        native = RobotWinPolicySession._denormalize_action(holder, batch['action'])[0, 0]
        np.testing.assert_allclose(native, state, atol=2e-7)
        for kind in ('state', 'action'):
            pad = _static_dimension_is_pad(self.processor, kind)
            self.assertEqual((~pad).nonzero().flatten().tolist(), VALID_DIMS)
            self.assertTrue(torch.equal(normalized[:, pad], torch.zeros_like(normalized[:, pad])))

    def test_camera_order_and_original_clean_preprocessing(self):
        obs = self.observation()
        envelope = {'observation': {name: {'rgb': obs[key]} for name, key in
                    zip(('head_camera', 'left_camera', 'right_camera'), IMAGE_KEYS)}}
        rgb = current_rgb(self.processor, envelope)
        self.assertEqual(tuple(rgb.shape), (1, 3, 3, 256, 256))
        np.testing.assert_allclose(rgb[0].mean(dim=(1,2,3)).numpy(), np.array([0,80,160])/127.5-1, atol=1e-6)

    def test_phoenix_targets_pass_directly_to_drivers(self):
        sent = [[], []]
        controllers = [SimpleNamespace(
            set_target_joint_angles=lambda x, i=i: sent[i].append(('joints_deg', x.copy())),
            set_gripper_open_value=lambda x, i=i: sent[i].append(('gripper', x))) for i in range(2)]
        robot = PhoenixPiperRobot.__new__(PhoenixPiperRobot)
        robot.controllers = controllers
        robot.enabled = True
        robot.execute_steps = 10
        robot.action_hz = 30.
        actions = np.arange(32*14, dtype=np.float32).reshape(32,14) / 100
        with patch('wam.inference.piper_robot.time.sleep') as sleep:
            robot.execute(actions)
        self.assertEqual(sleep.call_count, 10)
        self.assertEqual([len(x) for x in sent], [20,20])
        for arm_index, arm in enumerate(sent):
            offset = arm_index * 7
            for step in range(10):
                np.testing.assert_allclose(arm[step*2][1], np.rad2deg(actions[step,offset:offset+6]))
                self.assertEqual(arm[step*2+1][1], float(actions[step,offset+6]))
        before = [len(x) for x in sent]
        actions[31,0] = np.nan
        with self.assertRaises(ValueError): robot.execute(actions)
        self.assertEqual([len(x) for x in sent], before)

    def test_native_hardware_observation_conversion(self):
        mapping = dict(zip(('head','left','right'), IMAGE_KEYS))
        frames = {name: {'rgb': np.full((2,3,3), [10,20,30], np.uint8)} for name in mapping}
        # Old positive timestamps are accepted: there is no pi0.5 age cutoff.
        snapshots = [dict(joints_deg=np.array([0,30,-60,90,-120,180]), gripper=grip,
                          joint_timestamp=1., gripper_timestamp=1., joint_hz=30., gripper_hz=30.)
                     for grip in (.25,.75)]
        obs = build_robot_observation(frames, snapshots, mapping, 'test')
        np.testing.assert_allclose(obs['state'][:6], np.deg2rad(snapshots[0]['joints_deg']), atol=2e-7)
        self.assertEqual(obs['state'][6], .25)
        self.assertEqual(obs['state'][13], .75)
        np.testing.assert_array_equal(obs['image'][0,0], [30,20,10])
        snapshots[1]['joint_hz'] = 0
        with self.assertRaisesRegex(RuntimeError, "right arm has not received.*joint_hz.*0"):
            build_robot_observation(frames, snapshots, mapping, 'test')

    def test_read_only_close_does_not_disable_arms(self):
        calls = []
        robot = PhoenixPiperRobot.__new__(PhoenixPiperRobot)
        robot.enable_attempted = False
        robot.cameras = SimpleNamespace(stop=lambda: calls.append('camera_stop'))
        robot.locks = []
        robot.controllers = [SimpleNamespace(
            latch_emergency_stop=lambda: self.fail('read-only stop'),
            finish_emergency_disable=lambda: self.fail('read-only disable'),
            disconnect_without_homing=lambda **kwargs: calls.append(kwargs) or True)]
        robot.close()
        self.assertEqual(calls, ['camera_stop', {'require_disabled': False, 'feedback_timeout_s': 1}])

    def test_execution_precedes_recording_and_records_age(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            actions = np.zeros((32,14), np.float32)
            def execute(chunk):
                self.assertFalse((out / 'chunk_0000.npz').exists())
                self.assertFalse((out / 'chunk_0000.json').exists())
                self.assertIs(chunk, actions)
            robot = SimpleNamespace(execute=execute)
            metrics = {'hardware_commands_sent': False}
            finish_chunk(out, 0, self.observation(),
                         {'actions': actions, 'normalized_actions': np.zeros((32,80), np.float32)},
                         metrics, robot, time.monotonic())
            report = json.loads((out / 'chunk_0000.json').read_text())
            self.assertTrue(report['hardware_commands_sent'])
            self.assertGreaterEqual(report['observation_age_before_execution_seconds'], 0)
            self.assertGreaterEqual(report['observation_age_after_execution_seconds'],
                                    report['observation_age_before_execution_seconds'])

    def test_driver_failure_still_saves_prediction(self):
        def fail(chunk):
            raise RuntimeError('driver failure')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'driver failure'):
                finish_chunk(out, 0, self.observation(),
                             {'actions': np.zeros((32,14)), 'normalized_actions': np.zeros((32,80))},
                             {'hardware_commands_sent': False},
                             SimpleNamespace(execute=fail),
                             time.monotonic()-3)
            report = json.loads((out / 'chunk_0000.json').read_text())
            self.assertIsNone(report['hardware_commands_sent'])
            self.assertIn('driver failure', report['execution_error'])
            with np.load(out / 'chunk_0000.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['actions'].shape, (32,14))


if __name__ == '__main__':
    unittest.main()
