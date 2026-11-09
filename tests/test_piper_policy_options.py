"""Policy execution options with CPU model doubles and no checkpoint or hardware."""
from contextlib import ExitStack
from pathlib import Path
import os
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from wam.inference import piper


class ModelDouble(torch.nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.mot = torch.nn.Linear(1, 1, bias=False)
        self.mot.weight.data.fill_(0.5)
        self.mot.attention_backend = backend


class PiperPolicyOptionsTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ))
        self.cfg = OmegaConf.create({
            'eval_num_inference_steps': 10,
            'data': {'train': {'processor': {}}},
            'model': {'load_text_encoder': False,
                      'mot_attention_backend': 'flex',
                      'mot_flex_masked_batches': True},
        })
        self.pad = torch.ones(80, dtype=torch.bool)
        self.pad[piper.VALID_DIMS] = False
        self.normalized = torch.arange(32 * 80, dtype=torch.float32).reshape(32, 80) / 1000
        self.payload = {'step': 10000, 'mot': {'weight': torch.tensor([[0.5]])}}
        for name, value in (
            ('install_import_guard', None),
            ('load_config', self.cfg),
            ('select_checkpoint', (Path('/unused/checkpoint.pt'), 10000)),
            ('validate_checkpoint', {}),
            ('prepare_text_cache', Path('/unused/text.pt')),
            ('official_text_padding', (torch.zeros(128, 4), torch.ones(128, dtype=torch.bool))),
            ('load_dataset_stats_from_json', {}),
            ('_static_dimension_is_pad', self.pad),
            ('_load_joint96_checkpoint', None),
            ('current_rgb', torch.zeros(1, 3, 3, 256, 256)),
        ):
            self.stack.enter_context(patch.object(piper, name, return_value=value))
        self.stack.enter_context(patch.object(piper.torch, 'load', side_effect=lambda path, **kw:
            self.payload if Path(path).name == 'checkpoint.pt' else {'context': None, 'mask': None}))
        self.stack.enter_context(patch.object(piper, 'instantiate'))
        self.factory = self.stack.enter_context(patch.object(piper, 'instantiate_piper_model',
            side_effect=lambda cfg, **kw: ModelDouble(cfg.model.mot_attention_backend)))
        self.stack.enter_context(patch('eval.robotwin.current_visual_adapter.configure_current_visual',
                                      return_value={'mode': 'repeat9_full_csvae'}))
        self.stack.enter_context(patch.object(piper.RobotWinPolicySession, '_normalize_state',
                                              return_value=torch.zeros(1, 80)))
        self.stack.enter_context(patch.object(piper.RobotWinPolicySession, '_denormalize_action',
            side_effect=lambda policy, action: action[:, piper.VALID_DIMS].numpy()[None]))
        self.sampler = self.stack.enter_context(patch.object(piper, 'infer_frame_joint_window',
            return_value={'action': self.normalized}))
        self.synchronize = self.stack.enter_context(patch.object(piper.torch.cuda, 'synchronize',
            side_effect=AssertionError('CPU policy must not call CUDA synchronize')))

    def policy(self, **kwargs):
        return piper.PiperPolicy('/unused', '/unused/run', 'fold towel', device='cpu', **kwargs)

    def observation(self):
        return {'state': np.zeros(14, np.float32), 'prompt': 'fold towel',
                **{key: np.zeros((2, 3, 3), np.uint8) for key in piper.IMAGE_KEYS}}

    def test_legacy_defaults_keep_saved_backend_and_video_reconstruction(self):
        policy = self.policy()
        policy.infer(self.observation())
        self.assertEqual(self.factory.call_args.args[0].model.mot_attention_backend, 'flex')
        self.assertTrue(self.cfg.model.mot_flex_masked_batches)
        self.assertTrue(self.sampler.call_args.kwargs['decode_video'])
        self.assertTrue(policy.audit['decode_video'])
        self.assertEqual(policy.audit['training_attention_backend'], 'flex')
        self.assertEqual(policy.audit['inference_attention_backend'], 'flex')
        self.assertFalse(policy.audit['rtc']['enabled'])
        self.synchronize.assert_not_called()

    def test_no_video_reconstruction_preserves_sampler_inputs_and_native_actions(self):
        legacy = self.policy()
        expected = legacy.infer(self.observation())
        legacy_kwargs = self.sampler.call_args.kwargs.copy()
        fast = self.policy(decode_video=False)
        actual = fast.infer(self.observation())
        kwargs = self.sampler.call_args.kwargs
        self.assertFalse(kwargs['decode_video'])
        self.assertFalse(fast.audit['decode_video'])
        self.assertFalse(fast.audit['rtc']['decode_video'])
        for key in ('num_frames', 'action_horizon', 'num_inference_steps', 'seed',
                    'rand_device', 'tiled', 'video_layout', 'video_view_names', 'rtc_guidance'):
            self.assertEqual(kwargs[key], legacy_kwargs[key])
        for key in ('context', 'context_mask', 'camera_is_pad', 'action_dim_is_pad',
                    'proprio_dim_is_pad', 'embodiment_ids'):
            self.assertTrue(torch.equal(kwargs[key], legacy_kwargs[key]))
        np.testing.assert_array_equal(actual['actions'], expected['actions'])
        np.testing.assert_array_equal(actual['normalized_actions'], expected['normalized_actions'])
        self.assertEqual(actual['actions'].shape, (32, 14))

    def test_explicit_sdpa_updates_runtime_and_audit(self):
        policy = self.policy(attention_backend='sdpa', decode_video=False)
        self.assertEqual(self.cfg.model.mot_attention_backend, 'sdpa')
        self.assertFalse(self.cfg.model.mot_flex_masked_batches)
        self.assertEqual(policy.audit['training_attention_backend'], 'flex')
        self.assertEqual(policy.audit['inference_attention_backend'], 'sdpa')
        self.assertEqual(policy.audit['rtc']['inference_attention_backend'], 'sdpa')

    def test_explicit_flex_is_supported(self):
        policy = self.policy(attention_backend='flex')
        self.assertEqual(policy.audit['inference_attention_backend'], 'flex')
        self.assertTrue(self.cfg.model.mot_flex_masked_batches)

    def test_legacy_rtc_defaults_to_sdpa_without_video_decode_and_forwards_guidance(self):
        policy = self.policy(rtc=True)
        guidance = {'prefix_action': torch.zeros(32, 80)}
        policy.infer(self.observation(), rtc_guidance=guidance)
        self.assertFalse(self.cfg.model.mot_flex_masked_batches)
        self.assertEqual(policy.audit['inference_attention_backend'], 'sdpa')
        self.assertFalse(self.sampler.call_args.kwargs['decode_video'])
        self.assertIs(self.sampler.call_args.kwargs['rtc_guidance'], guidance)

    def test_rtc_explicit_video_decode_is_respected(self):
        policy = self.policy(rtc=True, attention_backend='sdpa', decode_video=True)
        policy.infer(self.observation())
        self.assertTrue(self.sampler.call_args.kwargs['decode_video'])
        self.assertTrue(policy.audit['decode_video'])

    def test_invalid_options_fail_before_loading_model(self):
        for kwargs, message in (({'attention_backend': 'flash'}, 'attention_backend'),
                                ({'attention_backend': 'flex', 'rtc': True}, 'RTC guidance'),
                                ({'decode_video': 'false'}, 'decode_video')):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                self.policy(**kwargs)
        self.factory.assert_not_called()

    def test_static_device_tensors_are_reused_between_inferences(self):
        policy = self.policy(decode_video=False)
        policy.infer(self.observation())
        first = self.sampler.call_args.kwargs.copy()
        policy.infer(self.observation())
        second = self.sampler.call_args.kwargs
        for key in ('camera_is_pad', 'action_dim_is_pad', 'proprio_dim_is_pad', 'embodiment_ids'):
            self.assertIs(first[key], second[key])
            self.assertIs(first[key], getattr(policy, key))
            self.assertEqual(first[key].device.type, 'cpu')
        self.assertEqual(first['embodiment_ids'].tolist(), [14])
        self.assertEqual(first['camera_is_pad'].tolist(), [[False, False, False]])
        self.assertEqual((~first['action_dim_is_pad'][0]).nonzero().flatten().tolist(), piper.VALID_DIMS)
        self.assertEqual((~first['proprio_dim_is_pad'][0]).nonzero().flatten().tolist(), piper.VALID_DIMS)


if __name__ == '__main__':
    unittest.main()
