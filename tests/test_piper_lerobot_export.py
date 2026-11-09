"""LeRobot v2.1 round trips; synthetic observations, no hardware or policy."""
import json
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
import pyarrow.parquet as pq

from wam.inference.piper_lerobot_export import export_run, IMAGES


def replay_fixture(root, full=True):
    (root/'run.json').write_text(json.dumps({'episode': {'task': 'fixture task'}}))
    samples = [dict(seq=i, current=[i/10]*14, target=[1.]*14, driver_target=[.9]*14,
                    frame=i, auxiliary_15=42.+i, state_read_monotonic=100+i*.1) for i in range(4)]
    (root/'feedback.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
    commands = [dict(monotonic=100., target=[1.]*14, driver_target=[.9]*14),
                dict(monotonic=100.07, target=[2.]*14, driver_target=[1.9]*14)]
    (root/'commands.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in commands))
    if full:
        (root/'frames').mkdir()
        for i in range(2):
            path = root/'frames'/f'frame_{i:08d}'
            path.with_suffix('.json').write_text(json.dumps(dict(capture_monotonic=100+i*.2, missing_cameras=[])))
            colors = ([220, 20, 20], [20, 220, 20], [20, 20, 220])
            np.savez(path.with_suffix('.npz'), **{key: np.tile(np.uint8(color), (16, 16, 1)) for key, color in zip(IMAGES, colors)})


class LeRobotExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_replay_v21_parquet_video_counts_timestamps_and_color(self):
        replay_fixture(self.root)
        result = export_run(self.root, kind='replay', mode='full', fps=30)
        self.assertEqual(result['frames'], 10)
        info = json.loads((self.root/'meta/info.json').read_text())
        self.assertEqual(info['codebase_version'], 'v2.1')
        self.assertEqual(info['total_videos'], 3)
        rows = pq.read_table(self.root/'data/chunk-000/episode_000000.parquet').to_pylist()
        np.testing.assert_allclose([r['timestamp'] for r in rows], np.arange(10)/30, atol=1e-6)
        self.assertEqual([r['actions'][0] for r in rows[:4]], [1., 1., 1., 2.])
        self.assertAlmostEqual(rows[2]['state'][0], 0.)
        self.assertGreater(rows[2]['state_age_seconds'], .06)
        self.assertAlmostEqual(rows[3]['state'][0], .1)
        self.assertEqual(rows[3]['auxiliary_15'], 43.)
        self.assertTrue(rows[3]['auxiliary_15_valid'])
        self.assertGreater(rows[5]['camera_age_seconds'], .15)
        for channel, key in enumerate(IMAGES):
            path = self.root/'videos/chunk-000'/key/'episode_000000.mp4'
            with av.open(str(path)) as video:
                frames = list(video.decode(video=0))
                self.assertEqual(len(frames), len(rows))
                self.assertAlmostEqual(float(frames[-1].pts*frames[-1].time_base), .3, places=5)
                color = frames[0].to_ndarray(format='rgb24').mean((0, 1))
                self.assertEqual(int(color.argmax()), channel)
                self.assertGreater(color[channel], 200)
        self.assertTrue((self.root/'feedback.jsonl').exists())
        self.assertTrue((self.root/'meta/episodes_stats.jsonl').exists())

    def test_load_with_project_lerobot_v21_reader_including_videos(self):
        replay_fixture(self.root)
        export_run(self.root, kind='replay', mode='full', fps=30)
        from wam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata('local/recording-fixture', root=self.root)
        self.assertEqual(meta.total_frames, 10)
        self.assertEqual(meta.total_episodes, 1)
        self.assertEqual(meta.video_keys, list(IMAGES))
        self.assertEqual(tuple(meta.stats['state']['mean'].shape), (14,))
        dataset = LeRobotDataset('local/recording-fixture', root=self.root, video_backend='pyav')
        for i in (0, 5, 9):
            row = dataset[i]
            self.assertEqual(tuple(row['state'].shape), (14,))
            self.assertEqual(tuple(row['image'].shape), (3, 16, 16))

    def test_policy_preserves_prediction_horizon_and_labels_semantics(self):
        for i in range(2):
            np.savez(self.root/f'chunk_{i:06d}.npz', state=np.full(14, i, np.float32), prompt='fixture task',
                     actions=np.arange(32*14, dtype=np.float32).reshape(32, 14)+i)
            (self.root/f'chunk_{i:06d}.json').write_text(json.dumps(dict(
                observation_monotonic=100.+i*.5, request_tick=i*15, publish_tick=i*15+10, accepted_plan=i==0)))
        result = export_run(self.root, kind='policy', mode='actions')
        self.assertEqual(result['frames'], 16)
        self.assertIn('not_executed', result['action_semantics'])
        rows = pq.read_table(self.root/'data/chunk-000/episode_000000.parquet').to_pylist()
        self.assertEqual(np.asarray(rows[0]['predicted_actions']).shape, (32, 14))
        self.assertEqual(rows[-1]['actions'][0], 1.)
        self.assertFalse(rows[-1]['accepted_plan'])
        from wam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset('local/policy-recording-fixture', root=self.root)
        self.assertEqual(tuple(dataset[0]['predicted_actions'].shape), (32, 14))

    def test_no_camera_and_no_timestamp_fail_explicitly_without_fake_dataset(self):
        replay_fixture(self.root, full=False)
        with self.assertRaisesRegex(ValueError, 'three-camera'):
            export_run(self.root, kind='replay', mode='full')
        self.assertFalse((self.root/'meta').exists())
        self.assertTrue((self.root/'feedback.jsonl').exists())
        (self.root/'chunk_000000.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            export_run(self.root, kind='policy', mode='actions')

    def test_custom_destination_and_no_overwrite(self):
        replay_fixture(self.root, full=False)
        out = self.root/'custom data'
        export_run(self.root, kind='replay', mode='actions', output=out)
        old = (out/'meta/info.json').read_bytes()
        with self.assertRaises(FileExistsError):
            export_run(self.root, kind='replay', mode='actions', output=out)
        self.assertEqual((out/'meta/info.json').read_bytes(), old)
        self.assertFalse((out/'videos').exists())


if __name__ == '__main__':
    unittest.main()
