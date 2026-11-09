import csv
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from app import create_app
from catalog import Catalog
from engine import ReplayEngine, valid_rate
from hardware import Cancelled, DemoRobot, PiperRobot
from video import RecordingReader
from recording import ImageRecorder, recording_settings, CAMERAS
from paths import default_project_root


def wait_for(predicate, timeout=3):
    deadline = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out")
        time.sleep(.005)


class FakeCatalog:
    def __init__(self, length=20):
        self.length = length

    def listing(self):
        return []

    def load(self, source_id, episode, field="action"):
        target = np.repeat((np.arange(self.length)/100)[:, None], 14, axis=1)
        return dict(id=source_id, title="test", episode=episode, field=field, fps=30., frames=self.length,
                    task="test", original_unit="radians", parquet="fixture", videos=[], target=target,
                    timestamps=np.arange(self.length)/30, auxiliary=np.ones(self.length))


class RecorderRobot(DemoRobot):
    def __init__(self):
        super().__init__()
        self.sent = []

    def send(self, target):
        self.sent.append((time.monotonic(), np.asarray(target).copy()))
        return super().send(target)


def camera_fixture():
    return {key: dict(rgb=np.tile(np.array([11, 22, 33], dtype=np.uint8), (16, 16, 1)),
                      timestamp=time.monotonic(), frame_id=19, device_timestamp_ms=2000., device_frame_number=21)
            for key in CAMERAS}


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.robot = RecorderRobot()
        self.engine = ReplayEngine(FakeCatalog(), self.robot, Path(self.temp.name))

    def tearDown(self):
        self.engine.close()
        self.temp.cleanup()

    def ready(self):
        self.engine.submit("select", dict(dataset="test", episode=0))
        wait_for(lambda: self.engine.episode is not None and not self.engine.busy)
        self.engine.submit("connect")
        wait_for(lambda: self.robot.connected and not self.engine.busy)

    def test_full_recording_three_rgb_views_state_commands_and_timestamps(self):
        self.robot.capture_frames = camera_fixture
        self.engine.submit('recording', dict(mode='full', image_hz=30))
        wait_for(lambda: self.engine.record_settings['mode'] == 'full' and not self.engine.busy)
        self.ready()
        self.engine.submit('start', dict(hz=100, align_seconds=0))
        wait_for(lambda: self.engine.snapshot()['phase'] == 'completed')
        wait_for(lambda: not self.engine.snapshot()['recording']['saving'])
        run = self.engine.log_path.parent
        commands = [json.loads(line) for line in (run / 'commands.jsonl').read_text().splitlines()]
        self.assertEqual([x['source_frame'] for x in commands], list(range(20)))
        np.testing.assert_array_equal([x['target'] for x in commands], self.engine.episode['target'])
        feedback = [json.loads(line) for line in (run / 'feedback.jsonl').read_text().splitlines()]
        by_seq = {x['seq']: x for x in feedback}
        frames = sorted((run / 'frames').glob('*.json'))
        self.assertTrue(frames)
        for file in frames:
            meta = json.loads(file.read_text())
            sample = by_seq[meta['seq']]
            self.assertGreaterEqual(meta['state_age_seconds'], 0)
            self.assertEqual(meta['auxiliary_15'], 1.)
            self.assertEqual(meta['missing_cameras'], [])
            self.assertEqual(meta['color_order'], 'RGB')
            with np.load(file.with_suffix('.npz'), allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['state'], sample['current'])
                np.testing.assert_array_equal(arrays['action'], sample['target'])
                np.testing.assert_array_equal(arrays['driver_action'], sample['driver_target'])
                for key in CAMERAS.values():
                    self.assertEqual(arrays[key].shape, (16, 16, 3))
                    np.testing.assert_array_equal(arrays[key][0, 0], [33, 22, 11])
        stats = json.loads((run / 'summary.json').read_text())['recording']
        self.assertFalse(stats['active'])
        self.assertEqual(stats['images']['written'], len(frames))
        self.assertEqual(stats['images']['complete_groups'], len(frames))
        self.assertEqual(stats['images']['images_written'], 3*len(frames))
        self.assertEqual(stats['recorded_actions'], 20)
        self.assertEqual(stats['images']['unfinished'], 0)
        self.assertEqual(stats['lerobot']['status'], 'complete')
        info = json.loads((run/'meta/info.json').read_text())
        self.assertEqual(info['codebase_version'], 'v2.1')
        self.assertEqual(info['total_videos'], 3)

    def test_recording_configuration_locks_during_pause_and_missing_cameras_are_explicit(self):
        self.ready()
        self.engine.submit('recording', dict(mode='full', image_hz=30))
        wait_for(lambda: not self.engine.busy)
        self.engine.submit('goto_frame', dict(frame=2, align_seconds=0))
        wait_for(lambda: self.engine.snapshot()['recording']['images'] and
                 self.engine.snapshot()['recording']['images']['written'] >= 1)
        self.engine.submit('pause')
        wait_for(lambda: self.engine.phase == 'paused')
        with self.assertRaisesRegex(ValueError, '停止'):
            self.engine.submit('recording', dict(mode='actions'))
        self.engine.submit('stop')
        wait_for(lambda: not self.engine.snapshot()['recording']['active'])
        wait_for(lambda: not self.engine.snapshot()['recording']['saving'])
        stats = self.engine.snapshot()['recording']['images']
        self.assertEqual(stats['missing_camera_groups'], stats['written'])
        self.assertEqual(stats['complete_groups'], 0)
        self.assertEqual(stats['images_written'], 0)
        self.assertEqual(self.engine.snapshot()['recording']['lerobot']['status'], 'error')
        file = next((self.engine.log_path.parent / 'frames').glob('*.json'))
        self.assertEqual(set(json.loads(file.read_text())['missing_cameras']), set(CAMERAS))
        with np.load(file.with_suffix('.npz')) as arrays:
            self.assertNotIn('image', arrays.files)
        self.engine.submit('recording', dict(mode='actions', image_hz=10))
        wait_for(lambda: self.engine.record_settings['mode'] == 'actions')

    def test_image_disk_failure_stops_demo_and_reports_error(self):
        self.robot.capture_frames = camera_fixture
        self.ready()
        self.engine.submit('recording', dict(mode='full', image_hz=30))
        wait_for(lambda: not self.engine.busy)
        with patch.object(ImageRecorder, '_write', side_effect=OSError('disk full fixture')):
            self.engine.submit('start', dict(hz=10, align_seconds=0))
            wait_for(lambda: self.engine.snapshot()['phase'] == 'error')
            wait_for(lambda: not self.engine.snapshot()['recording']['saving'])
        self.assertFalse(self.robot.enabled)
        self.assertIn('disk full fixture', self.engine.error)
        stats = json.loads((self.engine.log_path.parent / 'summary.json').read_text())['recording']
        self.assertIn('disk full fixture', stats['images']['error'])
        self.assertFalse(stats['active'])

    def test_recording_http_validation(self):
        app = create_app(self.engine, Mock(errors={}))
        client = app.test_client()
        headers = {'X-Replay-Token': app.config['REPLAY_TOKEN']}
        for values in ({'mode': 'bad'}, {'mode': 'full', 'image_hz': 0}, {'mode': 'full', 'image_hz': 31}):
            self.assertEqual(client.post('/api/command/recording', json=values, headers=headers).status_code, 400)
        for mode, hz in (('bad', 10), ('full', float('nan')), ('actions', float('inf'))):
            with self.assertRaises(ValueError):
                recording_settings(mode, hz)
        html = client.get('/').get_data(as_text=True)
        for name in ('recordMode', 'recordImageHz', 'recordOutputDir', 'applyRecording', 'recordingStatus'):
            self.assertIn('id="' + name + '"', html)

    def test_select_output_directory_and_export_there(self):
        destination = Path(self.temp.name)/'selected output'
        app = create_app(self.engine, Mock(errors={}))
        client = app.test_client()
        headers = {'X-Replay-Token': app.config['REPLAY_TOKEN']}
        values = dict(mode='actions', image_hz=10, output_dir=str(destination))
        response = client.post('/api/command/recording', json=values, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['normalized']['output_dir'], str(destination))
        wait_for(lambda: not self.engine.busy)
        self.assertEqual(self.engine.output, destination)
        self.assertFalse(self.robot.enabled)
        self.ready()
        self.engine.submit('start', dict(hz=100, align_seconds=0))
        wait_for(lambda: self.engine.phase == 'completed')
        wait_for(lambda: not self.engine.snapshot()['recording']['saving'])
        status = self.engine.snapshot()['recording']
        self.assertEqual(status['lerobot']['status'], 'complete')
        self.assertEqual(Path(status['directory']).parent, destination)
        self.assertTrue((Path(status['directory'])/'meta/info.json').is_file())
        self.assertEqual(ReplayEngine._recording_output('runs'), Path(__file__).resolve().parent/'runs')
        for path in ('', str(self.engine.log_path)):
            self.assertEqual(client.post('/api/command/recording', json={**values, 'output_dir': path}, headers=headers).status_code, 400)

    def test_slow_image_writer_does_not_delay_actions_or_disable(self):
        self.robot.capture_frames = camera_fixture
        self.ready()
        self.engine.submit('recording', dict(mode='full', image_hz=30))
        wait_for(lambda: not self.engine.busy)
        started, release = threading.Event(), threading.Event()
        original = ImageRecorder._write
        def delayed(recorder, *args):
            started.set()
            if not release.wait(5):
                raise RuntimeError('test writer not released')
            original(recorder, *args)
        with patch.object(ImageRecorder, '_write', new=delayed):
            try:
                self.engine.submit('start', dict(hz=30, align_seconds=0))
                self.assertTrue(started.wait(2))
                wait_for(lambda: self.engine.snapshot()['phase'] == 'completed')
                status = self.engine.snapshot()['recording']
                self.assertTrue(status['saving'])
                self.assertEqual(len(self.robot.sent), 20)
                self.assertGreater(status['images']['dropped'], 0)
                self.assertLessEqual(status['images']['submitted'], 5)
                with self.assertRaisesRegex(ValueError, '保存'):
                    self.engine.submit('start', dict(hz=30, align_seconds=0))
                # Saving must not hold the control mutex or postpone disable.
                self.engine.submit('disable')
                wait_for(lambda: self.engine.phase == 'disabled' and not self.engine.busy, timeout=1)
                self.assertFalse(self.robot.enabled)
                self.assertFalse(release.is_set())
            finally:
                release.set()
        wait_for(lambda: not self.engine.snapshot()['recording']['saving'])
        stats = self.engine.snapshot()['recording']['images']
        self.assertEqual(stats['written'], stats['submitted'])
        self.assertTrue(stats['closed'])

    def test_selection_and_connection_send_no_motion(self):
        self.ready()
        self.engine.submit("seek", dict(frame=5))
        wait_for(lambda: self.engine.frame == 5 and not self.engine.busy)
        self.assertEqual(self.robot.sent, [])
        self.assertFalse(self.robot.enabled)

    def test_mit_configuration_is_recorded_and_does_not_change_dataset_targets(self):
        settings = dict(mode="mit", kp=19., kd=1.2, robot_rate=230.,
                        vel_ref=.1, t_ref=.2, speed_limit=65)
        self.engine.submit("control", settings)
        wait_for(lambda: self.robot.control.mode == "mit" and not self.engine.busy)
        self.assertEqual(self.robot.sent, [])
        self.assertFalse(self.robot.connected)
        self.assertEqual(self.engine.snapshot()["control_settings"], settings)
        self.ready()
        with self.assertRaisesRegex(ValueError, "断开"):
            self.engine.submit("control", {"mode": "joint"})
        self.engine.submit("start", dict(hz=100, align_seconds=0))
        wait_for(lambda: self.engine.phase == "completed")
        np.testing.assert_array_equal(np.array([x[1] for x in self.robot.sent]), self.engine.episode["target"])
        saved = json.loads((self.engine.log_path.parent / "run.json").read_text())
        self.assertEqual(saved["robot_control"]["mode"], "mit")
        self.assertEqual(saved["robot_control"]["mit"]["vel_ref"], .1)
        self.assertEqual(saved["controller_hz"], 230)
        self.engine.submit("disconnect")
        wait_for(lambda: not self.robot.connected and not self.engine.busy)
        self.engine.submit("control", {"mode": "joint"})
        wait_for(lambda: self.robot.control.mode == "joint" and not self.engine.busy)
        self.assertEqual(self.robot.control.controller_hz, 100)

    def test_mit_http_validation_and_ui_controls(self):
        app = create_app(self.engine, Mock(errors={}))
        client = app.test_client()
        headers = {"X-Replay-Token": app.config["REPLAY_TOKEN"]}
        for values in ({"mode": "other"}, {"mode": "mit", "kp": 501}, {"mode": "mit", "t_ref": 9}):
            self.assertEqual(client.post("/api/command/control", json=values, headers=headers).status_code, 400)
        self.assertEqual(client.post("/api/command/control", json={"mode": "mit"}, headers=headers).status_code, 200)
        wait_for(lambda: self.robot.control.mode == "mit" and not self.engine.busy)
        status = client.get("/api/status").get_json()
        self.assertEqual(status["robot_control"]["joint_api"], "JointMitCtrl")
        html = client.get("/").get_data(as_text=True)
        for field in ("controlMode", "mitKp", "mitKd", "mitVelRef", "mitTRef", "mitRobotRate", "mitSpeedLimit"):
            self.assertIn('id="' + field + '"', html)
        self.assertEqual(self.robot.sent, [])

    def test_replay_exact_samples_at_selected_rate_and_record_feedback(self):
        self.ready()
        self.engine.submit("start", dict(hz=60, align_seconds=0))
        wait_for(lambda: self.engine.phase == "completed")
        np.testing.assert_array_equal(np.array([x[1] for x in self.robot.sent]), self.engine.episode["target"])
        elapsed = self.robot.sent[-1][0]-self.robot.sent[0][0]
        self.assertGreaterEqual(elapsed, 19/60*.95)
        self.assertLess(elapsed, 19/60+.15)
        self.assertTrue(self.engine.samples)
        with self.engine.log_path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        self.assertIn("feedback_right_j6", rows[0])
        self.assertIn("driver_target_left_gripper", rows[0])

    def test_pause_hold_resume_does_not_drop_or_repeat_dataset_frames(self):
        self.ready()
        self.engine.submit("start", dict(hz=30, align_seconds=0))
        wait_for(lambda: len(self.robot.sent) >= 3)
        self.engine.submit("pause")
        wait_for(lambda: self.engine.phase == "paused")
        count = len(self.robot.sent)
        time.sleep(.08)
        self.assertEqual(len(self.robot.sent), count)
        self.engine.submit("start", dict(hz=100, align_seconds=0))
        wait_for(lambda: self.engine.phase == "completed")
        np.testing.assert_array_equal(np.array([x[1] for x in self.robot.sent]), self.engine.episode["target"])

    def test_cancel_during_enable_prevents_trajectory_commands(self):
        self.ready()
        started = threading.Event()
        def slow_enable(cancel):
            started.set()
            while not cancel.wait(.005):
                pass
            raise Cancelled("cancelled")
        self.robot.enable = slow_enable
        self.engine.submit("start", dict(hz=30, align_seconds=0))
        self.assertTrue(started.wait(1))
        self.engine.submit("disable")
        wait_for(lambda: self.engine.phase == "disabled")
        self.assertEqual(self.robot.sent, [])

    def test_alignment_is_separate_and_pause_cannot_skip_its_remainder(self):
        self.ready()
        self.engine.submit("seek", dict(frame=10))
        wait_for(lambda: self.engine.frame == 10 and not self.engine.busy)
        self.engine.submit("start", dict(hz=60, align_seconds=.2))
        wait_for(lambda: self.engine.phase == "aligning" and len(self.robot.sent)>2)
        self.engine.submit("pause")
        wait_for(lambda: self.engine.phase == "paused")
        self.engine.submit("start", dict(hz=60, align_seconds=.2))
        wait_for(lambda: self.engine.phase == "aligning")
        wait_for(lambda: self.engine.phase == "completed")
        np.testing.assert_array_equal(np.array([v for _, v in self.robot.sent[-10:]]), self.engine.episode["target"][10:])

    def test_rate_and_live_selection_validation(self):
        for value in (0, -2, float("inf"), float("nan"), 101):
            with self.assertRaises(ValueError):
                valid_rate(value)
        self.ready()
        self.engine.submit("start", dict(hz=15, align_seconds=0))
        wait_for(lambda: self.engine.phase == "playing")
        with self.assertRaises(ValueError):
            self.engine.submit("seek", dict(frame=3))
        self.engine.submit("rate", dict(hz=30))
        wait_for(lambda: self.engine.hz == 30)

    def test_completed_disable_restart_and_summary_preservation(self):
        self.ready()
        self.engine.submit("start", dict(hz=100, align_seconds=0))
        wait_for(lambda: self.engine.phase == "completed")
        wait_for(lambda: not self.engine.snapshot()['recording']['saving'])
        original_summary = self.engine.log_path.parent / "summary.json"
        saved = original_summary.read_text()
        self.engine.submit("disable")
        wait_for(lambda: self.engine.phase == "disabled")
        self.engine.submit("start", dict(hz=100, align_seconds=.05))
        wait_for(lambda: self.engine.phase == "aligning")
        wait_for(lambda: self.engine.phase == "completed")
        self.assertEqual(self.engine.sent_frames, 20)
        self.assertEqual(original_summary.read_text(), saved)
        self.engine.submit("select", dict(dataset="test", episode=1))
        wait_for(lambda: self.engine.episode["episode"] == 1 and not self.engine.busy)
        self.assertEqual(self.engine.sent_frames, 0)

    def test_single_frame_holds_exact_target_and_continues_feedback_recording(self):
        self.ready()
        self.engine.submit("goto_frame", dict(frame=9, align_seconds=0))
        wait_for(lambda: self.engine.phase == "holding")
        count = len(self.robot.sent)
        samples = self.engine._sequence
        time.sleep(.12)
        self.assertEqual(len(self.robot.sent), count)
        self.assertEqual(self.engine.frame, 9)
        self.assertEqual(self.engine.next_frame, 9)
        np.testing.assert_array_equal(self.engine.target, self.engine.episode["target"][9])
        self.assertGreater(self.engine._sequence, samples)
        self.assertFalse(self.engine.log_handle.closed)
        self.assertEqual(self.engine.last_run["operation"], "single_frame")
        self.engine.submit("start", dict(hz=100, align_seconds=0))
        wait_for(lambda: self.engine.phase == "completed")
        np.testing.assert_array_equal(np.array([v for _,v in self.robot.sent[count:]]),self.engine.episode["target"][9:])

    def test_single_frame_transition_finishes_at_requested_frame_without_autoplay(self):
        self.ready()
        self.engine.submit("goto_frame", dict(frame=19, align_seconds=.1))
        wait_for(lambda: self.engine.phase == "moving")
        with self.assertRaises(ValueError):
            self.engine.submit("goto_frame", dict(frame=2, align_seconds=0))
        wait_for(lambda: self.engine.phase == "holding")
        count=len(self.robot.sent)
        time.sleep(.07)
        self.assertEqual(len(self.robot.sent),count)
        self.assertEqual(self.engine.sent_frames,1)
        np.testing.assert_array_equal(self.robot.sent[-1][1],self.engine.episode["target"][19])
        self.engine.submit("goto_frame", dict(frame=2, align_seconds=.05))
        # The next frame is assigned before the command leaves the old holding phase.
        wait_for(lambda: not self.engine.busy and self.engine.frame==2 and self.engine.phase == "holding")
        np.testing.assert_array_equal(self.engine.target,self.engine.episode["target"][2])

    def test_single_frame_cancel_and_invalid_frame(self):
        self.ready()
        for frame in (-1,20,1.5,True):
            with self.assertRaises(ValueError):
                self.engine.submit("goto_frame", dict(frame=frame, align_seconds=0))
        self.engine.submit("goto_frame", dict(frame=19, align_seconds=.5))
        wait_for(lambda: self.engine.phase == "moving" and len(self.robot.sent)>2)
        self.engine.submit("disable")
        wait_for(lambda: self.engine.phase == "disabled")
        count=len(self.robot.sent)
        time.sleep(.08)
        self.assertEqual(len(self.robot.sent),count)
        self.assertFalse(self.robot.enabled)

    def test_http_requires_token_and_rejects_cross_origin(self):
        frames = Mock(errors={})
        app = create_app(self.engine, frames)
        client = app.test_client()
        self.assertEqual(client.post("/api/command/connect", json={}).status_code, 403)
        token = app.config["REPLAY_TOKEN"]
        self.assertEqual(client.post("/api/command/connect", json={}, headers={"X-Replay-Token":token,"Origin":"http://evil.example"}).status_code,403)
        self.assertFalse(self.robot.connected)
        self.assertEqual(client.post("/api/command/connect", json={}, headers={"X-Replay-Token":token}).status_code,200)
        wait_for(lambda:self.robot.connected)
        self.assertFalse(self.robot.enabled)
        self.assertEqual(client.get("/api/status?after=-1").status_code,200)

    def test_hardware_joint_conversion_and_stop_order_without_real_device(self):
        robot = PiperRobot.__new__(PiperRobot)
        robot.enabled = robot.enable_attempted = True
        events=[]
        robot.controllers=[]
        for name in ("left","right"):
            c=Mock(can_interface=name)
            c.get_target_joint_angles.return_value=np.array([10,20,30,40,50,60],float)
            c.get_gripper_open_value.return_value=.7
            c.latch_emergency_stop.side_effect=lambda name=name:(events.append(name+" latch") or True)
            c.finish_emergency_disable.side_effect=lambda name=name:(events.append(name+" disable") or True)
            robot.controllers.append(c)
        target=np.arange(14,dtype=float)/10
        robot.send(target)
        np.testing.assert_allclose(robot.controllers[1].set_target_joint_angles.call_args.args[0],np.rad2deg(target[7:13]))
        self.assertEqual(robot.controllers[0].set_gripper_open_value.call_args.args[0],target[6])
        robot.disable()
        self.assertEqual(events,["left latch","right latch","left disable","right disable"])


class OriginalDataTests(unittest.TestCase):
    def test_original_field_units_and_auxiliary_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory);root=project/'datasets/test'
            (root/'meta').mkdir(parents=True);(root/'data').mkdir()
            manifest=project/'resources/piper/dataset_v3/meta';manifest.mkdir(parents=True)
            spec=dict(root='datasets/test',state_key='state',action_key='actions',joint_unit='degrees',cameras=['a','b','c'])
            (manifest/'piper_interface.json').write_text(json.dumps(dict(sources=[spec])))
            (root/'meta/info.json').write_text(json.dumps(dict(fps=30,chunks_size=1000,data_path='data/{episode_index}.parquet',video_path='{video_key}.mp4')))
            (root/'meta/episodes.jsonl').write_text(json.dumps(dict(episode_index=3,length=2,tasks=['test'])))
            raw=np.ones((2,15))*90;raw[:,[6,13]]=.4;raw[:,14]=7
            pq.write_table(pa.table(dict(state=raw.tolist(),actions=(raw*.5).tolist(),frame_index=[0,1],timestamp=[0.,1/30])),root/'data/3.parquet')
            for name in ('a','b','c'):(root/f'{name}.mp4').touch()
            data=Catalog(project).load('test',3,'action')
            self.assertAlmostEqual(data['target'][0,0],np.pi/4)
            self.assertEqual(data['target'][0,6],.2)
            self.assertEqual(data['auxiliary'][0],3.5)
            self.assertEqual(data['target'].shape,(2,14))

    def test_original_three_video_views_can_seek_forward_and_back(self):
        root=default_project_root()
        if not (root/'resources/piper/dataset_v3/meta/piper_interface.json').is_file():
            self.skipTest('Optional local recordings are not included in the repository')
        if not (root/'datasets/piper_set_placemat_lerobot/meta/info.json').is_file():
            self.skipTest('Optional placemat recordings unavailable')
        episode=Catalog(root).load('piper_set_placemat_lerobot',0)
        for path in episode['videos']:
            reader=RecordingReader(path,30)
            try:
                first=reader.get(0).copy()
                self.assertEqual(reader.get(50).shape,(480,640,3))
                np.testing.assert_array_equal(reader.get(0),first)
            finally:reader.close()


if __name__ == '__main__':
    unittest.main()
