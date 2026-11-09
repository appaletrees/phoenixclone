"""Export Piper run records as one LeRobot v2.1 episode, without model imports.

The uniform dataset clock uses causal sample-and-hold. Original sample times
and ages remain explicit features; repeated observations are not new readings.
Replay actions are submitted targets. Policy actions are prediction[0], NOT
reconstructed executed actions; the full prediction is also retained.
"""
from bisect import bisect_right
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np


IMAGES = ('image', 'wrist_image_l', 'wrist_image_r')
JOINTS = [f'{arm}_joint_{i}' for arm in ('left', 'right') for i in range(1, 8)]
JOINTS[6], JOINTS[13] = 'left_gripper', 'right_gripper'
DATA_PATH = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
VIDEO_PATH = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'


def _json(path):
    return json.loads(Path(path).read_text())


def _lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _feature(dtype, shape=(1,), names=None):
    return dict(dtype=dtype, shape=list(shape), names=names)


class _Moments:
    def __init__(self):
        self.n = 0

    def add(self, value):
        x = np.asarray(value, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Cannot export non-finite sample')
        if not self.n:
            self.mean, self.m2, self.lo, self.hi = x.copy(), np.zeros_like(x), x.copy(), x.copy()
        else:
            delta = x-self.mean
            self.mean += delta/(self.n+1)
            self.m2 += delta*(x-self.mean)
            self.lo = np.minimum(self.lo, x)
            self.hi = np.maximum(self.hi, x)
        self.n += 1

    def result(self):
        return dict(mean=np.atleast_1d(self.mean).tolist(), std=np.atleast_1d(np.sqrt(np.maximum(0, self.m2/self.n))).tolist(),
                    min=np.atleast_1d(self.lo).tolist(), max=np.atleast_1d(self.hi).tolist(), count=[self.n])


class _Video:
    def __init__(self, path, pixels, fps):
        import av
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError('Expected RGB uint8[height,width,3]')
        self.shape = pixels.shape
        if pixels.shape[0] % 2 or pixels.shape[1] % 2:
            raise ValueError('H264 yuv420p requires even image dimensions')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), 'w')
        self.stream = self.container.add_stream('libx264', rate=Fraction(str(fps)))
        self.stream.width, self.stream.height = pixels.shape[1], pixels.shape[0]
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': '18', 'preset': 'veryfast', 'bf': '0'}
        self.stream.codec_context.thread_count = 1
        self.n = 0
        self.pixel_n = 0
        self.sum = self.square = self.lo = self.hi = None

    def add(self, pixels):
        import av
        if pixels.shape != self.shape or pixels.dtype != np.uint8:
            raise ValueError('Camera shape/dtype changed within episode')
        frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
        frame.pts = self.n
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        # Spatial subsampling bounds statistics cost, as in LeRobot's stats.
        p = pixels[::16, ::16].reshape(-1, 3).astype(np.float64)/255
        s, q, lo, hi = p.sum(0), (p*p).sum(0), p.min(0), p.max(0)
        if self.n == 0:
            self.sum, self.square, self.lo, self.hi = s, q, lo, hi
        else:
            self.sum += s
            self.square += q
            self.lo = np.minimum(self.lo, lo)
            self.hi = np.maximum(self.hi, hi)
        self.pixel_n += len(p)
        self.n += 1

    def stats(self):
        mean = self.sum/self.pixel_n
        values = dict(mean=mean, std=np.sqrt(np.maximum(0, self.square/self.pixel_n-mean*mean)),
                      min=self.lo, max=self.hi)
        return {**{k: v.reshape(3, 1, 1).tolist() for k, v in values.items()}, 'count': [self.n]}

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


def _sources(run, kind, full):
    if kind == 'replay':
        states = [s for s in _lines(run/'feedback.jsonl') if s['target'] is not None]
        samples = [(float(s['state_read_monotonic']), s) for s in states]
        commands = [(float(s['monotonic']), s) for s in _lines(run/'commands.jsonl') if s['target'] is not None]
        images = []
        if full:
            for path in sorted((run/'frames').glob('frame_*.json')):
                s = _json(path)
                if not s.get('missing_cameras'):
                    images.append((float(s['capture_monotonic']), path.with_suffix('.npz')))
        return samples, commands, images
    if kind != 'policy':
        raise ValueError('kind must be replay or policy')
    samples = []
    for path in sorted(run.glob('chunk_*.json')):
        s = _json(path)
        stamp = s.get('observation_monotonic')
        if stamp is None:
            stamp = (s.get('sensor_timing') or {}).get('read_monotonic')
        if stamp is None:
            raise ValueError('Prediction record lacks observation timestamp; cannot invent a time axis')
        if not path.with_suffix('.npz').is_file():
            raise FileNotFoundError(path.with_suffix('.npz'))
        samples.append((float(stamp), dict(s, path=path.with_suffix('.npz'))))
    return samples, [], [(t, s['path']) for t, s in samples] if full else []


def export_run(run, *, kind, mode, fps=30., output=None):
    """Export into run/{meta,data,videos}; existing datasets are never replaced.

    Raw diagnostics remain available for recovery and exact event timing.
    This runs only after recording ends, outside the robot control thread.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    fps = float(fps)
    if not math.isfinite(fps) or not 1 <= fps <= 100:
        raise ValueError('LeRobot FPS must be in 1..100')
    if mode not in ('actions', 'full'):
        raise ValueError('LeRobot export requires actions or full recording')
    run = Path(run).resolve()
    output = Path(output).expanduser().resolve() if output else run
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in ('meta', 'data', 'videos')):
        raise FileExistsError(f'LeRobot dataset already exists: {output}')
    samples, commands, images = _sources(run, kind, mode == 'full')
    if not samples or (kind == 'replay' and not commands):
        raise ValueError('No recorded state/action samples to export')
    if mode == 'full' and not images:
        raise ValueError('No complete three-camera sample; raw records retained, no fabricated video')
    for sequence in (samples, commands, images):
        if any(not math.isfinite(t) for t, _ in sequence) or any(b[0] < a[0] for a, b in zip(sequence, sequence[1:])):
            raise ValueError('Source timestamps must be finite and monotonic')
    receipt = _json(run/'run.json') if (run/'run.json').exists() else {}
    prompt = receipt.get('episode', {}).get('task') if kind == 'replay' else receipt.get('options', {}).get('prompt')
    if not prompt and kind == 'policy':
        with np.load(samples[0][1]['path'], allow_pickle=False) as saved:
            prompt = str(saved['prompt'].item())
    if not prompt:
        raise ValueError('Missing recorded task instruction')
    start = max(samples[0][0], images[0][0] if images else samples[0][0],
                commands[0][0] if commands else samples[0][0])
    end = samples[-1][0]
    if start > end:
        raise ValueError('No overlapping image/state interval')
    count = int(math.floor((end-start)*fps+1e-6))+1
    features = {key: _feature('int64') for key in ('index', 'episode_index', 'frame_index', 'task_index')}
    features.update(timestamp=_feature('float32'), state=_feature('float32', (14,), JOINTS),
                    actions=_feature('float32', (14,), JOINTS),
                    source_sample_index=_feature('int64'), source_timestamp=_feature('float64'),
                    state_sample_timestamp=_feature('float64'), state_age_seconds=_feature('float64'))
    if kind == 'replay':
        features.update(driver_actions=_feature('float32', (14,), JOINTS), source_frame_index=_feature('int64'))
        if any(s.get('auxiliary_15') is not None for _, s in samples):
            features.update(auxiliary_15=_feature('float32'), auxiliary_15_valid=_feature('bool'))
    else:
        features.update(predicted_actions=_feature('float32', (32, 14), ['horizon', 'action_dimension']),
                        request_tick=_feature('int64'), publish_tick=_feature('int64'), accepted_plan=_feature('bool'))
    if images:
        features.update(camera_sample_timestamp=_feature('float64'), camera_age_seconds=_feature('float64'))
    fields = []
    for key, feature in features.items():
        dtype = {'float32': pa.float32(), 'float64': pa.float64(), 'int64': pa.int64(), 'bool': pa.bool_()}[feature['dtype']]
        if feature['shape'] != [1]:
            for size in reversed(feature['shape']):
                dtype = pa.list_(dtype, size)
        fields.append(pa.field(key, dtype, nullable=False))
    schema = pa.schema(fields)
    stats = {key: _Moments() for key in features}
    stage = output/('.lerobot-'+uuid4().hex+'.partial')
    stage.mkdir()
    parquet = stage/DATA_PATH.format(episode_chunk=0, episode_index=0)
    parquet.parent.mkdir(parents=True)
    videos, cache = {}, {}
    writer = None
    committed = []
    try:
        writer = pq.ParquetWriter(parquet, schema, compression='zstd')
        times, command_times, image_times = [t for t, _ in samples], [t for t, _ in commands], [t for t, _ in images]
        previous_sample = previous_image = -1
        batch = []
        for i in range(count):
            t = start+i/fps
            index = max(0, bisect_right(times, t)-1)
            stamp, sample = samples[index]
            if kind == 'replay':
                command = commands[max(0, bisect_right(command_times, t)-1)][1]
                state, action = sample['current'], command['target']
                extra = dict(driver_actions=command['driver_target'], source_frame_index=sample['frame'])
                if 'auxiliary_15' in features:
                    aux = sample.get('auxiliary_15')
                    extra.update(auxiliary_15=0. if aux is None else aux, auxiliary_15_valid=aux is not None)
            else:
                if index != previous_sample:
                    with np.load(sample['path'], allow_pickle=False) as saved:
                        cache['state'] = saved['state'].copy()
                        cache['prediction'] = saved['actions'].copy()
                state, action = cache['state'], cache['prediction'][0]
                extra = dict(predicted_actions=cache['prediction'].tolist(), request_tick=sample['request_tick'],
                             publish_tick=sample['publish_tick'], accepted_plan=sample['accepted_plan'])
            previous_sample = index
            state, action = np.asarray(state, np.float32), np.asarray(action, np.float32)
            if state.shape != (14,) or action.shape != (14,):
                raise ValueError('Expected joint14 state and action')
            row = dict(index=i, episode_index=0, frame_index=i, task_index=0, timestamp=i/fps,
                       state=state.tolist(), actions=action.tolist(), source_sample_index=index,
                       source_timestamp=t, state_sample_timestamp=stamp, state_age_seconds=max(0., t-stamp), **extra)
            if images:
                image_index = max(0, bisect_right(image_times, t)-1)
                image_time, image_path = images[image_index]
                if image_index != previous_image:
                    with np.load(image_path, allow_pickle=False) as saved:
                        cache['images'] = {key: saved[key].copy() for key in IMAGES}
                previous_image = image_index
                row.update(camera_sample_timestamp=image_time, camera_age_seconds=max(0., t-image_time))
                for key, pixels in cache['images'].items():
                    if key not in videos:
                        path = stage/VIDEO_PATH.format(episode_chunk=0, episode_index=0, video_key=key)
                        videos[key] = _Video(path, pixels, fps)
                        h, w, _ = pixels.shape
                        features[key] = _feature('video', pixels.shape, ['height', 'width', 'channels'])
                        features[key]['info'] = {'video.height': h, 'video.width': w, 'video.codec': 'h264',
                            'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'video.fps': fps,
                            'video.channels': 3, 'has_audio': False}
                    videos[key].add(pixels)
            for key, acc in stats.items():
                acc.add(row[key])
            batch.append(row)
            if len(batch) >= 256:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        writer.close()
        writer = None
        for video in videos.values():
            video.close()
        numeric_stats = {k: v.result() for k, v in stats.items()}
        episode_stats = {**numeric_stats, **{k: v.stats() for k, v in videos.items()}}
        videos.clear()
        meta = stage/'meta'
        meta.mkdir()
        info = dict(codebase_version='v2.1', robot_type='piper', total_episodes=1, total_frames=count,
                    total_tasks=1, total_videos=len(IMAGES) if images else 0, total_chunks=1, chunks_size=1000,
                    fps=fps, splits={'train':'0:1'}, data_path=DATA_PATH, video_path=VIDEO_PATH if images else None,
                    features=features)
        provenance = dict(format='LeRobot v2.1', source_run=str(run), source_kind=kind, mode=mode,
                          episode_index=0, joint_unit='radians', gripper_unit='0..1', source_samples=len(samples),
                          source_image_groups=len(images), output_frames=count, fps=fps,
                          source_start_monotonic=start, source_end_monotonic=end,
                          camera_timestamp_semantics='snapshot read time; per-camera capture/device times remain in raw diagnostics',
                          resampling='causal zero-order hold; repeated rows/images are NOT new sensor measurements',
                          action_semantics='submitted_target' if kind == 'replay' else 'first_action_of_model_prediction_not_executed_target',
                          raw_records_retained=True)
        for name, value in [('info.json', info), ('stats.json', episode_stats), ('recording.json', provenance)]:
            (meta/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        for name, value in [('tasks.jsonl', dict(task_index=0, task=prompt)),
                            ('episodes.jsonl', dict(episode_index=0, tasks=[prompt], length=count)),
                            ('episodes_stats.jsonl', dict(episode_index=0, stats=episode_stats))]:
            (meta/name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        # The manifest is committed last: partial output is never advertised as
        # a valid dataset. Only this invocation's newly created paths are moved.
        for name in ('data', 'videos', 'meta'):
            if (stage/name).exists():
                if (output/name).exists():
                    raise FileExistsError(output/name)
                (stage/name).rename(output/name)
                committed.append(name)
        return dict(status='complete', directory=str(output), frames=count, episodes=1, fps=fps,
                    videos=len(IMAGES) if images else 0, codebase_version='v2.1',
                    action_semantics=provenance['action_semantics'])
    except BaseException:
        for name in committed:
            shutil.rmtree(output/name)
        raise
    finally:
        if writer is not None:
            writer.close()
        for video in videos.values():
            try:
                video.close()
            except Exception:
                pass
        shutil.rmtree(stage)
