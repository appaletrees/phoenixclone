"""Bounded recording of live RGB snapshots, independent of target scheduling."""
import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np


CAMERAS = {"cam_external": "image", "cam_left_wrist": "wrist_image_l",
           "cam_right_wrist": "wrist_image_r"}


def recording_settings(mode, image_hz):
    hz = float(image_hz)
    if mode not in ("actions", "full"):
        raise ValueError("记录模式需为 actions 或 full")
    if not math.isfinite(hz) or not 1 <= hz <= 30:
        raise ValueError("图像记录频率需在 1–30 Hz 内")
    return dict(mode=mode, image_hz=hz)


class ImageRecorder:
    """The control thread only publishes a small immutable state dictionary.

    A separate sampler copies cached camera frames, and a bounded writer queue
    compresses them. Timestamps describe the actual samples, not fake hardware
    synchronization. A slow disk drops image groups, never waits in update().
    """
    def __init__(self, directory, robot, hz=10., max_pending=4):
        self.hz = recording_settings("full", hz)["image_hz"]
        if max_pending < 1:
            raise ValueError("Image queue must be positive")
        self.directory = Path(directory) / "frames"
        self.directory.mkdir()
        self.robot = robot
        self.mapping = dict(getattr(robot, "config", {}).get("camera_mapping", CAMERAS))
        self.latest = None
        self.pending = queue.Queue(maxsize=max_pending)
        self.stop = threading.Event()
        self.closed = False
        self.error = None
        self.attempted = self.submitted = self.written = self.dropped = self.skipped_ticks = 0
        self.missing_groups = 0
        self.complete_groups = self.images_written = 0
        self.cancelled_on_stop = self.capture_failures = 0
        self.last_missing = []
        self.writer = threading.Thread(target=self._write_loop, name="replay-image-writer", daemon=True)
        self.sampler = threading.Thread(target=self._sample_loop, name="replay-image-sampler", daemon=True)
        self.writer.start()
        self.sampler.start()

    def update(self, sample):
        self.latest = sample  # sample is immutable after publication

    def check(self):
        if self.error:
            raise RuntimeError("图像记录失败: " + self.error)

    def _sample_loop(self):
        due = time.monotonic()
        while not self.stop.is_set():
            if self.stop.wait(max(0., due - time.monotonic())):
                break
            sample = self.latest
            if sample is not None and self.error is None:
                index = self.attempted
                self.attempted += 1
                if self.pending.full():
                    self.dropped += 1
                else:
                    try:
                        item = self._capture(index, sample)
                        if not self.stop.is_set():
                            self.pending.put_nowait(item)
                            self.submitted += 1
                            self.last_missing = item[2]['missing_cameras']
                            self.missing_groups += int(bool(self.last_missing))
                        else:
                            self.cancelled_on_stop += 1
                    except Exception as exc:
                        self.capture_failures += 1
                        self.error = f"{type(exc).__name__}: {exc}"
            due += 1/self.hz
            now = time.monotonic()
            if due < now:
                missed = int((now-due)*self.hz) + 1
                self.skipped_ticks += missed
                due += missed/self.hz

    def _capture(self, index, sample):
        started = time.monotonic()
        frames = self.robot.capture_frames()  # cached copies only; no camera wait
        mono, wall = time.monotonic(), time.time()
        arrays = {"state": np.asarray(sample["current"], dtype=np.float64)}
        for key, source in (("action", "target"), ("driver_action", "driver_target")):
            if sample[source] is not None:
                arrays[key] = np.asarray(sample[source], dtype=np.float64)
        metadata = dict(sample, image_index=index, capture_started_monotonic=started,
                        capture_monotonic=mono, capture_wall_time=wall,
                        state_age_seconds=mono-sample["state_read_monotonic"],
                        color_order="RGB", cameras={}, missing_cameras=[])
        for name, key in self.mapping.items():
            frame = frames.get(name, {})
            pixels = frame.get("rgb")
            camera = {field: frame.get(field) for field in
                      ("timestamp", "frame_id", "device_timestamp_ms", "device_frame_number")}
            camera["array_key"] = key
            camera["age_seconds"] = None if camera["timestamp"] is None else mono-float(camera["timestamp"])
            camera["present"] = pixels is not None
            metadata["cameras"][name] = camera
            if pixels is None:
                metadata["missing_cameras"].append(name)
            else:
                if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
                    raise ValueError(f"Invalid BGR camera frame: {name}")
                # The camera manager calls the field rgb but returns BGR.
                arrays[key] = pixels[..., ::-1].copy()
        return index, arrays, metadata

    def _write(self, index, arrays, metadata):
        stem = self.directory / f"frame_{index:08d}"
        # JSON appears last and denotes a completed image group.
        temporary = stem.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(stem.with_suffix(".npz"))
        temporary = stem.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
        temporary.replace(stem.with_suffix(".json"))

    def _write_loop(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    try:
                        self._write(*item)
                        self.written += 1
                        self.images_written += sum(key in item[1] for key in self.mapping.values())
                        self.complete_groups += int(not item[2]['missing_cameras'])
                    except Exception as exc:
                        self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.pending.task_done()

    def stats(self):
        return dict(image_hz=self.hz, attempted=self.attempted, submitted=self.submitted,
                    written=self.written, dropped=self.dropped, skipped_ticks=self.skipped_ticks,
                    missing_camera_groups=self.missing_groups, last_missing_cameras=self.last_missing,
                    complete_groups=self.complete_groups, images_written=self.images_written,
                    cancelled_on_stop=self.cancelled_on_stop, capture_failures=self.capture_failures,
                    queue_capacity=self.pending.maxsize, error=self.error,
                    unfinished=self.submitted-self.written, closed=self.closed)

    def close(self):
        if self.closed:
            return
        self.stop.set()
        self.sampler.join(timeout=5)
        if self.sampler.is_alive():
            self.error = self.error or "Camera recorder did not stop within 5 seconds"
        self.pending.put(None)
        self.writer.join()
        self.closed = True
