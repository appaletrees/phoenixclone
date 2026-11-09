"""Three original recordings and three live views; independent of action timing."""
import threading
import time

import av
import cv2
import numpy as np


class RecordingReader:
    def __init__(self, path, fps):
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = 1
        self.fps = fps
        self.iterator = iter(self.container.decode(video=0))
        self.index = -1
        self.image = None

    def get(self, index):
        if index == self.index:
            return self.image
        if index < self.index or index-self.index > 120:
            self.container.seek(int((index/self.fps)/self.stream.time_base), stream=self.stream, backward=True)
            self.iterator = iter(self.container.decode(video=0))
        for frame in self.iterator:
            frame_index = int(round(float(frame.time)*self.fps))
            if frame_index >= index:
                self.index, self.image = frame_index, frame.to_ndarray(format="bgr24")
                return self.image
        return self.image

    def close(self):
        self.container.close()


class Frames:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frames = {}
        self.errors = {}
        self.thread = threading.Thread(target=self._loop, name="piper-replay-video", daemon=True)
        self.thread.start()

    def _store(self, key, image):
        image = cv2.resize(image, (480, 360))
        success, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if success:
            with self.lock:
                self.frames[key] = jpeg.tobytes()

    def _clear(self, mode, camera=None):
        with self.lock:
            for key in list(self.frames):
                if key[0] == mode and (camera is None or key[1] == camera):
                    del self.frames[key]

    def get(self, mode, camera):
        with self.lock:
            value = self.frames.get((mode, camera))
        if value is not None:
            return value
        image = np.full((360, 480, 3), (32, 39, 48), np.uint8)
        text = "Connect live cameras" if mode == "live" else "Select a recorded episode"
        cv2.putText(image, text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, .62, (175, 190, 205), 1, cv2.LINE_AA)
        return cv2.imencode(".jpg", image)[1].tobytes()

    def _loop(self):
        episode = None
        readers = []
        last_frame = -1
        while not self.stop_event.is_set():
            started = time.monotonic()
            chosen, index = self.engine.video_position()
            try:
                if chosen is not episode:
                    self._clear("recorded")
                    for reader in readers:
                        reader.close()
                    readers = []
                    episode, last_frame = chosen, -1
                    if chosen is not None:
                        for path in chosen["videos"]:
                            readers.append(RecordingReader(path, chosen["fps"]))
                if readers and index != last_frame:
                    for camera, reader in enumerate(readers):
                        image = reader.get(index)
                        if image is not None:
                            self._store(("recorded", camera), image)
                    last_frame = index
                    self.errors.pop("recorded", None)
            except Exception as exc:
                self._clear("recorded")
                self.errors["recorded"] = str(exc)
            try:
                if self.engine.robot.connected:
                    frames = self.engine.robot.capture_frames()
                    for camera, name in enumerate(("cam_external", "cam_left_wrist", "cam_right_wrist")):
                        value = frames.get(name, {}).get("rgb")
                        if value is not None:
                            self._store(("live", camera), value)
                        else:
                            self._clear("live", camera)
                    self.errors.pop("live", None)
                else:
                    self._clear("live")
                    self.errors.pop("live", None)
            except Exception as exc:
                self._clear("live")
                self.errors["live"] = str(exc)
            self.stop_event.wait(max(.005, .1-(time.monotonic()-started)))
        for reader in readers:
            reader.close()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
