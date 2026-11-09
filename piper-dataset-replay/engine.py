"""One owner thread for commands and playback; web/image work never sets targets."""
from collections import deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import threading
import time
import tempfile
from uuid import uuid4

import numpy as np

from catalog import public_episode
from hardware import Cancelled
from recording import ImageRecorder, recording_settings


def valid_rate(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 100:
        raise ValueError("回放目标更新频率需大于 0 且不超过 100 Hz")
    return value


class ReplayEngine:
    def __init__(self, catalog, robot, output, *, record="actions", image_hz=10., record_fps=30.):
        self.catalog, self.robot, self.output = catalog, robot, Path(output).expanduser().resolve()
        self.record_fps = valid_rate(record_fps)
        if self.record_fps < 1:
            raise ValueError('LeRobot FPS 必须至少为 1')
        self.lock = threading.RLock()
        self.commands = queue.Queue()
        self.cancel, self.closing = threading.Event(), threading.Event()
        self.phase = "idle"
        self.busy = False
        self.error = ""
        self.message = "选择原始轨迹；连接设备后开始回放"
        self.episode = None
        self.frame = self.next_frame = 0
        self.hz, self.align_seconds = 30., 3.
        self.current = self.target = self.applied = None
        self._due = 0.
        self._align_from = self._align_to = None
        self._align_started = 0.
        self._started = time.monotonic()
        self._sequence = self._stop_version = 0
        self.samples = deque(maxlen=36000)
        self.history_epoch = 0
        self._paused_was_playing = False
        self.operation = "replay"
        self.command_times = deque(maxlen=100)
        self.sent_frames = self.late_ticks = self.clamped_ticks = 0
        self.log_path = self.log_handle = self.log_writer = None
        self.record_settings = recording_settings(record, image_hz)
        self.feedback_log = self.action_log = self.image_recorder = None
        self.last_recording = None
        self.lerobot_export = None
        self._recording_finalizer = None
        self.recorded_actions = self.recorded_states = 0
        self.last_run = None
        self._last_flush = 0.
        self.thread = threading.Thread(target=self._loop, name="piper-replay-control", daemon=True)
        self.thread.start()

    def submit(self, name, values=None):
        values = dict(values or {})
        allowed = {"select", "connect", "disconnect", "start", "goto_frame", "pause", "stop", "disable", "seek", "rate", "control", "recording"}
        if name not in allowed:
            raise ValueError("未知操作")
        with self.lock:
            if name in ("pause", "stop", "disable", "disconnect"):
                self._stop_version += 1
                self.cancel.set()
            elif self.busy:
                raise ValueError("上一个操作尚未完成")
            if name == "control":
                if self.robot.connected or getattr(self.robot, "controllers", []):
                    raise ValueError("请先断开设备，再修改控制模式或 MIT 参数")
                self.robot.validate_control(values)
            if name == "recording":
                if self.log_handle is not None or self._saving_images():
                    raise ValueError("请先停止本次回放，再修改记录配置（暂停仍在记录）")
                output = self._recording_output(values.get('output_dir', str(self.output)))
                values = dict(recording_settings(values.get("mode"), values.get("image_hz", 10)), output_dir=str(output))
            if name in ("select", "seek") and self.phase in ("playing", "aligning", "moving"):
                raise ValueError("请先暂停或停止回放，再切换轨迹或帧")
            if name in ("start", "goto_frame"):
                self._check_recording_start(name)
                if self.phase in ("playing", "aligning", "moving"):
                    raise ValueError("正在回放")
                if not self.robot.connected or self.episode is None:
                    raise ValueError("请先选择轨迹并连接设备")
                values["hz"] = valid_rate(values.get("hz", self.hz))
                alignment = float(values.get("align_seconds", self.align_seconds))
                if not math.isfinite(alignment) or alignment < 0 or alignment > 60:
                    raise ValueError("起点对齐时长需在 0–60 秒内")
                values["align_seconds"] = alignment
                self.cancel.clear()
            if name == "rate":
                values["hz"] = valid_rate(values.get("hz"))
            if name in ("seek", "goto_frame"):
                frame = values.get("frame")
                if self.episode is None or type(frame) is not int or not 0 <= frame < self.episode["frames"]:
                    raise ValueError("起始帧超出轨迹范围")
            if name in ("select", "connect", "disconnect", "start", "goto_frame", "seek", "control", "recording"):
                self.busy = True
            self.commands.put((name, values, self._stop_version))
            return values

    def _new_recording(self):
        self.output.mkdir(parents=True, exist_ok=True)
        run = self.output / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
        run.mkdir()
        self.log_path = run / "joints.csv"
        self.log_handle = self.log_path.open("w", newline="")
        self.log_writer = csv.writer(self.log_handle)
        self.feedback_log = (run / "feedback.jsonl").open("w")
        self.action_log = (run / "commands.jsonl").open("w")
        self.recorded_actions = self.recorded_states = 0
        self.last_recording = None
        self.lerobot_export = {'status':'recording', 'codebase_version':'v2.1', 'directory':str(run.resolve())}
        names = [f"{arm}_j{i}" for arm in ("left", "right") for i in range(1, 8)]
        names[6], names[13] = "left_gripper", "right_gripper"
        self.log_writer.writerow(["sequence", "elapsed_s", "source_frame", "source_time_s", "phase", "requested_hz"] +
                                 [f"{group}_{n}" for group in ("target", "feedback", "driver_target") for n in names])
        self.last_run = dict(run_id=run.name, hardware=self.robot.kind == "hardware", episode=public_episode(self.episode),
                             source_parquet=self.episode["parquet"], joint_unit="radians", gripper_unit="0..1",
                             initial_hz=self.hz, alignment_seconds=self.align_seconds,
                             operation=self.operation, start_frame=self.next_frame,
                             auxiliary_15="preserved in source; never sent to robot",
                             controller_hz=self.robot.control.controller_hz,
                             robot_control=self.robot.control.audit(),
                             feedback_sampling_hz=30, smoothing_during_replay=False,
                             schema_version=2, output_directory=str(run.resolve()),
                             cameras=getattr(self.robot, 'config', {}).get('cameras', {}),
                             recording=dict(self.record_settings, image_queue_capacity=4,
                                            format='lerobot_v2.1', dataset_fps=self.record_fps,
                                            image_color_order="RGB", image_source="live_cameras",
                                            state_action_order="left_j1..j6,left_gripper,right_j1..j6,right_gripper",
                                            alignment="timestamps; images and feedback are not hardware synchronized"))
        (run / "run.json").write_text(json.dumps(self.last_run, ensure_ascii=False, indent=2))
        if self.record_settings["mode"] == "full":
            self.image_recorder = ImageRecorder(run, self.robot, self.record_settings["image_hz"])

    def _recording_status(self):
        return dict(active=self.log_handle is not None, settings=dict(self.record_settings, output_dir=str(self.output)),
                    format='lerobot_v2.1', dataset_fps=self.record_fps, lerobot=self.lerobot_export,
                    saving=self._saving_images(),
                    recorded_actions=self.recorded_actions, recorded_states=self.recorded_states,
                    images=(self.image_recorder or self.last_recording).stats() if (self.image_recorder or self.last_recording) else None,
                    directory=None if self.log_path is None else str(self.log_path.parent.resolve()))

    def _saving_images(self):
        return self._recording_finalizer is not None and self._recording_finalizer.is_alive()

    @staticmethod
    def _recording_output(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('保存目录不能为空')
        path = Path(value.strip()).expanduser()
        path = (Path(__file__).resolve().parent/path).resolve()
        if path.exists() and not path.is_dir():
            raise ValueError('保存路径必须是目录')
        return path

    def _check_recording_start(self, name):
        if self._saving_images():
            raise ValueError("上一轮图像仍在后台保存，保存完成后再开始")
        resumed = name == 'start' and self.phase == 'paused' and self._paused_was_playing
        if self.image_recorder is not None and not resumed:
            raise ValueError("请先停止当前记录，保存完成后再开始新的回放或单帧移动")

    def _record_action(self, kind, source_frame):
        if self.action_log is not None:
            mono = time.monotonic()
            event = dict(sequence=self.recorded_actions, kind=kind, phase=self.phase,
                         monotonic=mono, wall_time=time.time(), elapsed_s=mono-self._started,
                         source_frame=source_frame, source_time_s=float(self.episode["timestamps"][source_frame]),
                         requested_hz=self.hz, target=None if self.target is None else self.target.tolist(),
                         driver_target=None if self.applied is None else self.applied.tolist())
            self.action_log.write(json.dumps(event, allow_nan=False) + "\n")
            self.recorded_actions += 1

    def _finish_recording(self):
        with self.lock:
            if self.log_handle is None:
                return
            recorder = self.image_recorder
            if recorder is not None:
                recorder.stop.set()
                self.last_recording = recorder
                self.image_recorder = None
            self.log_handle.flush()
            self.log_handle.close()
            self.log_handle = self.log_writer = None
            for stream in (self.feedback_log, self.action_log):
                if stream is not None:
                    stream.close()
            self.feedback_log = self.action_log = None
            if self.last_run and self.log_path:
                path = self.log_path.parent / "summary.json"
                summary = dict(
                    phase=self.phase, sent_frames=self.sent_frames, late_ticks=self.late_ticks,
                    driver_clamped_ticks=self.clamped_ticks, error=self.error,
                    elapsed_seconds=time.monotonic()-self._started,
                    recording=self._recording_status(), finalized=False)
                summary['recording']['saving'] = True
                self.lerobot_export = {'status':'exporting', 'directory':str(path.parent), 'codebase_version':'v2.1'}
                summary['recording']['lerobot'] = self.lerobot_export
                path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                # Draining the disk queue must not hold engine.lock or delay
                # stop/disable/disconnect. Block only a new recording until
                # this bounded queue has drained.
                def finalize():
                    try:
                        if recorder is not None:
                            recorder.close()
                            summary['recording']['images'] = recorder.stats()
                            if recorder.error:
                                raise RuntimeError(recorder.error)
                        from wam.inference.piper_lerobot_export import export_run
                        self.lerobot_export = export_run(path.parent, kind='replay',
                            mode=summary['recording']['settings']['mode'], fps=self.record_fps)
                    except Exception as exc:
                        self.lerobot_export = dict(status='error', directory=str(path.parent), error=f'{type(exc).__name__}: {exc}')
                        summary['error'] = summary['error'] or self.lerobot_export['error']
                    finally:
                        summary['recording']['lerobot'] = self.lerobot_export
                        summary['recording']['saving'] = False
                        summary['finalized'] = True
                        try:
                            temporary = path.with_suffix('.json.tmp')
                            temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                            temporary.replace(path)
                        except Exception as exc:
                            self.lerobot_export = dict(status='error', error=f'Summary write failed: {exc}')
                self._recording_finalizer = threading.Thread(target=finalize, name='replay-recording-finalizer', daemon=True)
                self._recording_finalizer.start()

    def _command(self, name, values, version):
        if name == "recording":
            if self.log_handle is not None or self._saving_images():
                raise ValueError("请先停止本次回放，再修改记录配置")
            self.record_settings = recording_settings(values["mode"], values["image_hz"])
            self.output = self._recording_output(values['output_dir'])
            self.message = "记录配置已应用，下次回放或移动到指定帧时生效"
        elif name == "control":
            self.robot.configure_control(values)
            self.error = ""
            mode = "MIT 阻抗" if self.robot.control.mode == "mit" else "Joint 位置"
            self.message = f"已选 {mode}控制；连接并开始回放后生效"
        elif name == "connect":
            self.message = "正在连接设备和相机…"
            self.robot.connect()
            with self.lock:
                self.current = self.robot.state()
                self.message = "设备已连接，尚未发送回放动作"
        elif name == "select":
            episode = self.catalog.load(values.get("dataset"), values.get("episode"), values.get("field", "action"))
            with self.lock:
                self._finish_recording()
                self.episode, self.frame, self.next_frame = episode, 0, 0
                self.phase = "idle"
                self.target = self.applied = None
                self.samples.clear()
                self.history_epoch += 1
                self.sent_frames = self.late_ticks = self.clamped_ticks = 0
                self.command_times.clear()
                self._started = time.monotonic()
                self.message = "轨迹已加载；选择数据不会移动机械臂"
        elif name == "seek":
            with self.lock:
                self._finish_recording()
                self.frame = self.next_frame = int(values["frame"])
                self.phase = "idle"
                self.target = self.applied = None
                self.samples.clear()
                self.history_epoch += 1
                self.sent_frames = self.late_ticks = self.clamped_ticks = 0
                self.command_times.clear()
                self._started = time.monotonic()
                self.message = "起始帧已更新；点击开始后执行起点对齐"
        elif name == "rate":
            self.hz = values["hz"]
            self.command_times.clear()
            self._due = time.monotonic() + 1/self.hz
        elif name in ("start", "goto_frame"):
            self._check_recording_start(name)
            if version != self._stop_version or self.cancel.is_set():
                raise Cancelled("启动已取消")
            single_frame = name == "goto_frame"
            resumed = not single_frame and self.phase == "paused" and self._paused_was_playing
            if not resumed:
                try:
                    self.output.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryFile(dir=self.output):
                        pass
                except OSError as exc:
                    raise ValueError(f'保存目录不可写：{self.output}: {exc}') from exc
            self.hz, self.align_seconds = values["hz"], values["align_seconds"]
            self.message = "正在同步当前姿态并使能…"
            self.robot.enable(self.cancel)
            if self.cancel.is_set():
                raise Cancelled("启动已取消")
            self.current = self.robot.state()
            if self.current is None:
                raise RuntimeError("没有关节反馈，无法开始")
            if not resumed:
                self._finish_recording()
                self.operation = "single_frame" if single_frame else "replay"
                if single_frame:
                    self.frame = self.next_frame = int(values["frame"])
                elif self.next_frame >= self.episode["frames"]:
                    self.frame = self.next_frame = 0
                self._started = time.monotonic()
                self.samples.clear()
                self.history_epoch += 1
                self.sent_frames = self.late_ticks = self.clamped_ticks = 0
                self.command_times.clear()
                with self.lock:
                    self._new_recording()
            self.command_times.clear()
            self.error = ""
            if not resumed and self.align_seconds > 0:
                self.phase = "moving" if single_frame else "aligning"
                self._align_started = time.monotonic()
                self._align_from = self.current.copy()
                self._align_to = self.episode["target"][self.next_frame].copy()
            elif single_frame:
                if self._send(self.episode["target"][self.frame]):
                    self.sent_frames = 1
                self.phase = "holding"
            else:
                self.phase = "playing"
            self._due = time.monotonic()
            self.message = (f"移动到第 {self.frame} 帧，之后保持该目标" if self.phase == "moving" else
                            f"保持第 {self.frame} 帧目标，不继续播放；蓝线为固定目标" if self.phase == "holding" else
                            "起点对齐中" if self.phase == "aligning" else "按原始帧顺序回放")
        elif name in ("pause", "stop"):
            held = self.robot.hold() if self.robot.connected else None
            self._paused_was_playing = self.phase == "playing"
            self.target = self.applied = held
            self._record_action("hold", self.frame)
            self.phase = "paused" if name == "pause" and self.episode is not None else "idle"
            self.message = "已暂停，保持当前姿态" if self.phase == "paused" else "已停止回放，保持当前姿态"
            if name == "stop":
                self._finish_recording()
        elif name in ("disable", "disconnect"):
            self.robot.disable()
            self.phase = "disabled"
            self.message = "已停止并失能；没有自动回零"
            self.target = self.applied = None
            self._finish_recording()
            if name == "disconnect":
                self.robot.close()
                self.current = None
                self.message = "设备已断开"

    def _send(self, value, source_frame=None):
        if self.cancel.is_set():
            return False
        applied = self.robot.send(value)
        self.target = np.asarray(value).copy()
        self.applied = np.asarray(applied).copy()
        self._record_action("target", self.frame if source_frame is None else source_frame)
        if not np.allclose(self.target, self.applied, atol=1e-6):
            self.clamped_ticks += 1
        return True

    def _tick(self, now):
        if self.cancel.is_set() or now < self._due:
            return
        if self.phase in ("aligning", "moving"):
            amount = min(1., (now-self._align_started)/self.align_seconds)
            # Explicit configurable entry transition; recorded samples remain untouched.
            value = self._align_to if amount >= 1 else self._align_from + amount*(self._align_to-self._align_from)
            sent = self._send(value, self.next_frame)
            self._due = now + .01
            if sent and amount >= 1:
                if self.phase == "moving":
                    self.phase = "holding"
                    self.sent_frames = 1
                    self.message = f"保持第 {self.frame} 帧目标，不继续播放；蓝线为固定目标"
                else:
                    self.phase = "playing"
                    self.message = "按原始帧顺序回放"
                self._due = now
        elif self.phase == "playing":
            if self.next_frame >= self.episode["frames"]:
                self.phase = "completed"
                self.message = "回放完成，保持末帧目标；可停止并失能"
                self._finish_recording()
                return
            if now-self._due > max(.005, .5/self.hz):
                self.late_ticks += 1
            index = self.next_frame
            if self._send(self.episode["target"][index], index):
                self.frame = index
                self.next_frame = index+1
                self.sent_frames += 1
                self.command_times.append(now)
            # Never drop frames or burst through overdue frames to catch up.
            self._due = now + 1/self.hz

    def _sample(self, now):
        if self.image_recorder is not None:
            self.image_recorder.check()
        current = self.robot.state()
        self.current = current
        if current is None:
            return
        self._sequence += 1
        source_time = float(self.episode["timestamps"][self.frame]) if self.episode is not None else 0.
        values = dict(seq=self._sequence, t=now-self._started, frame=self.frame, source_s=source_time,
                      phase=self.phase, target=None if self.target is None else self.target.tolist(), current=current.tolist())
        self.samples.append(values)
        if self.log_writer:
            sample = dict(values, state_read_monotonic=time.monotonic(), wall_time=time.time(),
                          requested_hz=self.hz, action_sequence=self.recorded_actions-1,
                          driver_target=None if self.applied is None else self.applied.tolist(),
                          feedback=getattr(self.robot, "last_feedback", {}),
                          auxiliary_15=None if self.episode["auxiliary"] is None else float(self.episode["auxiliary"][self.frame]))
            self.feedback_log.write(json.dumps(sample, allow_nan=False) + "\n")
            self.recorded_states += 1
            if self.image_recorder is not None:
                self.image_recorder.update(sample)
            blank = [""]*14
            self.log_writer.writerow([self._sequence, values["t"], self.frame, source_time, self.phase, self.hz] +
                                     (blank if self.target is None else self.target.tolist()) + current.tolist() +
                                     (blank if self.applied is None else self.applied.tolist()))
            if now-self._last_flush > 1:
                self.log_handle.flush()
                self.feedback_log.flush()
                self.action_log.flush()
                self._last_flush = now

    def _loop(self):
        sampled = 0.
        while not self.closing.is_set():
            command = None
            try:
                sample_due = sampled + 1/30 if self.robot.connected and self.phase != "error" else time.monotonic()+.02
                next_due = min(sample_due, self._due) if self.phase in ("playing", "aligning", "moving") else sample_due
                command = self.commands.get(timeout=max(.0002, min(.02, next_due-time.monotonic())))
            except queue.Empty:
                pass
            try:
                if command:
                    # Slow device connection/enable runs outside the snapshot mutex.
                    self._command(*command)
                    with self.lock:
                        self.busy = False
                        self.error = ""
                now = time.monotonic()
                with self.lock:
                    self._tick(now)
                    if self.robot.connected and self.phase != "error" and now-sampled >= 1/30:
                        self._sample(now)
                        sampled = now
            except (ValueError, KeyError, FileNotFoundError) as exc:
                with self.lock:
                    self.error = str(exc)
                    self.busy = False
            except Cancelled:
                with self.lock:
                    self.busy = False
                    self.phase = "idle"
                    self.message = "启动已取消"
            except Exception as exc:
                self.cancel.set()
                try:
                    self.robot.disable()
                except Exception as stop_error:
                    exc = RuntimeError(f"{exc}; {stop_error}")
                with self.lock:
                    self.error = str(exc)
                    self.phase = "error"
                    self.busy = False
                    self.message = "操作失败，已停止回放"
                    self._finish_recording()
                # Avoid hammering a disconnected device at 500 Hz.
                time.sleep(.1)
        try:
            self._finish_recording()
        finally:
            self._close_robot()
            if self._recording_finalizer is not None:
                self._recording_finalizer.join()

    def _close_robot(self):
        try:
            self.robot.close()
        except Exception as exc:
            self.error = str(exc)
            print("HARDWARE_CLOSE_ERROR", exc, flush=True)

    def snapshot(self, after=-1):
        with self.lock:
            ticks = self.command_times
            measured = (len(ticks)-1)/(ticks[-1]-ticks[0]) if len(ticks)>1 and ticks[-1]>ticks[0] else 0.
            history = [s for s in self.samples if s["seq"] > after][-3600:]
            aux = None if self.episode is None or self.episode["auxiliary"] is None else float(self.episode["auxiliary"][self.frame])
            return dict(mode=self.robot.kind, phase=self.phase, busy=self.busy, connected=self.robot.connected,
                        control_settings=self.robot.control_settings, robot_control=self.robot.control.audit(),
                        enabled=self.robot.enabled, message=self.message, error=self.error,
                        episode=public_episode(self.episode), frame=self.frame, hz=self.hz,
                        actual_hz=measured if self.phase == "playing" else 0., align_seconds=self.align_seconds,
                        current=None if self.current is None else self.current.tolist(),
                        target=None if self.target is None else self.target.tolist(), auxiliary=aux,
                        sent_frames=self.sent_frames, late_ticks=self.late_ticks, clamped_ticks=self.clamped_ticks,
                        camera_status=self.robot.camera_status, last_run=self.last_run,
                        recording=self._recording_status(),
                        operation=self.operation,
                        samples=history, latest_sequence=self._sequence, history_epoch=self.history_epoch)

    def video_position(self):
        with self.lock:
            return self.episode, self.frame

    def close(self):
        self.cancel.set()
        self.closing.set()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            raise RuntimeError("硬件关闭尚未完成，请检查终端")
