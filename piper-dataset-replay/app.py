#!/usr/bin/env python3
"""Independent local web UI for original Piper dataset playback."""
import argparse
import json
from pathlib import Path
import secrets
import signal
import sys
import time

sys.dont_write_bytecode = True

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.serving import make_server

from catalog import Catalog
from engine import ReplayEngine
from hardware import DemoRobot, PiperRobot
from video import Frames
from paths import default_project_root, default_interface_root

HERE = Path(__file__).resolve().parent


def create_app(engine, frames):
    app = Flask(__name__, static_folder=str(HERE / "web"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 16384
    token = secrets.token_urlsafe(32)
    app.config["REPLAY_TOKEN"] = token

    @app.before_request
    def validate_command_origin():
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
                return jsonify(error="操作来源不匹配"), 403
            if not secrets.compare_digest(request.headers.get("X-Replay-Token", ""), token):
                return jsonify(error="请刷新本机页面后重试"), 403

    @app.get("/")
    def index():
        return Response((HERE / "web/index.html").read_text().replace("__REPLAY_TOKEN__", token), mimetype="text/html")

    @app.get("/api/catalog")
    def catalog():
        return jsonify(datasets=engine.catalog.listing(), mode=engine.robot.kind)

    @app.get("/api/status")
    def status():
        try:
            after = int(request.args.get("after", "-1"))
        except ValueError:
            return jsonify(error="Invalid sequence"), 400
        data = engine.snapshot(after)
        data["video_errors"] = dict(frames.errors)
        return jsonify(data)

    @app.post("/api/command/<name>")
    def command(name):
        try:
            values = request.get_json(force=False) or {}
            if not isinstance(values, dict):
                raise ValueError("需要 JSON 对象")
            normalized = engine.submit(name, values)
            return jsonify(accepted=True, normalized=normalized)
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/stream/<mode>/<int:camera>")
    def stream(mode, camera):
        if mode not in ("live", "recorded") or camera not in (0, 1, 2):
            return jsonify(error="未知相机"), 404
        def generate():
            while not frames.stop_event.is_set():
                image = frames.get(mode, camera)
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(image)).encode() + b"\r\n\r\n" + image + b"\r\n"
                time.sleep(.1)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/api/recording.csv")
    def download_recording():
        with engine.lock:
            path = engine.log_path
            if engine.log_handle and not engine.log_handle.closed:
                engine.log_handle.flush()
        if path is None or not path.is_file():
            return jsonify(error="开始回放后会生成关节记录"), 404
        return send_file(path, as_attachment=True, download_name=path.parent.name + "-joints.csv", max_age=0)

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    parser.add_argument("--interface-root", type=Path,
                        help="External controller directory; defaults to Desktop/piper_tele beside the project")
    parser.add_argument("--robot-config", type=Path, default=Path("configs/inference/piper_robot.json"),
                        help="Relative to project root")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Relative to this standalone tool")
    parser.add_argument("--record", choices=("actions", "full"), default="actions",
                        help="actions: state/actions/diagnostics; full: also three live RGB cameras")
    parser.add_argument("--record-image-hz", type=float, default=10., help="Live image groups per second, 1..30")
    parser.add_argument("--record-fps", type=float, default=30., help="LeRobot v2.1 uniform timeline FPS (1..100)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--hardware", action="store_true", help="Allow hardware connection from the page; does not auto-enable")
    modes.add_argument("--demo", action="store_true", help="Offline UI demo with explicitly simulated feedback (default)")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    project = args.project_root.resolve()
    catalog = Catalog(project)
    interface_root = (args.interface_root or default_interface_root(project)).resolve()
    robot = (PiperRobot(project, interface_root,
                        json.loads((project / args.robot_config).read_text())) if args.hardware else DemoRobot(project))
    engine = ReplayEngine(catalog, robot, HERE / args.output_dir.expanduser(),
                          record=args.record, image_hz=args.record_image_hz, record_fps=args.record_fps)
    frames = Frames(engine)
    app = create_app(engine, frames)
    server = make_server("127.0.0.1", args.port, app, threaded=True)
    def shutdown_signal(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown_signal)
    try:
        print(f"PIPER_REPLAY_READY http://127.0.0.1:{args.port} mode={robot.kind}", flush=True)
        print("Hardware is connected/enabled only by explicit page controls. Ctrl+C stops and disconnects without homing.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.cancel.set()
        frames.close()
        engine.close()
        server.server_close()


if __name__ == "__main__":
    main()
