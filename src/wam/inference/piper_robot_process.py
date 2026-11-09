"""Keep RealSense/CAN Python threads outside the GPU inference interpreter."""
import logging
import multiprocessing as mp
import signal
import threading
import traceback

LOG = logging.getLogger(__name__)


def _hardware_worker(connection, root, config, kwargs, factory):
    # Ctrl+C is handled by the parent, which stops the action clock then sends
    # close. Avoid racing a second driver cleanup in the hardware interpreter.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    robot = None
    try:
        if factory is None:
            from wam.inference.piper_robot import PhoenixPiperRobot
            factory = PhoenixPiperRobot
        robot = factory(root, config, **kwargs)
        connection.send({"ok": True, "value": robot.action_hz})
        while True:
            operation, payload = connection.recv()
            try:
                if operation == "observe":
                    result = robot.observe()
                elif operation == "enable":
                    result = robot.enable()
                elif operation == "send_action":
                    result = robot.send_action(payload)
                elif operation == "preview_action":
                    # Exercise the same IPC/cadence as execution, without
                    # changing driver targets or enabling motors.
                    import numpy as np
                    action = np.asarray(payload)
                    if action.shape != (14,) or not np.isfinite(action).all():
                        raise ValueError("Expected one finite preview joint14 target")
                    result = None
                elif operation == "close":
                    robot.close()
                    robot = None
                    connection.send({"ok": True, "value": None})
                    break
                else:
                    raise ValueError(f"Unknown hardware operation: {operation}")
                connection.send({"ok": True, "value": result})
            except Exception:
                connection.send({"ok": False, "error": traceback.format_exc()})
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    except BaseException:
        try:
            connection.send({"ok": False, "error": traceback.format_exc()})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if robot is not None:
            robot.close()
        connection.close()


class PiperRobotProcess:
    """Small synchronous RPC transport; the action clock stays in the parent.

    Only one request uses the pipe at a time. The hardware process itself never
    creates a model or CUDA context. Closing it uses the normal driver cleanup.
    """
    def __init__(self, root, config, *, prompt, execute_steps=10, _factory=None):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.lock = threading.Lock()
        self.closed = False
        self.process = context.Process(
            target=_hardware_worker,
            args=(child, root, config, {"prompt": prompt, "execute_steps": execute_steps}, _factory),
            name="phoenix-piper-hardware", daemon=True)
        self.process.start()
        child.close()
        try:
            self.action_hz = self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        message = self.connection.recv()
        if not message["ok"]:
            raise RuntimeError("Piper hardware process failed:\n" + message["error"])
        return message["value"]

    def _request(self, operation, payload=None):
        with self.lock:
            if self.closed:
                raise RuntimeError("Piper hardware process is closed")
            self.connection.send((operation, payload))
            return self._receive()

    def observe(self):
        return self._request("observe")

    def enable(self):
        return self._request("enable")

    def send_action(self, action):
        return self._request("send_action", action)

    def preview_action(self, action):
        return self._request("preview_action", action)

    def close(self):
        with self.lock:
            if self.closed:
                return
            try:
                if self.process.is_alive():
                    self.connection.send(("close", None))
                    if self.connection.poll(15):
                        self._receive()
                    else:
                        LOG.error("Piper hardware process did not confirm shutdown")
            except (EOFError, BrokenPipeError, OSError):
                # EOF also invokes the worker's finally/driver cleanup.
                pass
            finally:
                self.closed = True
                self.connection.close()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    LOG.error("Piper hardware process is still shutting down")
