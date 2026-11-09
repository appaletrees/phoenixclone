"""Joint-only return trajectory; no model, camera, SDK or process imports."""
import math
import threading
import time

import numpy as np


def validate_return_options(target, duration):
    if target not in ("initial", "zero", "none"):
        raise ValueError("return-on-interrupt must be initial, zero or none")
    if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("return-duration must be finite and positive")


def _joints(value):
    value = np.array(value, dtype=np.float64, copy=True)
    if value.shape != (12,) or not np.isfinite(value).all():
        raise ValueError("Return requires twelve finite joint angles in degrees")
    return value


def return_joints(read_joints, send_joints, target, *, duration=5., hz=30.,
                  cancel=None, settle_timeout=5., tolerance_deg=2.):
    """Quintic time interpolation followed by measured arrival confirmation.

    Only joint targets are written: the driver's existing gripper targets are
    retained. The caller must first stop the policy clock and must close the
    transport after completion, cancellation or failure.
    """
    validate_return_options("initial", duration)
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError("Return frequency must be finite and positive")
    cancel = threading.Event() if cancel is None else cancel
    target = _joints(target)
    if cancel.is_set():
        return {"status": "aborted", "sent_targets": 0}
    start = _joints(read_joints())
    started = time.monotonic()
    sent = 0
    within_since = None
    feedback = start
    while not cancel.is_set():
        elapsed = time.monotonic() - started
        feedback = _joints(read_joints())
        u = min(1., elapsed / duration)
        blend = u**3 * (10. - 15. * u + 6. * u**2)
        command = start + (target - start) * blend
        if cancel.is_set():
            break
        send_joints(command)
        sent += 1
        error = float(np.max(np.abs(feedback - target)))
        if u == 1.:
            if error <= tolerance_deg:
                within_since = time.monotonic() if within_since is None else within_since
                if time.monotonic() - within_since >= .2:
                    return dict(status="completed", sent_targets=sent,
                                elapsed_seconds=time.monotonic() - started,
                                max_error_deg=error, target_joints_deg=target.tolist(),
                                final_joints_deg=feedback.tolist())
            else:
                within_since = None
            if elapsed >= duration + settle_timeout:
                raise TimeoutError(f"Return pose not reached: maximum joint error {error:.3f} degrees")
        cancel.wait(1. / hz)
    return dict(status="aborted", sent_targets=sent, elapsed_seconds=time.monotonic() - started,
                target_joints_deg=target.tolist(), final_joints_deg=feedback.tolist())
