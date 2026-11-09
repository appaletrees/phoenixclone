"""Read-only resolution of Piper CAN interfaces by physical USB identity."""
from pathlib import Path
import re

from wam.inference.piper_feedback import feedback_id_offsets


def discover_can_interfaces(sysfs_root=Path("/sys/class/net")):
    """No SDK import, socket connection, network mutation or robot command."""
    interfaces = {}
    for path in Path(sysfs_root).iterdir():
        try:
            if (path / "type").read_text().strip() != "280":
                continue
            device = (path / "device").resolve()
            serial = next(((parent / "serial").read_text().strip()
                           for parent in (device, *device.parents)
                           if (parent / "serial").is_file()), None)
            interfaces[path.name] = {
                "serial": serial,
                # CAN operstate may be UNKNOWN even when the interface is UP.
                "up": bool(int((path / "flags").read_text().strip(), 16) & 1),
            }
        except FileNotFoundError:
            # A USB device can disappear between enumeration and reading.
            continue
    return interfaces


def resolve_can_config(config, *, sysfs_root=Path("/sys/class/net"), require_up=True):
    """Preserve arm semantics and feedback offsets across renames/reboots.

    Configured names are fallback names only when no serials are configured.
    When serials are present, a missing device must never fall back to another
    arm that happens to have the expected interface name.
    """
    names = {arm: config[f"{arm}_can"] for arm in ("left", "right")}
    if (len(set(names.values())) != 2 or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
            for name in names.values())):
        raise ValueError("Expected distinct CAN interface names")
    offsets = feedback_id_offsets(config, tuple(names.values()))
    serials = config.get("can_usb_serials")
    interfaces = discover_can_interfaces(sysfs_root)
    resolved = dict(names)
    if serials is not None:
        if (not isinstance(serials, dict) or set(serials) != {"left", "right"}
                or any(not isinstance(value, str) or not value.strip() for value in serials.values())
                or len(set(serials.values())) != 2):
            raise ValueError("can_usb_serials must contain distinct nonempty left/right serials")
        for arm, serial in serials.items():
            matches = [name for name, info in interfaces.items() if info["serial"] == serial]
            if len(matches) != 1:
                raise RuntimeError(f"Piper {arm} CAN adapter serial {serial} has {len(matches)} matches; "
                                   f"detected CAN interfaces: {interfaces}")
            resolved[arm] = matches[0]
    missing = [name for name in resolved.values() if name not in interfaces]
    if missing:
        raise RuntimeError(f"Piper CAN interfaces are missing: {missing}; detected: {interfaces}. "
                           "Activate the CAN adapters before inference.")
    down = [name for name in resolved.values() if not interfaces[name]["up"]]
    if require_up and down:
        raise RuntimeError(f"Piper CAN adapters were found but are DOWN: {down}. "
                           "Run CAN setup at 1000000 bit/s before inference; "
                           "a reboot or USB reconnect resets the link configuration.")
    return {**config, **{f"{arm}_can": name for arm, name in resolved.items()},
            "feedback_id_offsets": {resolved[arm]: offsets[names[arm]] for arm in names}}
