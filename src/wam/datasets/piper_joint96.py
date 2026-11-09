"""Dual-arm Piper data through the original Joint96 online RGB/label pipeline."""
import json
from pathlib import Path

from .joint96_posttrain import Joint96PosttrainDataset
from .lerobot.domain_aug_dataset import DomainAugmentedRobotVideoDataset, sample_domain
from .piper_prepare import CAMERAS, SCHEMA
from .robotwin_joint96_io import BoundedJoint96RGB
from .robotwin_joint96_io import official_text_padding
from wam.utils.logging_config import get_logger

logger = get_logger(__name__)


def source_episode_indices(root, manifest, source_root):
    """Resolve original-source membership without copying or rewriting data."""
    sources = {source["root"]: source for source in manifest["sources"]}
    members = {root for root, spec in sources.items()
               if source_root in (root, spec.get("source_group", root))}
    if not members:
        choices = sorted(set(sources) | {spec["source_group"] for spec in sources.values() if "source_group" in spec})
        raise ValueError(f"Unknown Piper source_root={source_root!r}; choose one of {choices}")
    rows = json.loads((Path(root) / "meta/source_episodes.json").read_text())
    indices = [row["episode_index"] for row in rows]
    if (any(type(index) is not int for index in indices)
            or sorted(indices) != list(range(manifest["episodes"]))
            or any(row["source_root"] not in sources for row in rows)):
        raise ValueError("Invalid Piper source-to-episode mapping")
    for source, spec in sources.items():
        if sum(row["source_root"] == source for row in rows) != spec["episodes"]:
            raise ValueError(f"Piper source episode count mismatch: {source}")
    selected = sorted(row["episode_index"] for row in rows if row["source_root"] in members)
    if not selected:
        raise ValueError(f"No episodes for Piper source: {source_root}")
    return selected


class PiperJoint96OnlineDataset(BoundedJoint96RGB, DomainAugmentedRobotVideoDataset, Joint96PosttrainDataset):
    def __init__(self, *, text_cache_source_root=None, source_root=None, **kwargs):
        roots = kwargs.get("dataset_dirs", [])
        if len(roots) != 1 or kwargs.get("domain") != "piper" or kwargs.get("episode_selection") is not None:
            raise ValueError("Piper requires one prepared dataset, domain=piper; use source_root to select a source")
        root = Path(roots[0])
        manifest = json.loads((root / "meta/piper_interface.json").read_text())
        info = json.loads((root / "meta/info.json").read_text())
        if (manifest.get("schema") != SCHEMA or manifest.get("joint_unit") != "radians"
                or manifest.get("camera_order") != list(CAMERAS)
                or info.get("robot_type") != "piper" or info.get("codebase_version") != "v2.1"
                or info.get("total_episodes") != manifest.get("episodes")
                or info.get("total_frames") != manifest.get("frames")
                or info.get("fps") != manifest.get("fps")):
            raise ValueError("Invalid prepared Piper interface; run tools/prepare_piper_data.py")
        if any(info["features"][key]["shape"] != [14] for key in ("action", "observation.state")):
            raise ValueError("Prepared Piper labels must be native joint14 before canonical80 mapping")
        self.piper_interface = manifest
        self.source_root = source_root
        if source_root is not None:
            kwargs["episode_selection"] = {
                "mode": "explicit", "indices": source_episode_indices(root, manifest, source_root),
            }
        super().__init__(**kwargs)
        logger.info("Piper source selection: source=%s episodes=%d frames=%d windows=%d",
                    source_root or "all", len(self._episode_starts),
                    sum(end - start for start, end in zip(self._episode_starts, self._episode_ends)),
                    len(self))

    def _get_cached_text_context(self, prompt):
        context, mask = super()._get_cached_text_context(prompt)
        return official_text_padding(context, mask)

    def __getitem__(self, index):
        previous = self._domain_params
        self._domain_params = sample_domain(self.image_domain_jitter, self.is_training_set, self.video_view_names)
        try:
            return super().__getitem__(index)
        finally:
            self._domain_params = previous
