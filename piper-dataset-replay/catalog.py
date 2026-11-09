"""Read original collection files, using the existing explicit unit map."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
TITLES = {"piper_clean_table_lerobot": "清理桌面", "piper_set_placemat_lerobot": "铺桌垫",
          "fold_towel_mit": "折毛巾", "connector_plugin": "物体抓取"}


def read_json(path):
    return json.loads(path.read_text())


class Catalog:
    def __init__(self, project):
        self.project = Path(project).resolve()
        interface = read_json(self.project / "resources/piper/dataset_v3/meta/piper_interface.json")
        self.sources = {}
        for source in interface["sources"]:
            root = self.project / source["root"]
            if not (root / "meta/info.json").is_file():
                continue
            info = read_json(root / "meta/info.json")
            episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
            self.sources[root.name] = dict(source=source, root=root, info=info, episodes=episodes)
        if not self.sources:
            raise ValueError("找不到原始采集数据")

    def listing(self):
        return [dict(id=key, title=TITLES.get(key, key), fps=float(v["info"]["fps"]),
                     count=len(v["episodes"]), original_unit=v["source"]["joint_unit"],
                     episodes=[dict(index=int(e["episode_index"]), frames=int(e["length"]),
                                    duration=int(e["length"])/float(v["info"]["fps"])) for e in v["episodes"]])
                for key, v in self.sources.items()]

    def load(self, source_id, episode_id, field="action"):
        if field not in ("action", "state"):
            raise ValueError("请选择 action 或 state")
        if not isinstance(source_id, str) or source_id not in self.sources:
            raise ValueError("未知数据集")
        if type(episode_id) is not int:
            raise ValueError("轨迹编号必须是整数")
        source = self.sources[source_id]
        row = next((e for e in source["episodes"] if e["episode_index"] == episode_id), None)
        if row is None:
            raise ValueError("该原始轨迹不存在")
        root, info, spec = source["root"], source["info"], source["source"]
        fields = dict(episode_index=episode_id, episode_chunk=episode_id // int(info["chunks_size"]))
        path = root / info["data_path"].format(**fields)
        table = pq.read_table(path)
        raw = np.asarray(table[spec[field + "_key"]].to_pylist(), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] not in (14, 15) or len(raw) != row["length"] or not len(raw):
            raise ValueError("原始数据维度或轨迹长度不匹配")
        if not np.isfinite(raw).all():
            raise ValueError("原始轨迹含有非有限数值")
        target = raw[:, :14].copy()
        if spec["joint_unit"] == "degrees":
            target[:, JOINTS] = np.deg2rad(target[:, JOINTS])
        elif spec["joint_unit"] != "radians":
            raise ValueError("缺少明确的角度单位")
        if np.any(target[:, [6, 13]] < 0) or np.any(target[:, [6, 13]] > 1):
            raise ValueError("夹爪值必须在采集定义的 0–1 范围内")
        timestamps = np.asarray(table["timestamp"], dtype=np.float64)
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("原始轨迹时间戳无效")
        np.testing.assert_array_equal(np.asarray(table["frame_index"]), np.arange(len(raw)))
        videos = [root / info["video_path"].format(**fields, video_key=key) for key in spec["cameras"]]
        if not all(p.is_file() for p in videos):
            raise FileNotFoundError("原始三路录像不完整")
        target.setflags(write=False)
        return dict(id=source_id, title=TITLES.get(source_id, source_id), episode=episode_id,
                    field=field, fps=float(info["fps"]), frames=len(target),
                    task=" / ".join(row.get("tasks", [])), original_unit=spec["joint_unit"],
                    parquet=str(path), videos=videos, target=target, timestamps=timestamps,
                    auxiliary=None if raw.shape[1] == 14 else raw[:, 14].copy())


def public_episode(episode):
    if episode is None:
        return None
    return {key: episode[key] for key in ("id", "title", "episode", "field", "fps", "frames", "task", "original_unit")}
