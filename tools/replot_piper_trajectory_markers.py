"""Redraw saved offline trajectories with visible label sample markers.

CPU-only presentation update: never imports a policy or changes predictions,
labels, metrics, checkpoints, or evaluator source fingerprints.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STYLE = "collected_action_circle_markers_v1"
TASK_TITLES = {"clean_table": "清理桌面", "placemat": "铺桌垫", "fold_towel": "折毛巾并放入盘子"}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(base):
    for steps in (10, 2, 4, 6):
        parent = base if steps == 10 else base / f"steps_{steps:02d}"
        for task, title in TASK_TITLES.items():
            folder = parent / task
            if not (folder / "manifest.json").is_file():
                continue
            manifest = read_json(folder / "manifest.json")
            if int(manifest["model"]["num_inference_steps"]) != steps:
                raise ValueError(f"Mismatched denoising step count: {folder}")
            yield folder, title, manifest


def jobs_for(base):
    jobs = []
    for folder, title, manifest in groups(base):
        reports = sorted((folder / "episodes").glob("episode_*.json"))
        if len(reports) != manifest["expected_episodes"]:
            raise ValueError(f"Expected complete evaluation artifacts: {folder}")
        for report in reports:
            jobs.append((str(report), title, manifest["model"]["step"],
                         manifest["model"]["num_inference_steps"]))
    return jobs


def render(job):
    report_path, title, checkpoint_step, denoising_steps = job
    report_path = Path(report_path)
    stem = report_path.with_suffix("")
    archive = stem.with_suffix(".npz")
    stamp_path = stem.parent.parent / "plot_metadata" / (stem.name + ".json")
    stamp_path.parent.mkdir(exist_ok=True)
    before = file_hash(archive)
    report_before = file_hash(report_path)
    if stamp_path.exists():
        saved = read_json(stamp_path)
        if (saved.get("style") == STYLE and saved.get("npz_sha256") == before and
                saved.get("metrics_sha256") == report_before and
                all(stem.with_suffix("." + ext).is_file() for ext in ("png", "pdf"))):
            return str(stem)
    report = read_json(report_path)
    with np.load(archive, allow_pickle=False) as arrays:
        times = arrays["timestamps"].copy()
        labels = arrays["label"].copy()
        predicted = arrays["prediction"].copy()
        source_id = int(arrays["source_episode_index"])
    if labels.shape != predicted.shape or labels.shape != (len(times), 14):
        raise ValueError(f"Invalid saved arrays: {archive}")
    if not all(np.isfinite(a).all() for a in (times, labels, predicted)):
        raise ValueError(f"Nonfinite values: {archive}")
    if source_id != report["source_episode_index"]:
        raise ValueError(f"Episode mismatch: {archive}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    # Retain all 30 Hz values in both lines; sparsify only decorative markers.
    marker_stride = max(1, math.ceil(len(times) / 280))
    marker_indices = np.arange(0, len(times), marker_stride)
    fig, axes = plt.subplots(6, 2, figsize=(13.6, 13.2), sharex=True)
    fig.subplots_adjust(left=.073, right=.972, bottom=.092, top=.885, hspace=.56, wspace=.22)
    fig.text(.073, .972, f"双臂 Piper  |  {title}  |  采集数据离线预测  |  轨迹 {source_id}",
             fontsize=17, va="top")
    fig.text(.073, .942, f"Checkpoint step {checkpoint_step:,}  ·  {denoising_steps} 步去噪  ·  "
             "30 Hz  ·  每 32 帧读取采集观测  ·  32 步预测块", fontsize=10, color="#626970", va="top")
    black, blue = "#292d32", "#2882b4"
    fig.legend(handles=[Line2D([0], [0], color=black, lw=.9, marker="o", markersize=3.0,
                              markeredgewidth=0, label="采集 action 标签（圆点为采样点）"),
                        Line2D([0], [0], color=blue, lw=1., label="模型预测动作")],
               loc="upper right", bbox_to_anchor=(.972, .925), frameon=False, ncol=2,
               handlelength=2.6, columnspacing=2.4, fontsize=10)
    for joint in range(6):
        for col, (arm, offset) in enumerate((("左臂", 0), ("右臂", 7))):
            ax = axes[joint, col]
            dim = joint + offset
            ax.plot(times, labels[:, dim], color=black, lw=.8, alpha=.9, zorder=2)
            ax.plot(times, predicted[:, dim], color=blue, lw=.9, alpha=.9, zorder=3)
            # Put the markers above both lines so coinciding predictions cannot hide them.
            ax.plot(times[marker_indices], labels[marker_indices, dim], linestyle="none",
                    marker="o", markersize=2.6, markeredgewidth=0, color=black, alpha=.9, zorder=4)
            ax.set_title(f"{arm} · 关节 {joint + 1}", loc="left", fontsize=10.8, pad=7)
            ax.set_ylabel("角度（rad）", fontsize=9.5)
            ax.set_xlim(0, max(float(times[-1]), 1 / 30))
            ax.grid(True, color="#e3e7eb", linewidth=.55, alpha=.8)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.tick_params(labelsize=9, length=3)
            if joint == 5:
                ax.set_xlabel("采集时间（s）")
    fig.text(.073, .052,
             f"12 关节 MAE {report['joint_mae_deg']:.2f}°  ·  RMSE {report['joint_rmse_deg']:.2f}°  ·  "
             f"{len(times):,} 帧  ·  数据集轨迹编号 {report['episode_index']}", fontsize=9, color="#626970")
    fig.text(.073, .033,
             f"黑线与圆点：采集 action 标签；蓝线：离线预测。连线保留全部 30 Hz 数据，圆点每 {marker_stride} 帧标一个，便于看清。",
             fontsize=9, color="#626970")
    fig.text(.073, .014,
             "使用训练数据回放；不是本次真机执行反馈。未做 RTC、平滑或时间偏移；末块仅比较真实存在的帧。",
             fontsize=9, color="#626970")
    try:
        for extension in ("png", "pdf"):
            destination = stem.with_suffix("." + extension)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            fig.savefig(temporary, format=extension, dpi=170)
            temporary.replace(destination)
    finally:
        plt.close(fig)
    if file_hash(archive) != before or file_hash(report_path) != report_before:
        raise RuntimeError("Numeric inputs changed during rendering")
    write_json(stamp_path, dict(style=STYLE, npz_sha256=before, metrics_sha256=report_before,
                              marker_stride=marker_stride, marker_count=len(marker_indices),
                              line_frames=len(times), label="collected action", prediction="offline model action",
                              updated_at=datetime.now().astimezone().isoformat(timespec="seconds")))
    return str(stem)


def annotate_indexes(base):
    note = ('<p id="marker-style-note"><b>带采样点版本：</b>黑线和圆点为采集 action 标签，蓝线为离线模型预测。'
            '连线保留全部 30 Hz 数据；圆点隔帧显示，具体间隔写在每张图下方。'
            '与之前真机图中的“实际关节反馈”含义不同。预测数值和指标保持原样。</p>')
    paths = [base / "denoising_comparison.html"]
    paths += [folder / "index.html" for folder, _, _ in groups(base)]
    for path in paths:
        if not path.exists():
            continue
        content = path.read_text()
        if 'id="marker-style-note"' not in content:
            content = content.replace("</h1>", "</h1>" + note, 1)
        # Browsers that have an older gallery open should fetch the new PNGs.
        content = content.replace('.png"', f'.png?style={STYLE}"')
        path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/piper-dataset-eval/20261028"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, help="Preview this many pending plots before a full redraw")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    os.chdir(ROOT)
    base = args.output_dir.resolve()
    with (base / "replot.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        all_jobs = jobs_for(base)
        jobs = all_jobs if args.limit is None else all_jobs[:args.limit]
        (base / "replot.pid").write_text(str(os.getpid()) + "\n")
        started = time.monotonic()
        status = dict(style=STYLE, state="running", completed=0, scheduled=len(jobs),
                      total_available=len(all_jobs), workers=args.workers, pid=os.getpid())
        write_json(base / "replot_status.json", status)
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
                for i, result in enumerate(pool.map(render, jobs, chunksize=1), 1):
                    status.update(completed=i, last_plot=result, elapsed_seconds=time.monotonic() - started)
                    if i % 20 == 0 or i == len(jobs):
                        write_json(base / "replot_status.json", status)
                        print("REPLOT_PROGRESS", json.dumps(status, ensure_ascii=False), flush=True)
            annotate_indexes(base)
            status.update(state="complete" if len(jobs) == len(all_jobs) else "preview_complete")
            write_json(base / "replot_status.json", status)
            print("REPLOT_FINISHED", json.dumps(status, ensure_ascii=False), flush=True)
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            write_json(base / "replot_status.json", status)
            raise


if __name__ == "__main__":
    main()
