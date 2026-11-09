"""Background queue for offline evaluation on selected Piper tasks.

Waits for the existing 10-step job's lock before acquiring the GPU. All variants
use the same recorded observations, checkpoints, seed and unmodified sampler.
"""
import argparse
import csv
from datetime import datetime
import fcntl
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluate_piper_trajectories import TASKS, digest, read_json, write_json

STEP_ORDER = (10, 2, 4, 6)
DEFAULT_TASKS = ("clean_table", "placemat")


def result_directory(base, steps):
    # Keep the existing 10-step artifacts at their original location.
    return base if steps == 10 else base / f"steps_{steps:02d}"


def model_fingerprints(root, tasks=DEFAULT_TASKS):
    result = {}
    for task in tasks:
        run_name, _, _ = TASKS[task]
        run = root / "postcheckpoints" / run_name
        weights = run / "checkpoints/weights"
        latest = read_json(weights / "weights_manifest.json")["latest"]
        checkpoint = (weights / latest["path"]).resolve()
        if not checkpoint.is_relative_to(weights.resolve()) or not checkpoint.is_file():
            raise ValueError("Invalid checkpoint path")
        result[task] = dict(checkpoint=str(checkpoint.relative_to(root)), step=int(latest["step"]),
                            size=checkpoint.stat().st_size, mtime_ns=checkpoint.stat().st_mtime_ns,
                            config_sha256=digest(run / "config.yaml"),
                            normalization_sha256=digest(run / "dataset_stats.json"))
    for name in ("meta/info.json", "meta/source_episodes.json", "meta/episodes.jsonl", "meta/tasks.jsonl"):
        result[name] = digest(root / "resources/piper/dataset_v3" / name)
    for name in ("tools/evaluate_piper_trajectories.py", "tools/replot_piper_trajectory_markers.py",
                 "src/wam/inference/piper.py", "src/wam/inference/frame_joint.py"):
        result[name] = digest(root / name)
    return result


def write_overview(base, status, tasks=DEFAULT_TASKS, step_order=STEP_ORDER):
    rows = []
    summaries = []
    for steps in step_order:
        folder = result_directory(base, steps)
        for task in tasks:
            _, title, expected = TASKS[task]
            path = folder / task / "summary.json"
            summary = read_json(path) if path.exists() else {}
            count = int(summary.get("completed_episodes", 0))
            mae = summary.get("joint_mae_deg")
            rmse = summary.get("joint_rmse_deg")
            relative = (folder / task / "index.html").relative_to(base).as_posix()
            label = html.escape(title)
            if (folder / task / "index.html").exists():
                label = f'<a href="{relative}">{label}</a>'
            rows.append(f'<tr><td>{steps}</td><td>{label}</td><td>{count} / {expected}</td>'
                        f'<td>{mae:.3f}°</td><td>{rmse:.3f}°</td></tr>' if mae is not None and rmse is not None else
                        f'<tr><td>{steps}</td><td>{label}</td><td>{count} / {expected}</td><td>—</td><td>—</td></tr>')
            summaries.append(dict(inference_steps=steps, task=task, **summary))
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    status = {**status, "updated_at": timestamp, "results": summaries}
    write_json(base / "sweep_status.json", status)
    state = {"waiting_for_lock": "等待此目录的评测任务完成", "running": "后台评估进行中",
             "complete": "全部评估已完成", "failed": "评估已停止，请查看日志"}.get(status["state"], status["state"])
    task_description = " + ".join(f"{html.escape(TASKS[t][1])} {TASKS[t][2]} 条" for t in tasks)
    expected_total = sum(TASKS[t][2] for t in tasks)
    plot_count = expected_total * len(step_order)
    step_description = " / ".join(map(str, sorted(step_order)))
    comparison_links = ('<p><a href="denoising_summary.csv">各步数误差与耗时汇总 CSV</a> · '
                        '<a href="denoising_metrics.csv">全部逐轨迹指标 CSV</a></p>') if (base / "denoising_summary.csv").exists() else ''
    (base / "denoising_comparison.html").write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta http-equiv="refresh" content="60"><title>Piper 去噪步数对比</title>
<style>body{{font:17px system-ui;max-width:1100px;margin:45px auto;padding:0 20px;line-height:1.7;color:#263442}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #ddd}}a{{color:#2375a3}}</style>
<h1>Piper：{step_description} 步去噪对比</h1><p>{state}。当前步骤：{status.get('active_steps') or '—'}。</p>
<p>{task_description}，各用对应权重。每组均读取采集 RGB 和 state，每 32 帧预测 32 步绝对关节动作；
seed 0、SDPA、不做 RTC 或平滑。黑线及圆点是采集 action 标签，蓝线是预测，图中关节单位 rad。
连线保留全部 30 Hz 数据，圆点隔帧显示。</p>
<p>各组最终均覆盖 {expected_total:,} 条轨迹，共 {plot_count:,} 张 PNG 和 {plot_count:,} 份 PDF。下表的误差使用已完成轨迹汇总；
不同组完成数量不同时，暂不用于横向排名。数据来自训练集，不能代表真机成功率。</p>
<table><thead><tr><th>去噪步数</th><th>任务 / 图片索引</th><th>已完成</th><th>MAE</th><th>RMSE</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table><p>更新时间：{timestamp}，此页每 60 秒刷新。</p>
<p><a href="sweep.log">队列日志</a> · <a href="sweep_status.json">队列状态</a> ·
<a href="sweep_plan.json">固定配置</a></p>{comparison_links}</html>''')


def export_comparison(base, tasks=DEFAULT_TASKS, step_order=STEP_ORDER):
    records, aggregates = [], []
    for steps in step_order:
        for task in tasks:
            folder = result_directory(base, steps) / task
            summary = read_json(folder / "summary.json")
            if not summary["complete"]:
                raise RuntimeError(f"Incomplete evaluation: {steps} steps / {task}")
            episodes = [read_json(p) for p in sorted((folder / "episodes").glob("episode_*.json"))]
            for r in episodes:
                records.append(dict(inference_steps=steps, task=task,
                                    **{k: r[k] for k in ("source_episode_index", "episode_index", "frames", "chunks",
                                                        "joint_mae_deg", "joint_rmse_deg", "mean_latency_seconds")}))
            chunks = sum(r["chunks"] for r in episodes)
            aggregates.append(dict(inference_steps=steps, task=task,
                                   **{k: summary[k] for k in ("completed_episodes", "evaluated_frames", "inference_chunks",
                                                             "joint_mae_deg", "joint_rmse_deg")},
                                   mean_latency_seconds=sum(r["mean_latency_seconds"] * r["chunks"] for r in episodes) / chunks))
    for filename, rows in (("denoising_metrics.csv", records), ("denoising_summary.csv", aggregates)):
        with (base / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/piper-dataset-eval/20261028"))
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(DEFAULT_TASKS))
    parser.add_argument("--steps", nargs="+", type=int, choices=[2, 4, 6, 10], default=list(STEP_ORDER),
                        help="Denoising variants in execution order")
    args = parser.parse_args()
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.steps)) != len(args.steps):
        parser.error("tasks and steps must not contain duplicates")
    os.chdir(ROOT)
    base = args.output_dir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    with (base / "sweep.lock").open("w") as queue_lock, (base / "evaluation.lock").open("a") as gpu_lock:
        try:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A denoising sweep is already queued or running")
        identity = dict(steps=args.steps, tasks=args.tasks,
                        expected_episodes_per_step=sum(TASKS[t][2] for t in args.tasks), seed=0,
                        attention_backend="sdpa", chunk_stride=32, rtc=False, smoothing=False,
                        plot_style="markers", fingerprints=model_fingerprints(ROOT, args.tasks))
        plan_path = base / "sweep_plan.json"
        if plan_path.exists() and read_json(plan_path) != identity:
            raise ValueError("Sweep inputs changed; use a different output directory")
        write_json(plan_path, identity)
        (base / "sweep.pid").write_text(str(os.getpid()) + "\n")
        status = dict(state="waiting_for_lock", active_steps=None, pid=os.getpid(),
                      started_at=datetime.now().astimezone().isoformat(timespec="seconds"))
        print("SWEEP_WAITING", json.dumps(status), flush=True)
        # This is a detached background queue. Waiting here does not block the chat.
        while True:
            write_overview(base, status, args.tasks, args.steps)
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(15)
        try:
            for steps in args.steps:
                folder = result_directory(base, steps)
                folder.mkdir(parents=True, exist_ok=True)
                summaries = {}
                for task in args.tasks:
                    if model_fingerprints(ROOT, args.tasks) != identity["fingerprints"]:
                        raise ValueError("Checkpoint, configuration or evaluation code changed while queued")
                    status.update(state="running", active_steps=steps, active_task=task)
                    write_overview(base, status, args.tasks, args.steps)
                    command = [sys.executable, "-u", "tools/evaluate_piper_trajectories.py", "--task", task,
                               "--output-dir", str(folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder),
                               "--inference-steps", str(steps), "--plot-style", "markers",
                               "--attention-backend", "sdpa", "--seed", "0"]
                    print("SWEEP_START", steps, task, json.dumps(command), flush=True)
                    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
                    log_path = folder / f"{task}_evaluation.log"
                    with log_path.open("ab", buffering=0) as log:
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   stdin=subprocess.DEVNULL, pass_fds=(queue_lock.fileno(), gpu_lock.fileno()))
                        while process.poll() is None:
                            status["worker_pid"] = process.pid
                            write_overview(base, status, args.tasks, args.steps)
                            time.sleep(15)
                        if process.returncode:
                            raise RuntimeError(f"{steps}-step / {task} evaluation failed (exit {process.returncode}); see {log_path}")
                    summaries[task] = read_json(folder / task / "summary.json")
                    if not summaries[task]["complete"]:
                        raise RuntimeError(f"{steps}-step / {task} evaluation did not finish all trajectories")
                write_json(folder / "summary.json", summaries)
                print("SWEEP_STEP_COMPLETE", steps, flush=True)
            export_comparison(base, args.tasks, args.steps)
            status.update(state="complete", active_steps=None, active_task=None, worker_pid=None)
            write_overview(base, status, args.tasks, args.steps)
            print("SWEEP_COMPLETE", flush=True)
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            write_overview(base, status, args.tasks, args.steps)
            raise


if __name__ == "__main__":
    main()
