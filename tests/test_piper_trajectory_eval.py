"""Offline evaluation alignment tests, without model loading or hardware access."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tools.evaluate_piper_trajectories import (
    IMAGE_KEYS, chunk_starts, decode_current_frames, metrics, replay_episode,
    selected_episodes, verify_identity, effective_inference_steps, LEGACY_10_STEP_EVALUATOR,
)
from tools import evaluate_piper_trajectories as evaluator
from tools.run_piper_denoising_sweep import STEP_ORDER, result_directory, model_fingerprints, write_overview


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_same_frame_labels_tail_and_no_label_leakage(self):
        length = 67
        state = np.repeat(np.arange(length, dtype=np.float32)[:, None], 14, axis=1)
        starts = chunk_starts(length)
        data = dict(state=state, label=np.full_like(state, -999), starts=starts, prompt="task",
                    images={key: [np.full((2, 2, 3), i, np.uint8) for i in starts] for key in IMAGE_KEYS})
        seen = []

        class Policy:
            def infer(self, obs):
                self_keys = {"state", "prompt", *IMAGE_KEYS}
                if set(obs) != self_keys:
                    raise AssertionError("Unexpected input (possible action label leakage)")
                start = int(obs["state"][0])
                seen.append(start)
                for key in IMAGE_KEYS:
                    np.testing.assert_array_equal(obs[key], np.full((2, 2, 3), start, np.uint8))
                return dict(actions=np.repeat(np.arange(start, start + 32)[:, None], 14, axis=1),
                            latency_seconds=.1)

        predictions, latency = replay_episode(Policy(), data)
        self.assertEqual(seen, [0, 32, 64])
        np.testing.assert_array_equal(predictions, state)
        self.assertEqual(len(latency), 3)
        self.assertEqual(predictions.shape, (67, 14))

    def test_joints_exclude_grippers(self):
        label = np.zeros((3, 14))
        prediction = label.copy()
        prediction[:, [6, 13]] = 100
        m = metrics(label, prediction)
        self.assertEqual(m["joint_mae_rad"], 0)
        self.assertEqual(m["gripper_mae"], [100, 100])

    def test_resume_rejects_different_model_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "manifest.json").write_text(json.dumps({"identity": {"step": 123, "seed": 0}}))
            verify_identity(out, {"step": 123, "seed": 0})
            with self.assertRaisesRegex(ValueError, "different evaluation"):
                verify_identity(out, {"step": 456, "seed": 0})

    def test_only_reviewed_ten_step_resume_is_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            previous = dict(evaluator_sha256=LEGACY_10_STEP_EVALUATOR, inference_steps=10, seed=0)
            (out / "manifest.json").write_text(json.dumps({"identity": previous}))
            verify_identity(out, {**previous, "evaluator_sha256": "current"})
            for mutation in ({"inference_steps": 4}, {"seed": 1}):
                with self.assertRaisesRegex(ValueError, "different evaluation"):
                    verify_identity(out, {**previous, "evaluator_sha256": "current", **mutation})
            (out / "manifest.json").write_text(json.dumps({"identity": {**previous, "evaluator_sha256": "unknown"}}))
            with self.assertRaisesRegex(ValueError, "different evaluation"):
                verify_identity(out, {**previous, "evaluator_sha256": "current"})

    def test_step_selection_and_separate_outputs(self):
        self.assertEqual(effective_inference_steps(10, None), 10)
        for steps in (2, 4, 6):
            self.assertEqual(effective_inference_steps(10, steps), steps)
        for steps in (0, -1):
            with self.assertRaises(ValueError):
                effective_inference_steps(10, steps)
        base = Path("runs/example")
        self.assertEqual(STEP_ORDER, (10, 2, 4, 6))
        self.assertEqual(result_directory(base, 10), base)
        self.assertEqual(len({result_directory(base, s) for s in STEP_ORDER}), 4)

    def test_override_reaches_policy_and_saved_manifest(self):
        # Exercise the real evaluator with a fake policy, without loading CUDA.
        initial_cwd = Path.cwd()
        try:
            for requested in (None, 2, 4, 6):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    run = root / "postcheckpoints/test_run"
                    run.mkdir(parents=True)
                    (run / "config.yaml").write_text("eval_num_inference_steps: 10\ndata:\n  train:\n    source_root: datasets/test\n")
                    (run / "dataset_stats.json").write_text("{}")
                    checkpoint = run / "weight.pt"
                    checkpoint.write_bytes(b"fake weight")
                    dataset = root / "dataset"
                    (dataset / "meta").mkdir(parents=True)
                    (dataset / "meta/info.json").write_text('{"fps":30}')
                    (dataset / "meta/source_episodes.json").write_text("[]")
                    for name in ("piper.py", "frame_joint.py"):
                        path = root / "src/wam/inference" / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("# fake source\n")
                    steps = 10 if requested is None else requested
                    policy = Mock()
                    policy.audit = dict(step=9087, num_inference_steps=steps, inference_attention_backend="sdpa")
                    policy.infer.return_value = dict(actions=np.zeros((32, 14), np.float32), latency_seconds=.1)
                    factory = Mock(return_value=policy)
                    row = dict(episode_index=435, source_episode_index=0, length=3, tasks=["task"])
                    data = dict(state=np.zeros((3, 14), np.float32), label=np.zeros((3, 14), np.float32),
                                timestamps=np.arange(3)/30, starts=np.array([0]), prompt="task",
                                images={key: [np.zeros((2, 2, 3), np.uint8)] for key in IMAGE_KEYS})
                    args = SimpleNamespace(project_root=root, dataset_dir=dataset, output_dir=root / "result",
                                           task="clean_table", inference_steps=requested, seed=0,
                                           attention_backend="sdpa", limit=None, device="cpu")
                    fake_module = SimpleNamespace(PiperPolicy=factory, select_checkpoint=lambda _: (checkpoint, 9087))
                    with patch.dict(sys.modules, {"wam.inference.piper": fake_module}), \
                         patch.dict(evaluator.TASKS, {"clean_table": ("test_run", "Test", 1)}), \
                         patch.object(evaluator, "selected_episodes", return_value=[row]), \
                         patch.object(evaluator, "read_episode", return_value=data), \
                         patch.object(evaluator, "render_plot"), patch.object(evaluator, "write_index", return_value={"complete":True}), \
                         patch("builtins.print"):
                        evaluator.evaluate(args)
                    self.assertEqual(factory.call_args.kwargs["inference_steps"], requested)
                    manifest = json.loads((root / "result/clean_table/manifest.json").read_text())
                    self.assertEqual(manifest["identity"]["inference_steps"], steps)
                    self.assertEqual(manifest["model"]["num_inference_steps"], steps)
        finally:
            os.chdir(initial_cwd)

    def test_selects_only_matching_task_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/source_episodes.json").write_text(json.dumps([
                dict(episode_index=1, source_root="datasets/a", source_episode_index=0),
                dict(episode_index=2, source_root="datasets/b", source_episode_index=0)]))
            (root / "meta/episodes.jsonl").write_text('\n'.join(json.dumps(
                dict(episode_index=i, length=100, tasks=[str(i)])) for i in [1, 2]))
            self.assertEqual([r["episode_index"] for r in selected_episodes(root, "datasets/b")], [2])

    def test_fold_only_sweep_does_not_require_deleted_task_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "postcheckpoints/fold_the_towel"
            weights = run / "checkpoints/weights"
            weights.mkdir(parents=True)
            (weights / "slot_1.pt").write_bytes(b"test weight")
            (weights / "weights_manifest.json").write_text(json.dumps(
                {"latest": {"path": "slot_1.pt", "step": 7210}}))
            for name in ("config.yaml", "dataset_stats.json"):
                (run / name).write_text("{}")
            for name in ("resources/piper/dataset_v3/meta/info.json",
                         "resources/piper/dataset_v3/meta/source_episodes.json",
                         "resources/piper/dataset_v3/meta/episodes.jsonl",
                         "resources/piper/dataset_v3/meta/tasks.jsonl",
                         "tools/evaluate_piper_trajectories.py", "tools/replot_piper_trajectory_markers.py",
                         "src/wam/inference/piper.py", "src/wam/inference/frame_joint.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test")
            before = model_fingerprints(root, ["fold_towel"])
            self.assertEqual(before["fold_towel"]["step"], 7210)
            self.assertNotIn("placemat", before)
            (run / "dataset_stats.json").write_text('{"changed":true}')
            self.assertNotEqual(model_fingerprints(root, ["fold_towel"]), before)

    def test_fold_only_overview_counts_and_step_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            folder = base / "steps_02/fold_towel"
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps(
                dict(completed_episodes=1, joint_mae_deg=1.5, joint_rmse_deg=2.)))
            (folder / "index.html").write_text("test")
            write_overview(base, dict(state="running", active_steps=2), ["fold_towel"], [2, 4, 6, 10])
            status = json.loads((base / "sweep_status.json").read_text())
            self.assertEqual(len(status["results"]), 4)
            self.assertEqual(status["results"][0]["completed_episodes"], 1)
            content = (base / "denoising_comparison.html").read_text()
            self.assertIn("900 张 PNG", content)
            self.assertIn('href="steps_02/fold_towel/index.html"', content)
            self.assertNotIn("placemat/", content)

    def test_video_timestamp_and_missing_frame_validation(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "test.mp4"
            with av.open(str(video), "w") as container:
                stream = container.add_stream("mpeg4", rate=30)
                stream.width = stream.height = 16
                stream.pix_fmt = "yuv420p"
                for i in range(4):
                    frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 30, np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            frames = decode_current_frames(video, np.array([0, 2]), np.arange(4) / 30)
            self.assertEqual(len(frames), 2)
            self.assertLess(abs(float(frames[1].mean()) - 60), 5)
            with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                decode_current_frames(video, np.array([2]), np.arange(4) / 30 + .04)
            with self.assertRaisesRegex(ValueError, "Missing video frames"):
                decode_current_frames(video, np.array([0, 5]), np.arange(6) / 30)


if __name__ == "__main__":
    unittest.main()
