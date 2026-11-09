"""Server dependency coverage and image-binary protection, without a GPU."""
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from wam.utils.server_compat import BASE_VERSIONS, PROFILE_VERSIONS, attention_smoke, check_install_plan, installed_versions, version_errors

ROOT = Path(__file__).resolve().parents[1]


def requirements(path):
    return [Requirement(line) for line in path.read_text().splitlines()
            if line.strip() and not line.startswith(("#", "-"))]


class ServerProfileTests(unittest.TestCase):
    def test_attention_backward_does_not_retain_graph(self):
        import torch
        grad = torch.autograd.grad
        calls = []

        def single_backward(*args, **kwargs):
            # Ignore PyTorch's own graph-construction calls inside FlexAttention.
            if sys._getframe(1).f_code is attention_smoke.__code__:
                self.assertIs(kwargs.get("retain_graph"), False)
                self.assertIs(kwargs.get("create_graph"), False)
                calls.append(kwargs)
            return grad(*args, **kwargs)

        with patch("torch.autograd.grad", new=single_backward):
            result = attention_smoke("cpu")
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["masked_flex_forward_backward"], "passed")

    def test_cuda_local_builds_are_accepted_without_replacement(self):
        versions = {**PROFILE_VERSIONS, "torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130"}
        self.assertEqual(version_errors(versions, (3, 12, 9), "13.0"), [])

    def test_wrong_python_cuda_and_old_training_packages_are_rejected(self):
        for versions, python, cuda in ((PROFILE_VERSIONS, (3, 11), "13.0"),
                                       (PROFILE_VERSIONS, (3, 12), "12.8"),
                                       ({**PROFILE_VERSIONS, "torchcodec": "0.5"}, (3, 12), "13.0"),
                                       ({**PROFILE_VERSIONS, "torch": "2.10.0rc1"}, (3, 12), "13.0")):
            with self.subTest(python=python, cuda=cuda, versions=versions):
                self.assertTrue(version_errors(versions, python, cuda))
        self.assertEqual(version_errors(BASE_VERSIONS, (3, 12), "13.0", base_only=True), [])

    def test_install_plan_preserves_compiled_image_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            for name in ("torch", "torchvision", "vllm", "triton", "nvidia-cublas", "flashinfer_python"):
                # Even the same public version may replace a custom CUDA build.
                plan.write_text(json.dumps({"install": [{"metadata": {"name": name, "version": "2.10.0"}}]}))
                with self.subTest(name=name), self.assertRaises(ValueError):
                    check_install_plan(plan)
            plan.write_text(json.dumps({"install": [{"metadata": {"name": "hydra-core", "version": "1.3.2"}}]}))
            self.assertEqual(check_install_plan(plan)["planned_packages"], 1)

    def test_first_visible_distribution_wins_over_shadowed_image_package(self):
        distributions = [SimpleNamespace(metadata={"Name": "numpy"}, version="2.2.6"),
                         SimpleNamespace(metadata={"Name": "numpy"}, version="1.26.4")]
        with patch("wam.utils.server_compat.metadata.distributions", return_value=distributions):
            self.assertEqual(installed_versions()["numpy"], "2.2.6")

    def test_failed_runtime_report_identifies_phase_and_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            report = Path(directory) / "report.json"
            plan.write_text(json.dumps({"install": [{"metadata": {"name": "torch"}}]}))
            process = subprocess.run([sys.executable, str(ROOT / "tools/check_server_runtime.py"),
                                      "--check-plan", str(plan), "--output", str(report)],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 1)
            result = json.loads(report.read_text())
            self.assertEqual(result["phase"], "installation_plan")
            self.assertIn("protected image binaries", result["error"])
            self.assertIn("check_install_plan", result["traceback"])

    @unittest.skipIf(tomllib is None, "Dependency coverage parsing requires Python 3.11+; the server uses 3.12")
    def test_full_install_list_covers_declared_training_and_logging_dependencies(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        required = project["dependencies"] + project["optional-dependencies"]["train"] + project["optional-dependencies"]["logging"]
        direct = {canonicalize_name(r.name): r for r in requirements(ROOT / "requirements/server-cu130-py312.txt")}
        # The base image supplies Torch and torchvision; no simulator extras are needed.
        self.assertEqual(set(direct), ({canonicalize_name(Requirement(r).name) for r in required}
                                      - {"torch", "torchvision"}) | {"scikit-learn"})
        self.assertEqual(str(direct["scikit-learn"].specifier), "==1.6.1")
        for line in required:
            req = Requirement(line)
            name = canonicalize_name(req.name)
            if name in PROFILE_VERSIONS:
                self.assertIn(PROFILE_VERSIONS[name], req.specifier)
            if name in direct and name not in PROFILE_VERSIONS:
                self.assertEqual(direct[name].specifier, req.specifier)
        pins = {canonicalize_name(r.name): r for r in requirements(ROOT / "requirements/server-cu130-py312.constraints.txt")}
        for name, version in PROFILE_VERSIONS.items():
            self.assertIn(version, pins[name].specifier)


if __name__ == "__main__":
    unittest.main()
