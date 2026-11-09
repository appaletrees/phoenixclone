"""Locate Phoenix when this tool is bundled or installed beside its checkout."""
from pathlib import Path


def default_project_root():
    parent = Path(__file__).resolve().parent.parent
    if (parent / "src/wam/inference/piper_control.py").is_file():
        return parent
    return parent / "phoenix-mwa"


def default_interface_root(project):
    return Path(project).resolve().parent / "Desktop/piper_tele/sink-bot-control_bk/example_agilex"
