"""Run from the repository root; creates a new portable dataset, never edits sources."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wam.datasets.piper_prepare import prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="configs/data/piper_sources.json")
    parser.add_argument("--output", default="resources/piper/dataset")
    args = parser.parse_args()
    print(json.dumps(prepare(args.sources, args.output), ensure_ascii=False, indent=2))
