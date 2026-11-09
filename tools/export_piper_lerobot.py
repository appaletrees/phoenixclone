"""Convert existing Piper raw records to LeRobot v2.1 without hardware/model use."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from wam.inference.piper_lerobot_export import export_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='Default: write meta/data/videos inside input-dir')
    parser.add_argument('--kind', choices=('policy', 'replay'), required=True)
    parser.add_argument('--record', choices=('full', 'actions'), default='full')
    parser.add_argument('--fps', type=float, default=30.)
    args = parser.parse_args()
    print(json.dumps(export_run(args.input_dir.expanduser(), kind=args.kind, mode=args.record,
                               fps=args.fps, output=args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
