#!/usr/bin/env python3
"""
Build a BirdNET V2.4 NCNN model from the trusted source ONNX using PNNX.

This script keeps the provenance chain explicit:
  source_models/BirdNET_V2.4.onnx
    -> temporary split CNN-only ONNX
    -> birdnet_cnn_only.param/bin

The resulting NCNN artifact matches the source ONNX logits closely enough for
BirdNET-Go parity checks, unlike the earlier onnx2ncnn-only conversion path.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ONNX = ROOT / "source_models" / "BirdNET_V2.4.onnx"
SPLIT_SCRIPT = ROOT / "source_models" / "split_birdnet_v24.py"
INPUT_SHAPE = "[1,2,96,511]"


def resolve_pnnx(explicit: str | None) -> str:
    if explicit:
        return explicit
    resolved = shutil.which("pnnx")
    if resolved:
        return resolved
    raise SystemExit("pnnx not found in PATH; pass --pnnx /path/to/pnnx")


def resolve_python(explicit: str | None, pnnx_path: str) -> str:
    if explicit:
        return explicit

    pnnx_bin_dir = Path(pnnx_path).resolve().parent
    for candidate_name in ("python", "python3"):
        candidate = pnnx_bin_dir / candidate_name
        if candidate.exists():
            return str(candidate)

    return sys.executable


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-onnx",
        default=str(SOURCE_ONNX),
        help="Path to the trusted BirdNET_V2.4.onnx source model",
    )
    parser.add_argument(
        "--split-onnx",
        default=None,
        help="Optional path for the generated split CNN-only ONNX (defaults to a temporary file)",
    )
    parser.add_argument(
        "--output-dir",
        default="build/ncnn-v24",
        help="Directory to receive birdnet_cnn_only.param/bin",
    )
    parser.add_argument(
        "--pnnx",
        default=None,
        help="Path to the pnnx executable (defaults to PATH lookup)",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter to use for split_birdnet_v24.py (defaults to the interpreter beside pnnx)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_onnx = Path(args.source_onnx).resolve()
    output_dir = Path(args.output_dir).resolve()
    pnnx = resolve_pnnx(args.pnnx)
    python_exe = resolve_python(args.python, pnnx)

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="birdnet-pnnx-") as tmpdir:
        tmp_path = Path(tmpdir)
        split_onnx = (
            Path(args.split_onnx).resolve()
            if args.split_onnx
            else tmp_path / "BirdNET_V2.4.cnn_only.onnx"
        )
        split_onnx.parent.mkdir(parents=True, exist_ok=True)

        run(
            [
                python_exe,
                str(SPLIT_SCRIPT),
                "--input",
                str(source_onnx),
                "--output",
                str(split_onnx),
            ],
            cwd=ROOT,
        )

        model_stem = tmp_path / "birdnet_cnn_only"

        run(
            [
                pnnx,
                str(split_onnx),
                f"inputshape={INPUT_SHAPE}",
                f"pnnxparam={model_stem}.pnnx.param",
                f"pnnxbin={model_stem}.pnnx.bin",
                f"pnnxpy={model_stem}_pnnx.py",
                f"pnnxonnx={model_stem}.pnnx.onnx",
                f"ncnnparam={model_stem}.ncnn.param",
                f"ncnnbin={model_stem}.ncnn.bin",
                f"ncnnpy={model_stem}_ncnn.py",
            ],
            cwd=ROOT,
        )

        shutil.copy2(model_stem.with_suffix(".ncnn.param"), output_dir / "birdnet_cnn_only.param")
        shutil.copy2(model_stem.with_suffix(".ncnn.bin"), output_dir / "birdnet_cnn_only.bin")

    print(f"wrote {output_dir / 'birdnet_cnn_only.param'}")
    print(f"wrote {output_dir / 'birdnet_cnn_only.bin'}")
    print("next step: validate with scripts/compare-soundscape.sh before creating birdnet-go.ncnn-validated")


if __name__ == "__main__":
    main()
