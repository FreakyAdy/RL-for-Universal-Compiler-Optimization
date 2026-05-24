#!/usr/bin/env python3
"""External opt driver: applies RL-UCO policy with -O3 fallback.

Usage:
  python infra/inference/opt_driver.py --ir input.ll --checkpoint checkpoints/best.pt -o output.ll
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_uco.rl.inference import InferenceEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RL-UCO opt driver")
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--isa", default="x86_64_v3")
    parser.add_argument("--fallback", default="O3")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    engine = InferenceEngine(args.checkpoint)
    ok = engine.optimize_ir(args.ir, args.output, isa=args.isa, fallback=args.fallback)
    if not ok:
        print("Optimization failed; fallback may have been used.", file=sys.stderr)
        return 2
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
