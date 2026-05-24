#!/usr/bin/env python3
"""Generate large synthetic corpus (e.g. 10k functions) for pipeline testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_uco.corpus.extract import _synthetic_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "corpus" / "synthetic_10k")
    parser.add_argument("--count", type=int, default=10_000)
    args = parser.parse_args()
    records = _synthetic_corpus(args.output, count=args.count)
    print(f"Generated {len(records)} functions at {args.output}")


if __name__ == "__main__":
    main()
