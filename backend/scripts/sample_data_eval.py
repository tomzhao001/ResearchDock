from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.database import SessionLocal
from app.evals.sample_data import run_sample_data_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample-data benchmark evaluation.")
    parser.add_argument("--mode", choices=["retrieval", "e2e", "both"], default="both")
    parser.add_argument("--subset", choices=["full", "smoke"], default="full")
    parser.add_argument("--judge-mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_sample_data_evaluation(
        SessionLocal,
        mode=args.mode,
        subset=args.subset,
        judge_mode=args.judge_mode,
    )
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
