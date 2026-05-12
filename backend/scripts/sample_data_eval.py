from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from app.database import SessionLocal
from app.evals.sample_data import run_sample_data_evaluation

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample-data benchmark evaluation.")
    parser.add_argument("--mode", choices=["retrieval", "e2e", "both"], default="both")
    parser.add_argument("--subset", choices=["full", "smoke"], default="full")
    parser.add_argument("--judge-mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    started_at = time.perf_counter()
    logger.info(
        "Starting sample-data evaluation: mode=%s subset=%s judge_mode=%s output=%s",
        args.mode,
        args.subset,
        args.judge_mode,
        args.output,
    )
    report = run_sample_data_evaluation(
        SessionLocal,
        mode=args.mode,
        subset=args.subset,
        judge_mode=args.judge_mode,
    )
    logger.info("Evaluation finished in %.2fs", time.perf_counter() - started_at)
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        logger.info("Evaluation report written to %s", args.output)
    print(content)


if __name__ == "__main__":
    main()
