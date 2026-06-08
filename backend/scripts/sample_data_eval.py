from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from queue import Empty

from app.database import SessionLocal
from app.evals.sample_data import run_sample_data_evaluation

logger = logging.getLogger(__name__)
_DEFAULT_SINGLE_RUN_TIMEOUT_SECONDS = 600
_REPORTS_DIR = Path("reports")


def _resolve_output_path(output: Path) -> Path:
    """Relative paths without a parent dir go under reports/ (not backend root)."""
    if output.is_absolute():
        return output
    if output.parent == Path("."):
        return _REPORTS_DIR / output.name
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample-data benchmark evaluation.")
    parser.add_argument("--mode", choices=["retrieval", "e2e", "both"], default="both")
    parser.add_argument("--subset", choices=["full", "smoke"], default="full")
    parser.add_argument("--judge-mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--question-id", type=str, default=None)
    parser.add_argument(
        "--question-id-file",
        type=Path,
        default=None,
        help="JSON file containing a list of question ids to evaluate as a slice.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Single-question run timeout in seconds. Defaults to 600 when --question-id is set.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _run_evaluation(
    *,
    mode: str,
    subset: str,
    judge_mode: str,
    question_id: str | None,
    question_ids: list[str] | None,
) -> dict:
    return run_sample_data_evaluation(
        SessionLocal,
        mode=mode,
        subset=subset,
        judge_mode=judge_mode,
        question_id=question_id,
        question_ids=question_ids,
    )


def _single_run_timeout_seconds(args: argparse.Namespace) -> int | None:
    if args.question_id is None:
        return None
    if args.timeout_seconds is None:
        return _DEFAULT_SINGLE_RUN_TIMEOUT_SECONDS
    if args.timeout_seconds <= 0:
        return None
    return args.timeout_seconds


def _eval_worker(
    mode: str,
    subset: str,
    judge_mode: str,
    question_id: str | None,
    question_ids: list[str] | None,
    result_path: str,
    result_queue,
) -> None:
    configure_logging()
    try:
        report = _run_evaluation(
            mode=mode,
            subset=subset,
            judge_mode=judge_mode,
            question_id=question_id,
            question_ids=question_ids,
        )
        Path(result_path).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        result_queue.put(
            {
                "ok": True,
                "result_path": result_path,
            }
        )
    except Exception:
        result_queue.put({"ok": False, "error": traceback.format_exc()})


def _run_single_with_timeout(
    *,
    mode: str,
    subset: str,
    judge_mode: str,
    question_id: str,
    question_ids: list[str] | None,
    timeout_seconds: int | None,
) -> dict:
    if timeout_seconds is None:
        return _run_evaluation(
            mode=mode,
            subset=subset,
            judge_mode=judge_mode,
            question_id=question_id,
            question_ids=question_ids,
        )

    ctx = get_context("spawn")
    result_queue = ctx.Queue()
    with tempfile.NamedTemporaryFile(prefix="sample-data-eval-", suffix=".json", delete=False) as temp_file:
        result_path = temp_file.name
    process = ctx.Process(
        target=_eval_worker,
        args=(mode, subset, judge_mode, question_id, question_ids, result_path, result_queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        logger.error(
            "Single-question evaluation timed out: question_id=%s timeout=%ss",
            question_id,
            timeout_seconds,
        )
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        result_queue.close()
        if os.path.exists(result_path):
            os.unlink(result_path)
        raise TimeoutError(
            f"Single-question evaluation timed out after {timeout_seconds}s for question_id={question_id}"
        )

    try:
        payload = result_queue.get(timeout=1)
    except Empty:
        payload = None
    result_queue.close()
    if not isinstance(payload, dict):
        if os.path.exists(result_path):
            os.unlink(result_path)
        raise RuntimeError(f"Single-question evaluation exited without a result (exit_code={process.exitcode})")
    if not payload.get("ok"):
        if os.path.exists(result_path):
            os.unlink(result_path)
        raise RuntimeError(str(payload.get("error") or "Single-question evaluation failed"))
    loaded_result_path = str(payload.get("result_path") or result_path)
    try:
        return json.loads(Path(loaded_result_path).read_text(encoding="utf-8"))
    finally:
        if os.path.exists(loaded_result_path):
            os.unlink(loaded_result_path)


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.question_id is not None and args.question_id_file is not None:
        raise SystemExit("--question-id 和 --question-id-file 不能同时使用")
    question_ids = None
    if args.question_id_file is not None:
        question_ids = json.loads(args.question_id_file.read_text(encoding="utf-8"))
        if not isinstance(question_ids, list):
            raise SystemExit("--question-id-file 必须是 JSON 数组")
        question_ids = [str(item) for item in question_ids]
    started_at = time.perf_counter()
    timeout_seconds = _single_run_timeout_seconds(args)
    logger.info(
        "Starting sample-data evaluation: mode=%s subset=%s judge_mode=%s question_id=%s question_ids=%s timeout_seconds=%s output=%s",
        args.mode,
        args.subset,
        args.judge_mode,
        args.question_id,
        question_ids,
        timeout_seconds,
        args.output,
    )
    if args.timeout_seconds is not None and args.question_id is None:
        logger.warning("--timeout-seconds is ignored unless --question-id is set")
    if args.question_id is not None:
        report = _run_single_with_timeout(
            mode=args.mode,
            subset=args.subset,
            judge_mode=args.judge_mode,
            question_id=args.question_id,
            question_ids=question_ids,
            timeout_seconds=timeout_seconds,
        )
    else:
        report = _run_evaluation(
            mode=args.mode,
            subset=args.subset,
            judge_mode=args.judge_mode,
            question_id=None,
            question_ids=question_ids,
        )
    logger.info("Evaluation finished in %.2fs", time.perf_counter() - started_at)
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        output_path = _resolve_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        logger.info("Evaluation report written to %s", output_path)
    try:
        print(content)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(content.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
