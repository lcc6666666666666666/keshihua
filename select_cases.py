import argparse
from pathlib import Path
from typing import Any, Dict, List

from favor_utils import DATASETS, DEFAULT_DATA_DIR, qa_answer_hit, read_json, write_json


def case_passes(item: Dict[str, Any], mode: str) -> bool:
    basic = item.get("basic", {})
    halluc = item.get("hallucination", {})
    basic_ok = qa_answer_hit(basic)
    halluc_ok = qa_answer_hit(halluc)

    if mode == "both_correct":
        return basic_ok and halluc_ok
    if mode == "hallucination_correct":
        return halluc_ok
    if mode == "basic_correct":
        return basic_ok
    if mode == "any_correct":
        return basic_ok or halluc_ok
    if mode == "any_wrong":
        return not (basic_ok and halluc_ok)
    raise ValueError(f"Unknown selection mode: {mode}")


def make_record(
    item: Dict[str, Any],
    eval_type: str,
    index: int,
    side: str,
    data_dir: Path | None,
) -> Dict[str, Any]:
    qa = item[side]
    record = {
        "eval_type": eval_type,
        "index": index,
        "side": side,
        "video": qa.get("video"),
        "question": qa.get("question"),
        "answer": qa.get("answer"),
        "predict": qa.get("predict"),
        "predict_full": qa.get("predict_full", qa.get("predict")),
        "extracted_answer": qa.get("extracted_answer"),
    }
    if data_dir is not None and eval_type in DATASETS and qa.get("video"):
        record["video_path"] = str(data_dir / DATASETS[eval_type]["video_dir"] / qa["video"])
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select VideoHallucer cases from FAVOR prediction files.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing *_predictions.json files.")
    parser.add_argument("--output", type=str, required=True, help="Output selected_cases.json path.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="VideoHallucer root to add absolute video_path.")
    parser.add_argument(
        "--eval_types",
        nargs="+",
        default=["obj_rel", "temporal", "semantic", "fact", "nonfact"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--mode",
        choices=["both_correct", "hallucination_correct", "basic_correct", "any_correct", "any_wrong"],
        default="both_correct",
    )
    parser.add_argument(
        "--side",
        choices=["basic", "hallucination", "both"],
        default="hallucination",
        help="Which QA side to export for visualization.",
    )
    parser.add_argument("--max_cases", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None

    selected: List[Dict[str, Any]] = []
    for eval_type in args.eval_types:
        pred_path = pred_dir / f"{eval_type}_predictions.json"
        if not pred_path.exists():
            print(f"[WARN] missing prediction file: {pred_path}")
            continue
        results = read_json(pred_path)
        for idx, item in enumerate(results):
            if not case_passes(item, args.mode):
                continue
            sides = ["basic", "hallucination"] if args.side == "both" else [args.side]
            for side in sides:
                selected.append(make_record(item, eval_type, idx, side, data_dir))
                if len(selected) >= args.max_cases:
                    write_json(args.output, selected)
                    print(f"selected {len(selected)} cases -> {args.output}")
                    return

    write_json(args.output, selected)
    print(f"selected {len(selected)} cases -> {args.output}")


if __name__ == "__main__":
    main()
