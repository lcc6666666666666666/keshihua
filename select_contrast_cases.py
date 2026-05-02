import argparse
from pathlib import Path
from typing import Any, Dict, List

from favor_utils import DATASETS, DEFAULT_DATA_DIR, qa_answer_hit, read_json, write_json


def load_predictions(pred_dir: Path, eval_type: str) -> List[Dict[str, Any]]:
    path = pred_dir / f"{eval_type}_predictions.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def make_record(
    favor_item: Dict[str, Any],
    instruct_item: Dict[str, Any],
    thinking_item: Dict[str, Any],
    eval_type: str,
    index: int,
    side: str,
    data_dir: Path,
) -> Dict[str, Any]:
    qa = favor_item[side]
    return {
        "eval_type": eval_type,
        "index": index,
        "side": side,
        "video": qa.get("video"),
        "video_path": str(data_dir / DATASETS[eval_type]["video_dir"] / qa["video"]),
        "question": qa.get("question"),
        "answer": qa.get("answer"),
        "predictions": {
            "favor": favor_item[side].get("predict"),
            "instruct": instruct_item[side].get("predict"),
            "thinking": thinking_item[side].get("predict"),
        },
        "predictions_full": {
            "favor": favor_item[side].get("predict_full", favor_item[side].get("predict")),
            "instruct": instruct_item[side].get("predict_full", instruct_item[side].get("predict")),
            "thinking": thinking_item[side].get("predict_full", thinking_item[side].get("predict")),
        },
        "extracted_answers": {
            "favor": favor_item[side].get("extracted_answer"),
            "instruct": instruct_item[side].get("extracted_answer"),
            "thinking": thinking_item[side].get("extracted_answer"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select cases where Qwen3-VL Instruct/Thinking are wrong and FAVOR is correct."
    )
    parser.add_argument("--favor_pred_dir", type=str, required=True)
    parser.add_argument("--instruct_pred_dir", type=str, required=True)
    parser.add_argument("--thinking_pred_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--eval_types",
        nargs="+",
        default=["temporal", "obj_rel", "semantic", "fact", "nonfact"],
        choices=list(DATASETS.keys()),
    )
    parser.add_argument("--side", choices=["basic", "hallucination"], default="hallucination")
    parser.add_argument("--max_cases", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    favor_dir = Path(args.favor_pred_dir)
    instruct_dir = Path(args.instruct_pred_dir)
    thinking_dir = Path(args.thinking_pred_dir)

    selected: List[Dict[str, Any]] = []
    for eval_type in args.eval_types:
        try:
            favor_results = load_predictions(favor_dir, eval_type)
            instruct_results = load_predictions(instruct_dir, eval_type)
            thinking_results = load_predictions(thinking_dir, eval_type)
        except FileNotFoundError as exc:
            print(f"[WARN] missing {eval_type} prediction file: {exc}")
            continue

        count = min(len(favor_results), len(instruct_results), len(thinking_results))
        for idx in range(count):
            favor_item = favor_results[idx]
            instruct_item = instruct_results[idx]
            thinking_item = thinking_results[idx]

            favor_ok = qa_answer_hit(favor_item[args.side])
            instruct_ok = qa_answer_hit(instruct_item[args.side])
            thinking_ok = qa_answer_hit(thinking_item[args.side])

            if favor_ok and not instruct_ok and not thinking_ok:
                selected.append(
                    make_record(
                        favor_item=favor_item,
                        instruct_item=instruct_item,
                        thinking_item=thinking_item,
                        eval_type=eval_type,
                        index=idx,
                        side=args.side,
                        data_dir=data_dir,
                    )
                )
                if len(selected) >= args.max_cases:
                    write_json(args.output, selected)
                    print(f"selected {len(selected)} contrast cases -> {args.output}")
                    return

    write_json(args.output, selected)
    print(f"selected {len(selected)} contrast cases -> {args.output}")


if __name__ == "__main__":
    main()
