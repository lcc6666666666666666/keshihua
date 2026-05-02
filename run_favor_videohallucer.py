import argparse
import gc
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from favor_utils import (
    DATASETS,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    cal_score,
    decode_generated_outputs,
    ensure_yes_no_prompt,
    generate_text,
    get_model_device,
    load_favor_model,
    move_inputs_to_device,
    prepare_qwen_video_inputs,
    read_json,
    strip_model_inputs,
    write_json,
)


def run_subset(
    model: Any,
    processor: Any,
    qa_path: Path,
    qa_type: str,
    video_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, float]:
    output_path = output_dir / f"{qa_type}_predictions.json"
    if args.skip_existing and output_path.exists():
        results = read_json(output_path)
        return cal_score(results)

    paired_qas: List[Dict[str, Any]] = read_json(qa_path)
    if args.limit_per_type is not None:
        paired_qas = paired_qas[: args.limit_per_type]

    device = get_model_device(model)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, qa_dct in enumerate(tqdm(paired_qas, desc=f"{qa_type}")):
        for side in ("basic", "hallucination"):
            qa = qa_dct[side]
            question = ensure_yes_no_prompt(qa["question"])
            video_path = video_dir / qa["video"]

            if not video_path.exists():
                qa["predict"] = "Video not found"
                continue

            try:
                inputs, _ = prepare_qwen_video_inputs(
                    processor=processor,
                    video_path=str(video_path),
                    question=question,
                    num_frames=args.num_frames,
                    fps=args.fps,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                    video_decode_backend=args.video_decode_backend,
                )
                inputs = strip_model_inputs(inputs)
                inputs = move_inputs_to_device(inputs, device)
                prediction, _, generated_ids = generate_text(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                )
                decoded = decode_generated_outputs(processor, generated_ids)
                qa["predict"] = decoded["text"]
                qa["predict_full"] = decoded["full_text"]
                qa["extracted_answer"] = decoded["extracted_answer"]
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                qa["predict"] = f"ERROR: {type(exc).__name__}: {exc}"
                qa["predict_full"] = qa["predict"]
                qa["extracted_answer"] = ""

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            write_json(output_path, paired_qas)

    write_json(output_path, paired_qas)
    return cal_score(paired_qas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FAVOR/Qwen3-VL on VideoHallucer.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Local FAVOR/Qwen3-VL model path.",
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="VideoHallucer root directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for prediction JSON files.")
    parser.add_argument(
        "--eval_types",
        nargs="+",
        default=["obj_rel", "temporal", "semantic", "fact", "nonfact"],
        choices=list(DATASETS.keys()),
        help="VideoHallucer subsets to run.",
    )
    parser.add_argument("--limit_per_type", type=int, default=None, help="Debug limit per subset.")
    parser.add_argument("--skip_existing", action="store_true", help="Reuse existing subset prediction files.")
    parser.add_argument("--continue_on_error", action="store_true", help="Record per-case errors and continue.")
    parser.add_argument("--save_every", type=int, default=20, help="Checkpoint predictions every N QA pairs.")

    parser.add_argument("--num_frames", type=int, default=16, help="Number of raw frames sampled by Qwen3-VL.")
    parser.add_argument("--fps", type=float, default=None, help="Alternative video sampling rate; mutually exclusive with num_frames.")
    parser.add_argument("--video_decode_backend", choices=["opencv", "processor"], default="opencv",
                        help="Use opencv to pre-sample frames and avoid ffmpeg/torchvision decoding errors.")
    parser.add_argument("--min_pixels", type=int, default=None, help="Optional processor shortest_edge pixel budget.")
    parser.add_argument("--max_pixels", type=int, default=None, help="Optional processor longest_edge pixel budget.")

    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Optional HF attention backend, e.g. sdpa or flash_attention_2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_frames is not None and args.fps is not None:
        raise ValueError("--num_frames and --fps are mutually exclusive.")

    model, processor = load_favor_model(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    all_scores: Dict[str, Dict[str, float]] = {}
    for qa_type in args.eval_types:
        spec = DATASETS[qa_type]
        qa_path = data_dir / spec["json_path"]
        video_dir = data_dir / spec["video_dir"]
        if not qa_path.exists():
            print(f"[WARN] Missing QA file for {qa_type}: {qa_path}")
            continue
        scores = run_subset(model, processor, qa_path, qa_type, video_dir, output_dir, args)
        all_scores[qa_type] = scores
        print(
            f"{qa_type}: basic={scores['basic_accuracy']:.4f}, "
            f"halluc={scores['halluc_accuracy']:.4f}, pair={scores['accuracy']:.4f}"
        )

    if all_scores:
        avg = {
            key: sum(score[key] for score in all_scores.values()) / len(all_scores)
            for key in ("basic_accuracy", "halluc_accuracy", "accuracy")
        }
        all_scores["all"] = avg
        write_json(output_dir / "favor_evaluation_results.json", all_scores)
        print(
            f"all: basic={avg['basic_accuracy']:.4f}, "
            f"halluc={avg['halluc_accuracy']:.4f}, pair={avg['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
