import argparse
import gc
import string
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from favor_utils import (
    DATASETS,
    DEFAULT_DATA_DIR,
    DEFAULT_INSTRUCT_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_THINKING_MODEL_PATH,
    attention_metrics,
    build_frame_token_map,
    decode_generated_outputs,
    ensure_yes_no_prompt,
    find_decoder_layers,
    frame_indices_from_metadata,
    generate_text,
    get_model_device,
    get_video_token_id,
    get_video_token_positions,
    load_favor_model,
    load_video_frames,
    move_inputs_to_device,
    prepare_qwen_video_inputs,
    read_json,
    strip_model_inputs,
    write_json,
)
from visualize_frame_attention import (
    DecoderAttentionCollector,
    heatmap_list_to_array,
    reduce_attention_records,
    replay_generated_tokens,
    select_final_answer_replay_tokens,
    token_scores_to_frame_outputs,
)


def load_case(args: argparse.Namespace) -> Dict[str, Any]:
    if args.case_file:
        cases = read_json(args.case_file)
        case = cases[args.case_index]
    else:
        if not args.video_path or not args.question:
            raise ValueError("Provide --case_file or both --video_path and --question.")
        case = {
            "eval_type": "manual",
            "index": 0,
            "side": "manual",
            "video_path": args.video_path,
            "question": args.question,
            "answer": args.answer or "",
        }

    return resolve_case_video_path(case, args)


def resolve_case_video_path(case: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if "video_path" not in case:
        if not args.data_dir:
            raise ValueError("Case lacks video_path; pass --data_dir.")
        case["video_path"] = str(Path(args.data_dir) / DATASETS[case["eval_type"]]["video_dir"] / case["video"])
    return case


def load_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.all_cases:
        return [load_case(args)]
    if not args.case_file:
        raise ValueError("--all_cases requires --case_file.")

    cases = read_json(args.case_file)
    start = max(0, args.start_case_index)
    end = len(cases) if args.end_case_index is None else min(len(cases), args.end_case_index)
    if end <= start:
        raise ValueError(f"No cases selected: start={start}, end={end}, total={len(cases)}")
    return [resolve_case_video_path(dict(case), args) for case in cases[start:end]]


def choose_layers(total_layers: int, layers: Optional[Sequence[int]], last_n_layers: int) -> List[int]:
    if layers:
        requested = list(layers)
    else:
        last_n = max(1, min(last_n_layers, total_layers))
        requested = list(range(total_layers - last_n, total_layers))
    valid = [idx for idx in requested if 0 <= idx < total_layers]
    if not valid:
        raise ValueError(f"No valid layer indices from {requested}; model has {total_layers} layers.")
    if valid != requested:
        print(f"[WARN] Dropped out-of-range layers. Requested={requested}, valid={valid}, total_layers={total_layers}")
    return valid


def compute_attention_for_model(
    label: str,
    model_path: str,
    case: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print(f"\n[{label}] loading {model_path}")
    model, processor = load_favor_model(
        model_path=model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    try:
        return compute_attention_with_loaded_model(label, model_path, model, processor, case, args)
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_attention_with_loaded_model(
    label: str,
    model_path: str,
    model: Any,
    processor: Any,
    case: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = get_model_device(model)
    question = case["question"]
    question_for_model = ensure_yes_no_prompt(question) if args.append_yes_no_prompt else question

    raw_inputs, _ = prepare_qwen_video_inputs(
        processor=processor,
        video_path=case["video_path"],
        question=question_for_model,
        num_frames=args.num_frames,
        fps=args.fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        return_metadata=True,
        video_decode_backend=args.video_decode_backend,
    )
    metadata = raw_inputs.get("video_metadata")
    inputs = strip_model_inputs(raw_inputs)
    inputs = move_inputs_to_device(inputs, device)

    video_token_id = get_video_token_id(model, processor)
    video_positions = get_video_token_positions(inputs, video_token_id)
    frame_map = build_frame_token_map(inputs, processor, video_positions)
    decoder_path, decoder_layers = find_decoder_layers(model)
    layer_ids = choose_layers(len(decoder_layers), args.layers, args.last_n_layers)
    print(f"[{label}] decoder_layers={decoder_path}, selected_layers={layer_ids}")
    print(f"[{label}] video_tokens={frame_map['total_video_tokens']}, grid_thw={frame_map['grid_thw']}")
    metadata_item = metadata[0] if isinstance(metadata, (list, tuple)) and metadata else metadata
    if metadata_item is not None:
        metadata_get = metadata_item.get if isinstance(metadata_item, dict) else lambda key, default=None: getattr(metadata_item, key, default)
        print(
            f"[{label}] sampled_frames={metadata_get('sampled_num_frames', 'unknown')}, "
            f"requested_frames={metadata_get('requested_num_frames', 'unknown')}"
        )

    prediction, _, generated_ids = generate_text(
        model=model,
        processor=processor,
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )
    decoded = decode_generated_outputs(processor, generated_ids)
    prediction = decoded["text"]
    replay_ids, collect_steps, replay_selection = select_final_answer_replay_tokens(
        processor,
        generated_ids,
        decoded["extracted_answer"],
    )
    if replay_ids.numel() == 0:
        raise RuntimeError(f"{label} produced no generated tokens to replay.")
    print(f"[{label}] selected answer token={replay_selection['selected_token_text']!r} step={collect_steps[0]}")

    collector = DecoderAttentionCollector(decoder_layers, layer_ids, video_positions)
    collector.install()
    try:
        replay_generated_tokens(model, inputs, replay_ids, collector, collect_token_indices=collect_steps)
    finally:
        collector.restore()

    token_scores = reduce_attention_records(
        collector.records,
        layer_ids=layer_ids,
        total_video_tokens=frame_map["total_video_tokens"],
    )
    ratios, heatmaps, raw_heatmaps = token_scores_to_frame_outputs(
        token_scores,
        frame_map,
        norm_mode=args.heatmap_norm,
        ratio_power=args.ratio_power,
    )
    frame_indices = frame_indices_from_metadata(
        metadata,
        num_bins=frame_map["num_bins"],
        temporal_patch_size=frame_map["temporal_patch_size"],
    )
    frames = load_video_frames(case["video_path"], frame_indices=frame_indices, count=frame_map["num_bins"])

    result = {
        "label": label,
        "model_path": model_path,
        "prediction": prediction,
        "prediction_full": decoded["full_text"],
        "extracted_answer": decoded["extracted_answer"],
        "layers": layer_ids,
        "num_replayed_tokens": int(replay_ids.shape[1]),
        "num_collected_tokens": len(collect_steps),
        **replay_selection,
        "frame_ratios": ratios,
        "metrics": attention_metrics(ratios),
        "heatmap_norm": args.heatmap_norm,
        "ratio_power": args.ratio_power,
        "frame_map": {key: value for key, value in frame_map.items() if key != "frame_slices"},
        "frames": frames,
        "heatmaps": heatmaps,
        "raw_heatmaps": raw_heatmaps,
        "token_scores": token_scores,
    }

    del inputs, raw_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def save_comparison_figure(
    output_path: Path,
    case: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    frame_cols: int,
    overlay_alpha: float,
) -> None:
    import matplotlib.pyplot as plt

    max_frames = max(len(result["frame_ratios"]) for result in results)
    frame_cols = max(1, min(frame_cols, max_frames))
    frame_rows = int(np.ceil(max_frames / frame_cols))
    total_cols = frame_cols * len(results)

    fig_w = 2.15 * total_cols
    fig_h = 2.45 * frame_rows + 1.9
    fig, axes = plt.subplots(frame_rows, total_cols, figsize=(fig_w, fig_h), squeeze=False)

    for model_idx, result in enumerate(results):
        offset = model_idx * frame_cols
        ratios = result["frame_ratios"]
        heatmaps = result["heatmaps"]
        frames = result["frames"]
        for frame_idx in range(frame_rows * frame_cols):
            ax = axes[frame_idx // frame_cols][offset + frame_idx % frame_cols]
            ax.axis("off")
            if frame_idx >= len(ratios):
                continue
            frame = frames[frame_idx] if frame_idx < len(frames) else np.zeros((224, 224, 3), dtype=np.uint8)
            ax.imshow(frame)
            heatmap = np.clip(np.asarray(heatmaps[frame_idx]), 0.0, 1.0)
            ax.imshow(
                heatmap,
                cmap="jet",
                alpha=overlay_alpha,
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
                extent=(0, frame.shape[1], frame.shape[0], 0),
            )
            ax.set_title(f"Frame {frame_idx}\nRatio: {ratios[frame_idx] * 100:.2f}%", fontsize=8, fontweight="bold")

        block_center = (offset + frame_cols / 2) / total_cols
        letter = string.ascii_lowercase[model_idx]
        fig.text(
            block_center,
            0.965,
            f"({letter}) {result['label']}",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        pred = result.get("extracted_answer") or result["prediction"] or "<empty>"
        metrics = result["metrics"]
        footer = (
            f"A: {pred}    GT: {case.get('answer', '')}\n"
            f"NormEnt: {metrics['normalized_entropy']:.3f}    CV: {metrics['coefficient_of_variation']:.3f}"
        )
        fig.text(block_center, 0.075, footer, ha="center", va="bottom", fontsize=9, fontweight="bold")

    wrapped_q = "\n".join(textwrap.wrap(f"Q: {case['question']}", width=150))
    fig.text(0.5, 0.012, wrapped_q, ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.14, 1, 0.93), w_pad=0.5, h_pad=0.55)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_outputs(output_dir: Path, case: Dict[str, Any], results: Sequence[Dict[str, Any]], frame_cols: int, overlay_alpha: float) -> None:
    stem = f"{case.get('eval_type', 'manual')}_{case.get('index', 0)}_{case.get('side', 'manual')}"
    fig_path = output_dir / f"{stem}_qwen_contrast_attention.png"
    metrics_path = output_dir / f"{stem}_qwen_contrast_metrics.json"
    save_comparison_figure(fig_path, case, results, frame_cols, overlay_alpha)

    metrics_payload = {
        "case": case,
        "heatmap_norm": results[0]["heatmap_norm"] if results else None,
        "ratio_power": results[0]["ratio_power"] if results else None,
        "models": [
            {
                "label": result["label"],
                "model_path": result["model_path"],
                "prediction": result["prediction"],
                "prediction_full": result["prediction_full"],
                "extracted_answer": result["extracted_answer"],
                "layers": result["layers"],
                "num_replayed_tokens": result["num_replayed_tokens"],
                "num_collected_tokens": result["num_collected_tokens"],
                "replay_token_mode": result["replay_token_mode"],
                "total_response_tokens": result["total_response_tokens"],
                "total_filtered_response_tokens": result["total_filtered_response_tokens"],
                "selected_replay_steps": result["selected_replay_steps"],
                "selected_token_id": result["selected_token_id"],
                "selected_token_text": result["selected_token_text"],
                "selected_answer": result["selected_answer"],
                "frame_ratios": result["frame_ratios"],
                "metrics": result["metrics"],
                "heatmap_norm": result["heatmap_norm"],
                "ratio_power": result["ratio_power"],
                "frame_map": result["frame_map"],
            }
            for result in results
        ],
    }
    write_json(metrics_path, metrics_payload)

    for result in results:
        safe_label = result["label"].lower().replace(" ", "_").replace("-", "_")
        np.savez_compressed(
            output_dir / f"{stem}_{safe_label}_attention.npz",
            token_scores=result["token_scores"],
            ratios=np.asarray(result["frame_ratios"]),
            frame_ratios=np.asarray(result["frame_ratios"]),
            heatmaps=heatmap_list_to_array(result["heatmaps"]),
            raw_heatmaps=heatmap_list_to_array(result["raw_heatmaps"]),
            heatmap_norm=result["heatmap_norm"],
            ratio_power=np.asarray(result["ratio_power"]),
            layers=np.asarray(result["layers"]),
        )

    print(f"\ncomparison figure: {fig_path}")
    print(f"metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a paper-style attention comparison for Qwen3-VL Instruct, Qwen3-VL Thinking, and FAVOR."
    )
    parser.add_argument("--case_file", type=str, default=None, help="JSON from select_contrast_cases.py or select_cases.py.")
    parser.add_argument("--case_index", type=int, default=0)
    parser.add_argument("--all_cases", action="store_true", help="Visualize every case in --case_file.")
    parser.add_argument("--start_case_index", type=int, default=0, help="Inclusive start index used with --all_cases.")
    parser.add_argument("--end_case_index", type=int, default=None, help="Exclusive end index used with --all_cases.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--answer", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--instruct_model_path", type=str, default=DEFAULT_INSTRUCT_MODEL_PATH)
    parser.add_argument("--thinking_model_path", type=str, default=DEFAULT_THINKING_MODEL_PATH)
    parser.add_argument("--favor_model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_order", nargs="+", default=["instruct", "thinking", "favor"], choices=["instruct", "thinking", "favor"])

    parser.add_argument("--append_yes_no_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Exact decoder layers. Default: last layer.")
    parser.add_argument("--last_n_layers", type=int, default=1, help="Default number of final decoder layers to average.")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--video_decode_backend", choices=["opencv", "processor"], default="opencv")
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument("--heatmap_norm", choices=["local", "global", "ratio_scaled"], default="ratio_scaled")
    parser.add_argument("--ratio_power", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_frames is not None and args.fps is not None:
        raise ValueError("--num_frames and --fps are mutually exclusive.")

    cases = load_cases(args)
    model_specs = {
        "instruct": ("Qwen3-VL-Instruct", args.instruct_model_path),
        "thinking": ("Qwen3-VL-Thinking", args.thinking_model_path),
        "favor": ("FAVOR", args.favor_model_path),
    }

    total_cases = len(cases)
    if args.all_cases:
        all_results: List[List[Dict[str, Any]]] = [[] for _ in cases]
        for key in args.model_order:
            label, model_path = model_specs[key]
            print(f"\n[{label}] loading {model_path}")
            model, processor = load_favor_model(
                model_path=model_path,
                dtype=args.dtype,
                device_map=args.device_map,
                attn_implementation=args.attn_implementation,
            )
            try:
                for case_pos, case in enumerate(cases):
                    case_name = case.get(
                        "case_id",
                        f"{case.get('eval_type', 'manual')}_{case.get('index', 0)}_{case.get('side', 'manual')}",
                    )
                    print(f"\n=== {label} | Case {case_pos + 1}/{total_cases}: {case_name} ===")
                    result = compute_attention_with_loaded_model(label, model_path, model, processor, case, args)
                    print(
                        f"[{label}] prediction={result['prediction']} | "
                        f"NormEnt={result['metrics']['normalized_entropy']:.3f} | "
                        f"CV={result['metrics']['coefficient_of_variation']:.3f}"
                    )
                    all_results[case_pos].append(result)
            finally:
                del model, processor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for case_pos, case in enumerate(cases):
            case_name = case.get(
                "case_id",
                f"{case.get('eval_type', 'manual')}_{case.get('index', 0)}_{case.get('side', 'manual')}",
            )
            print(f"\n=== Saving Case {case_pos + 1}/{total_cases}: {case_name} ===")
            save_outputs(Path(args.output_dir), case, all_results[case_pos], args.cols, args.overlay_alpha)
    else:
        case = cases[0]
        case_name = case.get(
            "case_id",
            f"{case.get('eval_type', 'manual')}_{case.get('index', 0)}_{case.get('side', 'manual')}",
        )
        print(f"\n=== Case 1/1: {case_name} ===")
        results: List[Dict[str, Any]] = []
        for key in args.model_order:
            label, model_path = model_specs[key]
            result = compute_attention_for_model(label, model_path, case, args)
            print(
                f"[{label}] prediction={result['prediction']} | "
                f"NormEnt={result['metrics']['normalized_entropy']:.3f} | "
                f"CV={result['metrics']['coefficient_of_variation']:.3f}"
            )
            results.append(result)

        save_outputs(Path(args.output_dir), case, results, args.cols, args.overlay_alpha)


if __name__ == "__main__":
    main()
