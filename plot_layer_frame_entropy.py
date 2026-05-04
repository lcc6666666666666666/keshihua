import argparse
import gc
import math
import os
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from favor_utils import (
    DATASETS,
    DEFAULT_DATA_DIR,
    build_frame_token_map,
    decode_generated_outputs,
    ensure_yes_no_prompt,
    find_decoder_layers,
    generate_text,
    get_model_device,
    get_video_token_id,
    get_video_token_positions,
    load_favor_model,
    move_inputs_to_device,
    prepare_qwen_video_inputs,
    read_json,
    strip_model_inputs,
    write_json,
)
from visualize_frame_attention import (
    count_collect_tokens,
    limit_collect_mask,
    non_special_token_mask,
    replay_generated_tokens,
)


DEFAULT_FAVOR_MODEL_PATH = os.environ.get("FAVOR_MODEL_PATH", "/data1/liugengyuan/models/FAVOR0.5-8B")
DEFAULT_VIDEO_R1_MODEL_PATH = os.environ.get("VIDEO_R1_MODEL_PATH", "/data1/liugengyuan/models/Video-R1")
DEFAULT_VIDEO_KTR_MODEL_PATH = os.environ.get("VIDEO_KTR_MODEL_PATH", "/data1/liugengyuan/models/Video-KTR")
DEFAULT_ONETHINKER_MODEL_PATH = os.environ.get("ONETHINKER_MODEL_PATH", "/data1/liugengyuan/models/OneThinker")
DEFAULT_AUTO_VIDEO_R1_MODEL_PATH = os.environ.get("AUTO_VIDEO_R1_MODEL_PATH", "/data1/liugengyuan/models/Auto-Video-R1")
DEFAULT_QWEN3_VL_INSTRUCT_MODEL_PATH = os.environ.get(
    "QWEN3_VL_INSTRUCT_MODEL_PATH",
    "/data1/liugengyuan/models/Qwen3-VL-8B-Instruct",
)
DEFAULT_QWEN3_VL_THINKING_MODEL_PATH = os.environ.get(
    "QWEN3_VL_THINKING_MODEL_PATH",
    "/data1/liugengyuan/models/Qwen3-VL-8B-Thinking",
)


class LayerFrameEntropyCollector:
    def __init__(
        self,
        layers: Sequence[Any],
        layer_ids: Sequence[int],
        video_positions: torch.Tensor,
        frame_slices: Sequence[Tuple[int, int, int]],
        eps: float = 1e-12,
    ) -> None:
        self.layers = layers
        self.layer_ids = list(layer_ids)
        self.video_positions = video_positions
        self.frame_slices = list(frame_slices)
        self.eps = eps
        self.enabled = False
        self._originals: List[Tuple[Any, Any]] = []
        self.records: Dict[int, List[Dict[str, float]]] = {idx: [] for idx in self.layer_ids}

    def install(self) -> None:
        for layer_idx in self.layer_ids:
            module = self.layers[layer_idx].self_attn
            original_forward = module.forward
            collector = self

            def wrapped_forward(module_self, *args, __orig=original_forward, __idx=layer_idx, **kwargs):
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None and args:
                    hidden_states = args[0]
                should_collect = (
                    collector.enabled
                    and torch.is_tensor(hidden_states)
                    and hidden_states.ndim >= 3
                    and hidden_states.shape[1] == 1
                )
                if not should_collect:
                    return __orig(*args, **kwargs)

                config = getattr(module_self, "config", None)
                old_impl = getattr(config, "_attn_implementation", None) if config is not None else None
                if config is not None:
                    config._attn_implementation = "eager"
                try:
                    outputs = __orig(*args, **kwargs)
                finally:
                    if config is not None:
                        config._attn_implementation = old_impl
                collector._store_attention(__idx, outputs)
                return outputs

            module.forward = types.MethodType(wrapped_forward, module)
            self._originals.append((module, original_forward))

    def restore(self) -> None:
        for module, original_forward in self._originals:
            module.forward = original_forward
        self._originals.clear()

    def _store_attention(self, layer_idx: int, outputs: Any) -> None:
        attn = None
        if isinstance(outputs, tuple):
            for item in outputs:
                if torch.is_tensor(item) and item.ndim == 4:
                    attn = item
                    break
        elif hasattr(outputs, "attentions"):
            candidate = outputs.attentions
            if torch.is_tensor(candidate):
                attn = candidate
            elif isinstance(candidate, (tuple, list)):
                for item in candidate:
                    if torch.is_tensor(item) and item.ndim == 4:
                        attn = item
                        break

        if attn is None:
            return

        key_len = attn.shape[-1]
        positions = self.video_positions.to(attn.device)
        positions = positions[positions < key_len]
        if positions.numel() == 0:
            return

        # Mean over heads, then sum tokens inside each video temporal bin/frame.
        values = attn[:, :, -1, positions][0].detach().float()
        video_scores = values.mean(dim=0).clamp_min(0.0)
        if video_scores.numel() == 0:
            return

        frame_scores = []
        for _, start, end in self.frame_slices:
            if start >= video_scores.numel():
                continue
            end = min(end, int(video_scores.numel()))
            if end > start:
                frame_scores.append(video_scores[start:end].sum())
        if len(frame_scores) <= 1:
            return

        frame_scores_t = torch.stack(frame_scores).float()
        score_sum = frame_scores_t.sum()
        if not torch.isfinite(score_sum) or float(score_sum) <= self.eps:
            return

        frame_ratios = frame_scores_t / score_sum
        entropy = -(frame_ratios * torch.log(frame_ratios.clamp_min(self.eps))).sum()
        norm_entropy = entropy / math.log(float(frame_ratios.numel()))
        first_frame_ratio = frame_ratios[0]
        if frame_ratios.numel() > 1:
            other_frame_avg_ratio = frame_ratios[1:].mean()
        else:
            other_frame_avg_ratio = torch.zeros((), dtype=frame_ratios.dtype, device=frame_ratios.device)

        self.records[layer_idx].append(
            {
                "entropy": float(entropy.detach().cpu()),
                "normalized_entropy": float(norm_entropy.detach().cpu()),
                "sink_ratio": float(first_frame_ratio.detach().cpu()),
                "first_frame_ratio": float(first_frame_ratio.detach().cpu()),
                "other_frame_avg_ratio": float(other_frame_avg_ratio.detach().cpu()),
                "max_frame_ratio": float(frame_ratios.max().detach().cpu()),
                "frame_ratio_variance": float(frame_ratios.var(unbiased=False).detach().cpu()),
                "video_attention_mass": float(score_sum.detach().cpu()),
            }
        )

    def layer_means(self) -> Dict[int, Dict[str, float]]:
        output: Dict[int, Dict[str, float]] = {}
        for layer_idx, rows in self.records.items():
            if not rows:
                continue
            keys = rows[0].keys()
            output[layer_idx] = {key: float(np.mean([row[key] for row in rows])) for key in keys}
            output[layer_idx]["num_response_tokens"] = float(len(rows))
        return output


def parse_layer_ids(args: argparse.Namespace, total_layers: int) -> List[int]:
    if args.layers:
        requested = list(args.layers)
    else:
        start = 0 if args.start_layer is None else args.start_layer
        end = total_layers - 1 if args.end_layer is None else args.end_layer
        stride = max(1, args.layer_stride)
        requested = list(range(start, end + 1, stride))

    valid = [idx for idx in requested if 0 <= idx < total_layers]
    if not valid:
        raise ValueError(f"No valid layer indices from {requested}; model has {total_layers} layers.")
    if valid != requested:
        print(f"[WARN] Dropped out-of-range layers. Requested={requested}, valid={valid}, total_layers={total_layers}")
    return valid


def case_to_record(
    item: Dict[str, Any],
    eval_type: str,
    index: int,
    side: str,
    data_dir: Path,
) -> Dict[str, Any]:
    qa = item[side]
    record = {
        "eval_type": eval_type,
        "index": index,
        "side": side,
        "video": qa.get("video"),
        "question": qa.get("question"),
        "answer": qa.get("answer"),
    }
    if qa.get("video"):
        record["video_path"] = str(data_dir / DATASETS[eval_type]["video_dir"] / qa["video"])
    return record


def resolve_case_video_path(case: Dict[str, Any], data_dir: Path) -> Dict[str, Any]:
    case = dict(case)
    if case.get("video_path") and Path(str(case["video_path"])).exists():
        return case
    eval_type = case.get("eval_type")
    video = case.get("video")
    if eval_type in DATASETS and video:
        case["video_path"] = str(data_dir / DATASETS[eval_type]["video_dir"] / video)
    return case


def load_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data_dir = Path(args.data_dir)
    if args.case_file:
        cases = read_json(args.case_file)
        cases = [resolve_case_video_path(case, data_dir) for case in cases]
    else:
        spec = DATASETS[args.eval_type]
        qa_path = data_dir / spec["json_path"]
        paired_qas = read_json(qa_path)
        cases = [
            case_to_record(item, args.eval_type, idx, args.side, data_dir)
            for idx, item in enumerate(paired_qas)
            if args.side in item
        ]

    if args.side_filter != "any":
        cases = [case for case in cases if case.get("side") == args.side_filter]

    start = max(0, args.case_offset)
    end = None if args.num_samples is None else start + max(0, args.num_samples)
    cases = cases[start:end]
    if not cases:
        raise ValueError("No cases selected.")
    return cases


def selected_model_specs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    specs = {
        "favor": ("FAVOR", args.favor_model_path),
        "video_r1": ("Video-R1", args.video_r1_model_path),
        "video_ktr": ("Video-KTR", args.video_ktr_model_path),
        "onethinker": ("OneThinker", args.onethinker_model_path),
        "auto_video_r1": ("Auto-Video-R1", args.auto_video_r1_model_path),
        "instruct": ("Qwen3-VL-Instruct", args.instruct_model_path),
        "thinking": ("Qwen3-VL-Thinking", args.thinking_model_path),
    }
    output = [specs[key] for key in args.model_order]
    missing = [label for label, path in output if not path]
    if missing:
        raise ValueError(f"Missing model paths for: {missing}")
    return output


def compute_case_stats(
    label: str,
    model: Any,
    processor: Any,
    decoder_layers: Sequence[Any],
    layer_ids: Sequence[int],
    case: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = get_model_device(model)
    question = str(case["question"])
    question_for_model = ensure_yes_no_prompt(question) if args.append_yes_no_prompt else question

    raw_inputs, _ = prepare_qwen_video_inputs(
        processor=processor,
        video_path=case["video_path"],
        question=question_for_model,
        num_frames=args.num_frames,
        fps=args.fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        return_metadata=False,
        video_decode_backend=args.video_decode_backend,
    )
    inputs = strip_model_inputs(raw_inputs)
    inputs = move_inputs_to_device(inputs, device)

    video_token_id = get_video_token_id(model, processor)
    video_positions = get_video_token_positions(inputs, video_token_id)
    frame_map = build_frame_token_map(inputs, processor, video_positions)

    _, _, generated_ids = generate_text(
        model=model,
        processor=processor,
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )
    decoded = decode_generated_outputs(processor, generated_ids)
    collect_mask = limit_collect_mask(
        non_special_token_mask(processor, generated_ids),
        args.max_replay_tokens,
        total_tokens=int(generated_ids.shape[1]),
    )
    num_collected_tokens = count_collect_tokens(collect_mask, generated_ids)
    if generated_ids.numel() == 0:
        raise RuntimeError(f"{label} produced no generated tokens to replay.")
    if num_collected_tokens == 0:
        raise RuntimeError(f"{label} produced no non-special generated tokens to collect.")

    collector = LayerFrameEntropyCollector(
        layers=decoder_layers,
        layer_ids=layer_ids,
        video_positions=video_positions,
        frame_slices=frame_map["frame_slices"],
    )
    collector.install()
    try:
        replay_generated_tokens(model, inputs, generated_ids, collector, collect_token_mask=collect_mask)
    finally:
        collector.restore()

    layer_stats = collector.layer_means()
    if not layer_stats:
        raise RuntimeError(f"{label} collected no decode-step attentions.")

    del inputs, raw_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "case_id": case.get("case_id", f"{case.get('eval_type')}_{case.get('index')}_{case.get('side')}"),
        "eval_type": case.get("eval_type"),
        "index": case.get("index"),
        "side": case.get("side"),
        "video": case.get("video"),
        "question": question,
        "answer": case.get("answer"),
        "prediction": decoded["text"],
        "prediction_full": decoded["full_text"],
        "extracted_answer": decoded["extracted_answer"],
        "num_replayed_tokens": int(generated_ids.shape[1]),
        "num_collected_tokens": num_collected_tokens,
        "frame_map": {key: value for key, value in frame_map.items() if key != "frame_slices"},
        "layer_stats": {str(layer): stats for layer, stats in layer_stats.items()},
    }


def aggregate_model_results(
    label: str,
    model_path: str,
    layer_ids: Sequence[int],
    case_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    for layer_idx in layer_ids:
        per_case = [
            case_result["layer_stats"][str(layer_idx)]
            for case_result in case_results
            if str(layer_idx) in case_result["layer_stats"]
        ]
        if not per_case:
            continue
        row: Dict[str, float] = {
            "layer": float(layer_idx),
            "num_cases": float(len(per_case)),
            "num_response_tokens": float(sum(item["num_response_tokens"] for item in per_case)),
        }
        metric_names = [
            "entropy",
            "normalized_entropy",
            "sink_ratio",
            "first_frame_ratio",
            "other_frame_avg_ratio",
            "max_frame_ratio",
            "frame_ratio_variance",
            "video_attention_mass",
        ]
        for metric in metric_names:
            values = np.asarray([item[metric] for item in per_case], dtype=np.float64)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=0))
        rows.append(row)
    return {
        "label": label,
        "model_path": model_path,
        "layers": list(layer_ids),
        "case_results": list(case_results),
        "layer_summary": rows,
    }


def write_csv(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    import csv

    fieldnames = [
        "model",
        "model_path",
        "layer",
        "num_cases",
        "num_response_tokens",
        "mean_entropy",
        "std_entropy",
        "mean_normalized_entropy",
        "std_normalized_entropy",
        "mean_sink_ratio",
        "std_sink_ratio",
        "mean_first_frame_ratio",
        "std_first_frame_ratio",
        "mean_other_frame_avg_ratio",
        "std_other_frame_avg_ratio",
        "mean_max_frame_ratio",
        "std_max_frame_ratio",
        "mean_frame_ratio_variance",
        "std_frame_ratio_variance",
        "mean_video_attention_mass",
        "std_video_attention_mass",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for row in summary["layer_summary"]:
                payload = {key: row.get(key, "") for key in fieldnames}
                payload["model"] = summary["label"]
                payload["model_path"] = summary["model_path"]
                writer.writerow(payload)


def save_metric_plot(
    output_path: Path,
    summaries: Sequence[Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for summary in summaries:
        rows = summary["layer_summary"]
        if not rows:
            continue
        xs = [row["layer"] for row in rows]
        ys = [row[f"mean_{metric}"] for row in rows]
        yerr = [row[f"std_{metric}"] for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=summary["label"])
        ax.fill_between(xs, np.asarray(ys) - np.asarray(yerr), np.asarray(ys) + np.asarray(yerr), alpha=0.14)

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_model(
    label: str,
    model_path: str,
    cases: Sequence[Dict[str, Any]],
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
        decoder_path, decoder_layers = find_decoder_layers(model)
        layer_ids = parse_layer_ids(args, len(decoder_layers))
        print(f"[{label}] decoder_layers={decoder_path}, selected_layers={layer_ids}")

        case_results = []
        total_cases = len(cases)
        for case_idx, case in enumerate(cases):
            case_id = case.get("case_id", f"{case.get('eval_type')}_{case.get('index')}_{case.get('side')}")
            print(f"[{label}] case {case_idx + 1}/{total_cases}: {case_id}")
            try:
                case_results.append(compute_case_stats(label, model, processor, decoder_layers, layer_ids, case, args))
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"[WARN] {label} {case_id}: {type(exc).__name__}: {exc}")

        if not case_results:
            raise RuntimeError(f"{label} has no successful cases.")
        return aggregate_model_results(label, model_path, layer_ids, case_results)
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot layer-wise video-internal frame attention sink metrics for response tokens."
    )
    parser.add_argument("--case_file", type=str, default=None, help="Optional selected case JSON.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--eval_type", choices=list(DATASETS.keys()), default="fact")
    parser.add_argument("--side", choices=["basic", "hallucination"], default="hallucination")
    parser.add_argument("--side_filter", choices=["any", "basic", "hallucination"], default="hallucination")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--case_offset", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--favor_model_path", type=str, default=DEFAULT_FAVOR_MODEL_PATH)
    parser.add_argument("--video_r1_model_path", type=str, default=DEFAULT_VIDEO_R1_MODEL_PATH)
    parser.add_argument("--video_ktr_model_path", type=str, default=DEFAULT_VIDEO_KTR_MODEL_PATH)
    parser.add_argument("--onethinker_model_path", type=str, default=DEFAULT_ONETHINKER_MODEL_PATH)
    parser.add_argument("--auto_video_r1_model_path", type=str, default=DEFAULT_AUTO_VIDEO_R1_MODEL_PATH)
    parser.add_argument("--instruct_model_path", type=str, default=DEFAULT_QWEN3_VL_INSTRUCT_MODEL_PATH)
    parser.add_argument("--thinking_model_path", type=str, default=DEFAULT_QWEN3_VL_THINKING_MODEL_PATH)
    parser.add_argument(
        "--model_order",
        nargs="+",
        choices=["favor", "video_r1", "video_ktr", "onethinker", "auto_video_r1", "instruct", "thinking"],
        default=["video_ktr", "onethinker", "auto_video_r1", "favor", "thinking"],
    )

    parser.add_argument("--append_yes_no_prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Exact decoder layers to evaluate.")
    parser.add_argument("--start_layer", type=int, default=None)
    parser.add_argument("--end_layer", type=int, default=None)
    parser.add_argument("--layer_stride", type=int, default=1)
    parser.add_argument("--frame_norm", choices=["sum", "minmax", "zscore", "none"], default="sum", help=argparse.SUPPRESS)

    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--video_decode_backend", choices=["opencv", "processor"], default="opencv")
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_replay_tokens", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_frames is not None and args.fps is not None:
        raise ValueError("--num_frames and --fps are mutually exclusive.")

    cases = load_cases(args)
    print(f"selected {len(cases)} cases")

    summaries = []
    for label, model_path in selected_model_specs(args):
        summaries.append(run_model(label, model_path, cases, args))

    output_dir = Path(args.output_dir)
    payload = {
        "settings": {
            "case_file": args.case_file,
            "data_dir": args.data_dir,
            "eval_type": args.eval_type,
            "side": args.side,
            "side_filter": args.side_filter,
            "num_samples": args.num_samples,
            "case_offset": args.case_offset,
            "num_frames": args.num_frames,
            "max_new_tokens": args.max_new_tokens,
            "max_replay_tokens": args.max_replay_tokens,
            "metric_definition": "For each response token and layer: mean heads, sum attention over tokens in each video frame/bin, normalize across video frames, then compute entropy and frame sink metrics.",
            "model_order": args.model_order,
            "model_paths": {
                "video_ktr": args.video_ktr_model_path,
                "onethinker": args.onethinker_model_path,
                "auto_video_r1": args.auto_video_r1_model_path,
                "favor": args.favor_model_path,
                "instruct": args.instruct_model_path,
                "thinking": args.thinking_model_path,
                "video_r1": args.video_r1_model_path,
            },
        },
        "cases": cases,
        "models": summaries,
    }
    write_json(output_dir / "layer_frame_entropy_results.json", payload)
    write_csv(output_dir / "layer_frame_entropy_summary.csv", summaries)
    save_metric_plot(
        output_dir / "layer_frame_entropy.png",
        summaries,
        metric="entropy",
        ylabel="Entropy",
        title="Video-internal frame attention entropy by layer",
    )
    save_metric_plot(
        output_dir / "layer_frame_normalized_entropy.png",
        summaries,
        metric="normalized_entropy",
        ylabel="Normalized entropy",
        title="Video-internal frame attention normalized entropy by layer",
    )
    save_metric_plot(
        output_dir / "layer_frame_sink_ratio.png",
        summaries,
        metric="sink_ratio",
        ylabel="First-frame ratio",
        title="Video-internal first-frame sink ratio by layer",
    )
    save_metric_plot(
        output_dir / "layer_frame_other_avg_ratio.png",
        summaries,
        metric="other_frame_avg_ratio",
        ylabel="Other-frame average ratio",
        title="Video-internal other-frame average ratio by layer",
    )
    save_metric_plot(
        output_dir / "layer_frame_max_ratio.png",
        summaries,
        metric="max_frame_ratio",
        ylabel="Max frame ratio",
        title="Video-internal max-frame attention ratio by layer",
    )
    save_metric_plot(
        output_dir / "layer_frame_ratio_variance.png",
        summaries,
        metric="frame_ratio_variance",
        ylabel="Frame ratio variance",
        title="Video-internal frame attention variance by layer",
    )
    save_metric_plot(
        output_dir / "layer_video_attention_mass.png",
        summaries,
        metric="video_attention_mass",
        ylabel="Video attention mass",
        title="Response-token total video attention mass by layer",
    )

    print(f"results: {output_dir / 'layer_frame_entropy_results.json'}")
    print(f"summary: {output_dir / 'layer_frame_entropy_summary.csv'}")
    print(f"entropy plot: {output_dir / 'layer_frame_entropy.png'}")


if __name__ == "__main__":
    main()
