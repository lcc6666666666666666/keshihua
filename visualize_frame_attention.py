import argparse
import textwrap
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from favor_utils import (
    DATASETS,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    attention_metrics,
    build_frame_token_map,
    clone_tensor_inputs,
    decode_generated_outputs,
    ensure_yes_no_prompt,
    find_decoder_layers,
    frame_indices_from_metadata,
    generate_text,
    get_model_device,
    get_video_token_id,
    get_video_token_positions,
    infer_grid_shape,
    load_favor_model,
    load_video_frames,
    move_inputs_to_device,
    prepare_qwen_video_inputs,
    read_json,
    strip_model_inputs,
    write_json,
)


class DecoderAttentionCollector:
    def __init__(self, layers: Sequence[Any], layer_ids: Sequence[int], video_positions: torch.Tensor):
        self.layers = layers
        self.layer_ids = list(layer_ids)
        self.video_positions = video_positions
        self.records: Dict[int, List[torch.Tensor]] = {idx: [] for idx in self.layer_ids}
        self.enabled = False
        self._originals: List[Tuple[Any, Any]] = []

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
            attn = outputs.attentions

        if attn is None:
            return

        key_len = attn.shape[-1]
        positions = self.video_positions.to(attn.device)
        positions = positions[positions < key_len]
        if positions.numel() == 0:
            return

        values = attn[:, :, -1, positions]
        self.records[layer_idx].append(values[0].detach().float().cpu())


def safe_forward(model: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        kwargs = dict(kwargs)
        kwargs.pop("logits_to_keep", None)
        return model(**kwargs)


def non_special_token_mask(processor: Any, generated_ids: torch.Tensor) -> Optional[List[bool]]:
    tokenizer = getattr(processor, "tokenizer", processor)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if not special_ids or generated_ids.shape[0] != 1:
        return None
    return [int(tok) not in special_ids for tok in generated_ids[0].detach().cpu().tolist()]


def limit_collect_mask(
    mask: Optional[Sequence[bool]],
    max_collect_tokens: Optional[int],
    total_tokens: Optional[int] = None,
) -> Optional[List[bool]]:
    if mask is None:
        if max_collect_tokens is None:
            return None
        if total_tokens is None:
            raise ValueError("total_tokens is required when limiting collection without an existing mask.")
        limit = max(0, int(max_collect_tokens))
        return [idx < limit for idx in range(int(total_tokens))]
    if max_collect_tokens is None:
        return [bool(item) for item in mask]
    remaining = max(0, int(max_collect_tokens))
    output: List[bool] = []
    for item in mask:
        collect = bool(item) and remaining > 0
        output.append(collect)
        if collect:
            remaining -= 1
    return output


def count_collect_tokens(mask: Optional[Sequence[bool]], generated_ids: torch.Tensor) -> int:
    if mask is None:
        return int(generated_ids.shape[1])
    return int(sum(1 for item in mask if item))


def replay_generated_tokens(
    model: Any,
    inputs: Dict[str, Any],
    generated_ids: torch.Tensor,
    collector: DecoderAttentionCollector,
    collect_token_mask: Optional[Sequence[bool]] = None,
) -> None:
    if collect_token_mask is not None and len(collect_token_mask) != int(generated_ids.shape[1]):
        raise ValueError(
            f"collect_token_mask length {len(collect_token_mask)} does not match generated length {generated_ids.shape[1]}."
        )

    model_inputs = strip_model_inputs(clone_tensor_inputs(inputs))
    device = model_inputs["input_ids"].device
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)

    prefill_kwargs = dict(model_inputs)
    prefill_kwargs.update({"use_cache": True, "return_dict": True, "logits_to_keep": 1})

    collector.enabled = False
    with torch.inference_mode():
        outputs = safe_forward(model, prefill_kwargs)
    past_key_values = outputs.past_key_values

    with torch.inference_mode():
        for step in range(generated_ids.shape[1]):
            collector.enabled = collect_token_mask is None or bool(collect_token_mask[step])
            token = generated_ids[:, step : step + 1].to(device)
            new_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)

            cache_position = torch.arange(
                input_ids.shape[1] + step,
                input_ids.shape[1] + step + 1,
                device=device,
            )
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)[:, -1:]
            position_ids = position_ids.view(1, position_ids.shape[0], 1).repeat(3, 1, 1).to(device)
            base_model = getattr(model, "model", model)
            rope_deltas = getattr(base_model, "rope_deltas", None)
            if rope_deltas is not None:
                delta = rope_deltas.to(device=position_ids.device, dtype=position_ids.dtype)
                if delta.shape[0] != position_ids.shape[1] and position_ids.shape[1] % delta.shape[0] == 0:
                    delta = delta.repeat_interleave(position_ids.shape[1] // delta.shape[0], dim=0)
                position_ids = position_ids + delta

            step_kwargs = {
                "input_ids": token,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "cache_position": cache_position,
                "position_ids": position_ids,
                "use_cache": True,
                "return_dict": True,
                "logits_to_keep": 1,
            }
            outputs = safe_forward(model, step_kwargs)
            past_key_values = outputs.past_key_values
    collector.enabled = False


def reduce_attention_records(
    records: Dict[int, List[torch.Tensor]],
    layer_ids: Sequence[int],
    total_video_tokens: int,
) -> np.ndarray:
    chunks: List[torch.Tensor] = []
    for layer_idx in layer_ids:
        layer_records = records.get(layer_idx, [])
        if not layer_records:
            continue
        stacked = torch.stack(layer_records, dim=0)
        chunks.append(stacked.reshape(-1, total_video_tokens))

    if not chunks:
        raise RuntimeError("No decode-step attentions were collected from the selected layers.")

    scores = torch.cat(chunks, dim=0).mean(dim=0).numpy()
    scores = np.maximum(scores, 0.0)
    total = float(scores.sum())
    if total <= 0:
        raise RuntimeError("Collected video-token attention is all zero.")
    return scores / total


def token_scores_to_frame_outputs(
    token_scores: np.ndarray,
    frame_map: Dict[str, Any],
    norm_mode: str = "ratio_scaled",
    ratio_power: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
    return _token_scores_to_frame_outputs_impl(
        token_scores,
        frame_map,
        norm_mode=norm_mode,
        ratio_power=ratio_power,
        eps=eps,
    )


def local_minmax_heatmap(hm_raw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    hm_min = float(hm_raw.min())
    hm_max = float(hm_raw.max())
    if hm_max <= hm_min:
        return np.zeros_like(hm_raw, dtype=np.float64)
    return (hm_raw - hm_min) / (hm_max - hm_min + eps)


def _token_scores_to_frame_outputs_impl(
    token_scores: np.ndarray,
    frame_map: Dict[str, Any],
    norm_mode: str = "ratio_scaled",
    ratio_power: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
    if norm_mode not in {"local", "global", "ratio_scaled"}:
        raise ValueError(f"Unsupported heatmap norm_mode: {norm_mode}")

    scores = np.asarray(token_scores, dtype=np.float64)
    raw_ratios: List[float] = []
    heatmaps: List[np.ndarray] = []
    raw_heatmaps: List[np.ndarray] = []
    default_shape = tuple(frame_map["heatmap_shape"])

    for _, start, end in frame_map["frame_slices"]:
        frame_scores = scores[start:end]
        raw_ratios.append(float(frame_scores.sum()))
        shape = default_shape
        if int(np.prod(shape)) != len(frame_scores):
            shape = infer_grid_shape(len(frame_scores))
        raw_heatmaps.append(frame_scores.reshape(shape))

    score_sum = float(scores.sum())
    ratio_denominator = score_sum if score_sum > 0 else eps
    ratios = [float(r / ratio_denominator) for r in raw_ratios]

    global_max = float(scores.max()) if scores.size else 0.0
    max_ratio = max(ratios) if ratios else 0.0

    for hm_raw, ratio in zip(raw_heatmaps, ratios):
        if norm_mode == "local":
            heatmap = local_minmax_heatmap(hm_raw, eps=eps)
        elif norm_mode == "global":
            if global_max > 0:
                heatmap = hm_raw / global_max
            else:
                heatmap = np.zeros_like(hm_raw, dtype=np.float64)
        else:
            hm_local = local_minmax_heatmap(hm_raw, eps=eps)
            if max_ratio > 0:
                scale_base = max(ratio, 0.0) / max_ratio
                scale = scale_base ** ratio_power if scale_base > 0 or ratio_power >= 0 else 0.0
            else:
                scale = 0.0
            heatmap = hm_local * scale
        heatmaps.append(np.clip(heatmap, 0.0, 1.0))

    return ratios, heatmaps, raw_heatmaps


def heatmap_list_to_array(heatmaps: Sequence[np.ndarray]) -> np.ndarray:
    if not heatmaps:
        return np.empty((0,), dtype=np.float32)
    try:
        return np.stack([np.asarray(hm, dtype=np.float32) for hm in heatmaps], axis=0)
    except ValueError:
        output = np.empty((len(heatmaps),), dtype=object)
        for idx, heatmap in enumerate(heatmaps):
            output[idx] = np.asarray(heatmap, dtype=np.float32)
        return output


def save_attention_figure(
    output_path: Path,
    frames: Sequence[np.ndarray],
    heatmaps: Sequence[np.ndarray],
    ratios: Sequence[float],
    question: str,
    prediction: str,
    answer: str,
    cols: int,
    overlay_alpha: float,
) -> None:
    import matplotlib.pyplot as plt

    n = len(ratios)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    fig_w = 3.0 * cols
    fig_h = 3.05 * rows + 1.2
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        if idx >= n:
            continue
        if idx < len(frames):
            frame = frames[idx]
        else:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        ax.imshow(frame)
        heatmap = np.clip(np.asarray(heatmaps[idx]), 0.0, 1.0)
        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=overlay_alpha,
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            extent=(0, frame.shape[1], frame.shape[0], 0),
        )
        ax.set_title(f"Frame {idx}\nRatio: {ratios[idx] * 100:.2f}%", fontsize=10, fontweight="bold")

    wrapped_q = "\n".join(textwrap.wrap(f"Q: {question}", width=110))
    footer = f"{wrapped_q}\nA: {prediction or '<empty>'}"
    if answer:
        footer += f"    GT: {answer}"
    fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_ratio_bar(output_path: Path, ratios: Sequence[float]) -> None:
    import matplotlib.pyplot as plt

    xs = np.arange(len(ratios))
    uniform = 1.0 / len(ratios) if ratios else 0.0
    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * len(ratios)), 3.2))
    ax.bar(xs, ratios, color="#2878b5")
    ax.axhline(uniform, color="#c82423", linestyle="--", linewidth=1.2, label="uniform")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs])
    ax.set_ylim(0, max(max(ratios) * 1.15, uniform * 1.5) if ratios else 1)
    ax.set_xlabel("Frame / temporal bin")
    ax.set_ylabel("Attention ratio")
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def load_case(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.case_file:
        if not args.video_path or not args.question:
            raise ValueError("Provide --case_file or both --video_path and --question.")
        return {
            "video_path": args.video_path,
            "question": args.question,
            "answer": args.answer or "",
            "predict": "",
            "eval_type": "manual",
            "index": 0,
            "side": "manual",
        }

    cases = read_json(args.case_file)
    case = cases[args.case_index]
    if "video_path" not in case:
        if not args.data_dir:
            raise ValueError("Selected case lacks video_path; pass --data_dir to resolve it.")
        eval_type = case["eval_type"]
        case["video_path"] = str(Path(args.data_dir) / DATASETS[eval_type]["video_dir"] / case["video"])
    return case


def parse_layers(args: argparse.Namespace, total_layers: int) -> List[int]:
    if args.layers:
        layers = args.layers
    elif args.start_layer is not None or args.end_layer is not None:
        start_layer = 0 if args.start_layer is None else args.start_layer
        end_layer = total_layers - 1 if args.end_layer is None else args.end_layer
        layers = list(range(start_layer, end_layer + 1))
    else:
        last_n = max(1, min(args.last_n_layers, total_layers))
        layers = list(range(total_layers - last_n, total_layers))
    valid = [idx for idx in layers if 0 <= idx < total_layers]
    if not valid:
        raise ValueError(f"No valid layer indices from {layers}; model has {total_layers} layers.")
    if len(valid) != len(layers):
        print(f"[WARN] Dropped out-of-range layers. Requested={layers}, valid={valid}, total_layers={total_layers}")
    return valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FAVOR/Qwen3-VL decoder attention over video frames.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--case_file", type=str, default=None, help="JSON generated by select_cases.py.")
    parser.add_argument("--case_index", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="VideoHallucer root if case_file lacks video_path.")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--answer", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--append_yes_no_prompt", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Exact decoder layers to average. If omitted, use the last N decoder layers.")
    parser.add_argument("--start_layer", type=int, default=None, help="Optional first decoder layer for a contiguous range.")
    parser.add_argument("--end_layer", type=int, default=None, help="Optional last decoder layer for a contiguous range.")
    parser.add_argument("--last_n_layers", type=int, default=1, help="Default number of final decoder layers to average.")
    parser.add_argument("--num_frames", type=int, default=16, help="Raw frames sampled by Qwen3-VL.")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--video_decode_backend", choices=["opencv", "processor"], default="opencv")
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Keep default/fast backend; selected decode steps are forced to eager inside hooks.")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument("--heatmap_norm", choices=["local", "global", "ratio_scaled"], default="ratio_scaled")
    parser.add_argument("--ratio_power", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_frames is not None and args.fps is not None:
        raise ValueError("--num_frames and --fps are mutually exclusive.")

    case = load_case(args)
    video_path = case["video_path"]
    question = case["question"]
    if args.append_yes_no_prompt:
        question_for_model = ensure_yes_no_prompt(question)
    else:
        question_for_model = question

    model, processor = load_favor_model(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    device = get_model_device(model)

    raw_inputs, _ = prepare_qwen_video_inputs(
        processor=processor,
        video_path=video_path,
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
    layer_ids = parse_layers(args, len(decoder_layers))
    print(f"decoder_layers={decoder_path}, selected_layers={layer_ids}")
    print(
        f"video_tokens={frame_map['total_video_tokens']}, grid_thw={frame_map['grid_thw']}, "
        f"temporal_bins={frame_map['num_bins']}"
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
    collect_mask = non_special_token_mask(processor, generated_ids)
    num_collected_tokens = count_collect_tokens(collect_mask, generated_ids)
    if generated_ids.numel() == 0:
        raise RuntimeError("Model produced no generated tokens to replay.")
    if num_collected_tokens == 0:
        raise RuntimeError("Model produced no non-special generated tokens to collect.")

    collector = DecoderAttentionCollector(decoder_layers, layer_ids, video_positions)
    collector.install()
    try:
        replay_generated_tokens(model, inputs, generated_ids, collector, collect_token_mask=collect_mask)
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
    frames = load_video_frames(video_path, frame_indices=frame_indices, count=frame_map["num_bins"])

    output_dir = Path(args.output_dir)
    stem = f"{case.get('eval_type', 'manual')}_{case.get('index', 0)}_{case.get('side', 'manual')}"
    fig_path = output_dir / f"{stem}_attention.png"
    bar_path = output_dir / f"{stem}_ratios.png"
    metrics_path = output_dir / f"{stem}_metrics.json"
    npz_path = output_dir / f"{stem}_attention.npz"

    save_attention_figure(
        output_path=fig_path,
        frames=frames,
        heatmaps=heatmaps,
        ratios=ratios,
        question=question,
        prediction=prediction,
        answer=str(case.get("answer", args.answer or "")),
        cols=args.cols,
        overlay_alpha=args.overlay_alpha,
    )
    save_ratio_bar(bar_path, ratios)

    metrics = {
        "case": case,
        "model_path": args.model_path,
        "generated_text": prediction,
        "generated_text_full": decoded["full_text"],
        "extracted_answer": decoded["extracted_answer"],
        "layers": layer_ids,
        "num_replayed_tokens": int(generated_ids.shape[1]),
        "num_collected_tokens": num_collected_tokens,
        "frame_ratios": ratios,
        "heatmap_norm": args.heatmap_norm,
        "ratio_power": args.ratio_power,
        "uniform_ratio": 1.0 / len(ratios) if ratios else 0.0,
        "metrics": attention_metrics(ratios),
        "frame_map": {
            key: value
            for key, value in frame_map.items()
            if key != "frame_slices"
        },
    }
    write_json(metrics_path, metrics)
    np.savez_compressed(
        npz_path,
        token_scores=token_scores,
        ratios=np.asarray(ratios),
        frame_ratios=np.asarray(ratios),
        heatmaps=heatmap_list_to_array(heatmaps),
        raw_heatmaps=heatmap_list_to_array(raw_heatmaps),
        heatmap_norm=args.heatmap_norm,
        ratio_power=np.asarray(args.ratio_power),
        layers=np.asarray(layer_ids),
    )

    print(f"prediction: {prediction}")
    print(f"attention figure: {fig_path}")
    print(f"ratio bar: {bar_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
