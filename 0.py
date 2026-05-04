"""
Motivation analysis: per-token attention metrics + hallucination detection + smoothing re-feed.

For each video QA sample:
1. Inference with output_attentions → per-token attention metrics
2. Gemini hallucination detection → map hallucinated phrases to token positions
3. Gemini smoothing → rewrite response without hallucinations
4. Re-feed smoothed prefix → check if accuracy improves

Usage:
    python analyze_motivation.py \
        --model_path /path/to/Qwen3-VL \
        --output_dir /path/to/stats \
        --num_samples 100
"""

import argparse
import json
import os
import re
import sys
import traceback

import numpy as np
import torch

# Import shared utilities from attention_sink
sys.path.insert(0, "/minimax-dialogue/users/garrick/code/tool/attention_sink")
from utils import (
    load_model_and_processor,
    load_data,
    prepare_model_inputs,
    parse_token_segments,
    extract_last_layer_attention,
    extract_answer,
    call_gemini,
    TokenSegments,
)


# ──────────────────────────────────────────────
# Accuracy checking (reused from analyze_attention_sink)
# ──────────────────────────────────────────────
def check_accuracy(model_response: str, ground_truth_solution: str) -> float:
    pred = extract_answer(model_response)
    gt = extract_answer(ground_truth_solution)
    if pred is None or gt is None:
        if gt and gt.lower() in model_response.lower():
            return 1.0
        return 0.0
    # Exact match
    if pred.strip().lower() == gt.strip().lower():
        return 1.0
    try:
        from mathruler.grader import grade_answer
        if grade_answer(pred, gt):
            return 1.0
    except ImportError:
        pass
    # Fallback: for yes/no, check if pred starts with the gt keyword
    gt_lower = gt.strip().lower()
    pred_lower = pred.strip().lower()
    if gt_lower in ("yes", "no") and pred_lower.startswith(gt_lower):
        return 1.0
    # Fallback: for multiple choice single letter, check if pred starts with it
    if len(gt.strip()) == 1 and gt.strip().isalpha() and pred_lower.startswith(gt_lower):
        return 1.0
    return 0.0


def build_query_with_options(item: dict) -> str:
    query = item.get("problem", "")
    if item.get("problem_type") == "multiple choice" and item.get("options"):
        opts = item["options"]
        if isinstance(opts, list):
            query = query + "\n" + "\n".join(opts)
    return query


# ──────────────────────────────────────────────
# Per-token attention metrics
# ──────────────────────────────────────────────
def compute_per_token_metrics(
    step_attns: list,
    segments: TokenSegments,
) -> dict:
    """Compute 6 attention metrics for each response token.

    Returns dict with per-token lists:
        first_frame_ratio:       attn[first_frame] / attn[all_video]
        other_frame_avg_ratio:   mean per-frame ratio for non-first frames
        text_sink_ratio:         attn[all_text] / (attn[all_text] + attn[all_video])
        video_attn_entropy:      entropy of attention distribution across frames
        max_frame_ratio:         max single frame attn / attn[all_video]
        video_attn_variance:     variance of per-frame attention ratios
    """
    first = segments.first_frame
    all_vid = segments.all_video
    # All text indices (system_prompt + timestamps + user_prompt)
    ts_all = [idx for ts in segments.timestamps for idx in ts]
    all_text = segments.all_text_prefix + ts_all + segments.user_prompt
    # All frames as separate groups
    all_frames = segments.video_frames
    other_frames = segments.video_frames[1:] if len(segments.video_frames) > 1 else []

    per_token_first_frame = []
    per_token_other_frame_avg = []
    per_token_text_sink = []
    per_token_entropy = []
    per_token_max_frame = []
    per_token_variance = []

    for attn in step_attns:
        if attn.ndim == 2:
            row = attn[-1, :]
        else:
            row = attn

        def safe_sum(indices):
            if not indices:
                return 0.0
            valid = [i for i in indices if i < len(row)]
            return float(row[valid].sum()) if valid else 0.0

        attn_first = safe_sum(first)
        attn_all_video = safe_sum(all_vid)
        attn_all_text = safe_sum(all_text)

        # first_frame_ratio
        if attn_all_video > 1e-12:
            per_token_first_frame.append(attn_first / attn_all_video)
        else:
            per_token_first_frame.append(0.0)

        # other_frame_avg_ratio
        if other_frames and attn_all_video > 1e-12:
            per_frame_ratios = []
            for frame_indices in other_frames:
                attn_frame = safe_sum(frame_indices)
                per_frame_ratios.append(attn_frame / attn_all_video)
            per_token_other_frame_avg.append(float(np.mean(per_frame_ratios)))
        else:
            per_token_other_frame_avg.append(0.0)

        # text_sink_ratio
        total_modality = attn_all_text + attn_all_video
        if total_modality > 1e-12:
            per_token_text_sink.append(attn_all_text / total_modality)
        else:
            per_token_text_sink.append(0.0)

        # Per-frame attention ratios for entropy/max/variance
        if all_frames and attn_all_video > 1e-12:
            frame_ratios = []
            for frame_indices in all_frames:
                frame_ratios.append(safe_sum(frame_indices) / attn_all_video)
            frame_ratios = np.array(frame_ratios)

            # video_attn_entropy: -sum(p * log(p))
            fr_safe = frame_ratios + 1e-12
            entropy = -float(np.sum(fr_safe * np.log(fr_safe)))
            per_token_entropy.append(entropy)

            # max_frame_ratio
            per_token_max_frame.append(float(np.max(frame_ratios)))

            # video_attn_variance
            per_token_variance.append(float(np.var(frame_ratios)))
        else:
            per_token_entropy.append(0.0)
            per_token_max_frame.append(0.0)
            per_token_variance.append(0.0)

    return {
        "per_token_first_frame_ratio": per_token_first_frame,
        "per_token_other_frame_avg_ratio": per_token_other_frame_avg,
        "per_token_text_sink_ratio": per_token_text_sink,
        "per_token_video_attn_entropy": per_token_entropy,
        "per_token_max_frame_ratio": per_token_max_frame,
        "per_token_video_attn_variance": per_token_variance,
    }


# ──────────────────────────────────────────────
# Hallucination phrase → token position mapping
# ──────────────────────────────────────────────
def map_phrases_to_token_positions(
    phrases: list,
    response_token_ids: list,
    tokenizer,
) -> tuple:
    """Map hallucinated phrases to token positions via character-level alignment.

    Returns:
        (hallucinated_indices, non_hallucinated_indices) — both response-relative
    """
    if not phrases or not response_token_ids:
        return [], list(range(len(response_token_ids)))

    # Decode full response
    response_text = tokenizer.decode(response_token_ids, skip_special_tokens=True)

    # Build char-level hallucination mask
    char_mask = [False] * len(response_text)
    for phrase in phrases:
        if not phrase:
            continue
        start = 0
        phrase_lower = phrase.lower()
        text_lower = response_text.lower()
        while True:
            idx = text_lower.find(phrase_lower, start)
            if idx == -1:
                break
            for c in range(idx, min(idx + len(phrase), len(response_text))):
                char_mask[c] = True
            start = idx + 1

    # Map tokens to character offsets
    hallucinated = []
    non_hallucinated = []
    char_offset = 0

    for tok_idx, tid in enumerate(response_token_ids):
        tok_str = tokenizer.decode([tid], skip_special_tokens=True)
        if not tok_str:
            non_hallucinated.append(tok_idx)
            continue

        pos = response_text.find(tok_str, char_offset)
        if pos == -1:
            # Try case-insensitive or nearby search
            non_hallucinated.append(tok_idx)
            continue

        tok_end = pos + len(tok_str)
        if any(char_mask[c] for c in range(pos, min(tok_end, len(response_text)))):
            hallucinated.append(tok_idx)
        else:
            non_hallucinated.append(tok_idx)
        char_offset = tok_end

    return hallucinated, non_hallucinated


# ──────────────────────────────────────────────
# Hallucination detection + smoothing (single Gemini call)
# ──────────────────────────────────────────────
DETECT_AND_SMOOTH_PROMPT = """\
Given a video QA task:
- Question: {question}
- Ground truth: {ground_truth}
- Model response: {model_response}

The model response has reasoning in <think>...</think> and a final answer in <answer>...</answer>.

1. Find the most critical hallucinated phrases in the <think> part (up to 3). Focus on phrases that are clearly inconsistent with the video content or have obvious logical errors. Ignore minor or ambiguous issues. Do NOT touch <answer>.
2. For each hallucinated phrase, provide a brief, concise replacement that removes the error.

Return JSON only:
```json
{{
  "is_hallucinated": true/false,
  "hallucinated_phrases": ["original1", "original2"],
  "corrected_phrases": ["concise fix1", "concise fix2"],
  "explanation": "one sentence"
}}
```
"""


def detect_and_smooth_hallucination(
    question: str,
    ground_truth: str,
    model_response: str,
    gemini_model: str = "gemini-2.5-flash",
) -> dict:
    """Single Gemini call: detect hallucinated phrases + produce smoothed response.

    Returns dict with keys: is_hallucinated, hallucinated_phrases, explanation, smoothed_response.
    """
    import re as _re

    prompt = DETECT_AND_SMOOTH_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        model_response=model_response,
    )
    msgs = [("user", [("t", prompt)])]
    resp = call_gemini(msgs, model_name=gemini_model)

    default = {
        "is_hallucinated": None,
        "hallucinated_phrases": [],
        "explanation": "Gemini API failed",
        "smoothed_response": None,
    }

    if resp is None:
        return default

    try:
        m = _re.search(r"```(?:json)?\s*(.*?)```", resp, _re.DOTALL)
        json_str = m.group(1).strip() if m else resp.strip()
        result = json.loads(json_str)

        # Build smoothed_response by replacing hallucinated phrases with corrected ones
        smoothed = None
        halluc_phrases = result.get("hallucinated_phrases", [])
        corrected_phrases = result.get("corrected_phrases", [])
        if halluc_phrases and corrected_phrases and result.get("is_hallucinated"):
            smoothed = model_response
            for old, new in zip(halluc_phrases, corrected_phrases):
                smoothed = smoothed.replace(old, new)

        return {
            "is_hallucinated": result.get("is_hallucinated"),
            "hallucinated_phrases": halluc_phrases,
            "corrected_phrases": corrected_phrases,
            "explanation": result.get("explanation", ""),
            "smoothed_response": smoothed,
        }
    except (json.JSONDecodeError, AttributeError):
        return default


# ──────────────────────────────────────────────
# Re-feed smoothed prefix to model
# ──────────────────────────────────────────────
def refeed_smoothed_prefix(
    model,
    processor,
    original_inputs: dict,
    original_prompt_length: int,
    smoothed_response_text: str,
    max_new_tokens: int = 256,
) -> tuple:
    """Re-feed smoothed response as prefix context, let model continue generating.

    Returns:
        (full_response_text, metadata_dict)
    """
    tokenizer = processor.tokenizer

    # Tokenize smoothed response
    smoothed_ids = tokenizer.encode(smoothed_response_text, add_special_tokens=False)
    smoothed_ids_tensor = torch.tensor([smoothed_ids], device=model.device)

    # Extract original prompt
    original_prompt_ids = original_inputs["input_ids"][:, :original_prompt_length]

    # Concatenate: prompt + smoothed response
    new_input_ids = torch.cat([original_prompt_ids, smoothed_ids_tensor], dim=1)
    new_attention_mask = torch.ones_like(new_input_ids)

    # Build new inputs, keeping visual features from original
    new_inputs = {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask,
    }
    for key in original_inputs:
        if key not in ("input_ids", "attention_mask"):
            new_inputs[key] = original_inputs[key]

    # Generate continuation (no attention output needed — just checking accuracy)
    with torch.no_grad():
        outputs = model.generate(
            **new_inputs,
            max_new_tokens=max_new_tokens,
        )

    # Decode: everything after original prompt is the full response
    generated_ids = outputs[0, original_prompt_length:]
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    metadata = {
        "smoothed_token_count": len(smoothed_ids),
        "new_prompt_length": new_input_ids.shape[1],
    }
    return full_response, metadata


# ──────────────────────────────────────────────
# Per-sample analysis
# ──────────────────────────────────────────────
def analyze_sample_motivation(
    model,
    processor,
    item: dict,
    sample_idx: int,
    num_frames: int = 32,
    max_new_tokens: int = 256,
    skip_smoothing: bool = False,
    use_gemini: bool = True,
) -> dict:
    """Full motivation analysis for a single sample."""

    query = build_query_with_options(item)
    video_path = item["path"]
    gt_solution = item.get("solution", "")
    gt_text = extract_answer(gt_solution) or gt_solution

    # 1. Prepare inputs & inference (always with output_attentions for sink metrics)
    thinking_suffix = (
        # "\nPlease think about this question as if you were a human pondering deeply. "
        "\nEngage in an internal dialogue using expressions such as 'let me think', 'wait', "
        "'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought "
        "expressions. It's encouraged to include self-reflection or verification in the "
        "reasoning process. Provide your detailed reasoning between the <think> and </think> "
        "tags, and then give your final answer between the <answer> and </answer> tags."
    )

    inputs, prompt_length = prepare_model_inputs(
        processor, model, video_path, query + thinking_suffix, num_frames=num_frames,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            use_cache=True,
        )

    # 2. Decode response
    generated_ids = outputs.sequences[0, prompt_length:]
    response_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    response_token_ids = generated_ids.tolist()

    # 3. Parse token segments — get actual frame count from video_grid_thw
    full_ids = outputs.sequences[0].tolist()
    actual_frames = None
    if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
        actual_frames = int(inputs["video_grid_thw"][0][0])

    segments = parse_token_segments(full_ids, prompt_length, num_frames=actual_frames)

    # 4. Attention metrics skipped during rollout (will be computed later via teacher-forcing)
    metrics = {}

    # 5. Accuracy
    accuracy = check_accuracy(response_text, gt_solution)

    # 6. Hallucination detection + smoothing (only if use_gemini)
    halluc_info = {
        "is_hallucinated": None,
        "hallucinated_phrases": [],
        "explanation": "skipped",
        "smoothed_response": None,
    }
    halluc_indices, non_halluc_indices = [], list(range(len(response_token_ids)))

    if use_gemini and accuracy == 0:
        halluc_info = detect_and_smooth_hallucination(
            question=query, ground_truth=gt_text, model_response=response_text
        )
        halluc_indices, non_halluc_indices = map_phrases_to_token_positions(
            halluc_info.get("hallucinated_phrases", []),
            response_token_ids,
            processor.tokenizer,
        )

    # 7. Aggregate attention metrics per halluc/non-halluc tokens
    agg = {}
    if use_gemini and metrics:
        def avg_metric_at_indices(metric_list, indices):
            if not indices:
                return None
            vals = [metric_list[i] for i in indices if i < len(metric_list)]
            return float(np.mean(vals)) if vals else None

        for metric_name in ["per_token_first_frame_ratio", "per_token_other_frame_avg_ratio", "per_token_text_sink_ratio"]:
            short_name = metric_name.replace("per_token_", "")
            agg[f"avg_{short_name}_hallucinated"] = avg_metric_at_indices(metrics[metric_name], halluc_indices)
            agg[f"avg_{short_name}_non_hallucinated"] = avg_metric_at_indices(metrics[metric_name], non_halluc_indices)

    # 8. Re-feed smoothed response (only if hallucinated and not skip_smoothing)
    smoothing_result = {
        "smoothed_response": None,
        "smoothed_full_response": None,
        "accuracy_after_smoothing": None,
        "smoothed_token_count": None,
        "tokens_changed": None,
    }

    smoothed_text = halluc_info.get("smoothed_response")
    if (halluc_info.get("is_hallucinated") and smoothed_text and not skip_smoothing):
            # Keep up to </think>, discard <answer> and after, let model re-generate answer
            think_end = smoothed_text.find("</think>")
            if think_end != -1:
                smoothed_prefix = smoothed_text[:think_end + len("</think>")]
            else:
                smoothed_prefix = smoothed_text
            smoothing_result["smoothed_response"] = smoothed_prefix
            try:
                full_resp, meta = refeed_smoothed_prefix(
                    model, processor, inputs, prompt_length,
                    smoothed_prefix, max_new_tokens=max_new_tokens,
                )
                smoothing_result["smoothed_full_response"] = full_resp
                smoothing_result["accuracy_after_smoothing"] = check_accuracy(full_resp, gt_solution)
                smoothing_result["smoothed_token_count"] = meta["smoothed_token_count"]
                smoothing_result["tokens_changed"] = abs(meta["smoothed_token_count"] - len(response_token_ids))
            except Exception as e:
                print(f"    Re-feed error: {e}")
                smoothing_result["smoothed_full_response"] = None
                smoothing_result["accuracy_after_smoothing"] = None

    # Build result
    result = {
        "sample_idx": sample_idx,
        "problem_id": item.get("problem_id"),
        "data_source": item.get("data_source", ""),
        "problem_type": item.get("problem_type", ""),
        "query": query[:500],
        "ground_truth": gt_solution,
        "model_response": response_text,
        "accuracy": accuracy,
        "num_input_frames": num_frames,
        "num_video_frames": len(segments.video_frames),
        "prompt_length": prompt_length,
        "response_length": len(response_token_ids),

        # Per-token metrics
        **metrics,

        # Hallucination detection
        "is_hallucinated": halluc_info.get("is_hallucinated"),
        "hallucinated_phrases": halluc_info.get("hallucinated_phrases", []),
        "corrected_phrases": halluc_info.get("corrected_phrases", []),
        "hallucination_explanation": halluc_info.get("explanation", ""),
        "hallucinated_token_indices": halluc_indices,

        # Aggregate comparison
        **agg,

        # Smoothing
        **smoothing_result,
    }

    return result


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Motivation analysis: attention + hallucination + smoothing")
    parser.add_argument("--model_path", type=str,
                        default="/minimax-dialogue/users/garrick/model/Qwen3-VL-Video-R1-CS-2B")
    parser.add_argument("--data_path", type=str,
                        default="/minimax-dialogue/users/garrick/data/Video-R1-Video/Video-R1-260k.json")
    parser.add_argument("--output_dir", type=str,
                        default="/minimax-dialogue/users/garrick/exp_log/DFAO_motivation/stats")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--skip_smoothing", action="store_true",
                        help="Skip hallucination smoothing and re-feed step")
    parser.add_argument("--use_gemini", action="store_true",
                        help="Enable Gemini hallucination detection + smoothing")
    args = parser.parse_args()

    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    dataset_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    output_file = os.path.join(dataset_dir, "results.jsonl")

    # Load model with flash_attention_2 for fast rollout
    print("Loading model and processor...")
    import json as _json
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Load processor (handle Qwen2.5-VL compatibility)
    config_path = os.path.join(args.model_path, "config.json")
    model_type = ""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_type = _json.load(f).get("model_type", "")

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2VLImageProcessor, Qwen2VLProcessor
        from transformers import Qwen2VLVideoProcessor
        image_processor = Qwen2VLImageProcessor.from_pretrained(args.model_path)
        video_processor = Qwen2VLVideoProcessor.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        tc_path = os.path.join(args.model_path, "tokenizer_config.json")
        chat_template = None
        if os.path.exists(tc_path):
            with open(tc_path, "r") as f:
                chat_template = _json.load(f).get("chat_template")
        processor = Qwen2VLProcessor(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Load data
    print("Loading data...")
    data = load_data(args.data_path, video_only=True)
    end_idx = min(args.start_idx + args.num_samples, len(data))
    data_slice = data[args.start_idx:end_idx]
    print(f"Processing samples {args.start_idx} to {end_idx - 1} ({len(data_slice)} samples)")
    print(f"Smoothing: {'OFF' if args.skip_smoothing else 'ON'}")
    print(f"Gemini: {'ON' if args.use_gemini else 'OFF'}")

    # Process samples
    stats = {"total": 0, "correct": 0, "errors": 0, "hallucinated": 0,
             "smoothed_correct": 0, "smoothed_total": 0}

    with open(output_file, "w") as fout:
        for i, item in enumerate(data_slice):
            global_idx = args.start_idx + i

            print(f"\n[{i + 1}/{len(data_slice)}] Sample {global_idx}: "
                  f"{item.get('problem', '')[:60]}...")

            try:
                result = analyze_sample_motivation(
                    model, processor, item,
                    sample_idx=global_idx,
                    num_frames=args.num_frames,
                    max_new_tokens=args.max_new_tokens,
                    skip_smoothing=args.skip_smoothing,
                    use_gemini=args.use_gemini,
                )

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                stats["total"] += 1
                if result["accuracy"] > 0:
                    stats["correct"] += 1
                if result.get("is_hallucinated"):
                    stats["hallucinated"] += 1
                if result.get("accuracy_after_smoothing") is not None:
                    stats["smoothed_total"] += 1
                    if result["accuracy_after_smoothing"] > 0:
                        stats["smoothed_correct"] += 1

                print(f"  Acc: {result['accuracy']:.0f} | "
                      f"Halluc: {result.get('is_hallucinated', 'N/A')} | "
                      f"Halluc tokens: {len(result.get('hallucinated_token_indices', []))} | "
                      f"Acc after smooth: {result.get('accuracy_after_smoothing', 'N/A')}")

            except Exception as e:
                stats["errors"] += 1
                print(f"  ERROR: {e}")
                traceback.print_exc()
                continue

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {stats['total']}")
    print(f"Correct: {stats['correct']} ({stats['correct'] / max(stats['total'], 1):.2%})")
    print(f"Hallucinated: {stats['hallucinated']}")
    print(f"Errors: {stats['errors']}")
    if stats["smoothed_total"] > 0:
        print(f"Smoothed correct: {stats['smoothed_correct']}/{stats['smoothed_total']} "
              f"({stats['smoothed_correct'] / stats['smoothed_total']:.2%})")
    print(f"Results saved to: {output_file}")

    # Aggregate hallucinated vs non-hallucinated comparison
    if stats["total"] > 0:
        results = []
        with open(output_file, "r") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        metric_keys = ["first_frame_ratio", "other_frame_avg_ratio", "text_sink_ratio"]
        halluc_vals = {k: [] for k in metric_keys}
        non_halluc_vals = {k: [] for k in metric_keys}

        for r in results:
            for k in metric_keys:
                v_h = r.get(f"avg_{k}_hallucinated")
                v_n = r.get(f"avg_{k}_non_hallucinated")
                if v_h is not None:
                    halluc_vals[k].append(v_h)
                if v_n is not None:
                    non_halluc_vals[k].append(v_n)

        print(f"\n--- Hallucinated token avg metrics (across {len(halluc_vals[metric_keys[0]])} samples) ---")
        for k in metric_keys:
            if halluc_vals[k]:
                print(f"  {k}: mean={np.mean(halluc_vals[k]):.4f}, std={np.std(halluc_vals[k]):.4f}")

        print(f"\n--- Non-hallucinated token avg metrics (across {len(non_halluc_vals[metric_keys[0]])} samples) ---")
        for k in metric_keys:
            if non_halluc_vals[k]:
                print(f"  {k}: mean={np.mean(non_halluc_vals[k]):.4f}, std={np.std(non_halluc_vals[k]):.4f}")


if __name__ == "__main__":
    main()