import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


DATASETS = {
    "obj_rel": {
        "json_path": "object_relation/object_relation.json",
        "video_dir": "object_relation/videos",
    },
    "temporal": {
        "json_path": "temporal/temporal.json",
        "video_dir": "temporal/videos",
    },
    "semantic": {
        "json_path": "semantic_detail/semantic_detail.json",
        "video_dir": "semantic_detail/videos",
    },
    "interaction": {
        "json_path": "interaction/interaction.json",
        "video_dir": "interaction/videos",
    },
    "fact": {
        "json_path": "external_factual/external_factual.json",
        "video_dir": "external_factual/videos",
    },
    "nonfact": {
        "json_path": "external_nonfactual/external_nonfactual.json",
        "video_dir": "external_nonfactual/videos",
    },
    "factdet": {
        "json_path": "fact_detect/fact_detect.json",
        "video_dir": "fact_detect/videos",
    },
}


VIDEO_METADATA_KEYS = {
    "total_num_frames",
    "fps",
    "width",
    "height",
    "duration",
    "video_backend",
    "frames_indices",
}


DEFAULT_MODEL_PATH = "/data1/lgy/model/FAVOR0.5-8B"
DEFAULT_INSTRUCT_MODEL_PATH = "/data1/lgy/model/Qwen3-VL-8B-Instruct"
DEFAULT_THINKING_MODEL_PATH = "/data1/lgy/model/Qwen3-VL-8B-Thinking"
DEFAULT_DATA_DIR = "/data1/lgy/eval/video_r1/Evaluation/VideoHallucer"
DEFAULT_MODEL_PATHS = {
    "favor": DEFAULT_MODEL_PATH,
    "instruct": DEFAULT_INSTRUCT_MODEL_PATH,
    "thinking": DEFAULT_THINKING_MODEL_PATH,
}


DROP_MODEL_INPUT_KEYS = {"token_type_ids", "video_metadata"}
YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def ensure_yes_no_prompt(question: str) -> str:
    marker = "Answer the question using"
    if marker.lower() in question.lower():
        return question
    return f"{question}\nAnswer the question using 'yes' or 'no'."


def _legacy_answer_hit_disabled(answer: str, prediction: str) -> bool:
    if prediction is None:
        return False
    extracted = extract_yes_no_answer(prediction)
    if extracted is not None:
        return extracted.lower() == str(answer).strip().lower()
    pattern = r"\b(" + re.escape(str(answer).strip()) + r")\b"
    return re.search(pattern, str(prediction), re.IGNORECASE) is not None


def _legacy_extract_yes_no_answer_disabled(prediction: str) -> Optional[str]:
    if prediction is None:
        return None
    text = str(prediction).strip()
    if not text:
        return None

    # Prefer the explicit final answer after common markers; otherwise use the last
    # yes/no token, which is more robust for thinking models that mention both.
    marker_patterns = [
        r"(?:final answer|answer|答案|最终答案)\s*[:：]\s*(yes|no)\b",
        r"\b(?:therefore|so),?\s*(?:the answer is\s*)?(yes|no)\b",
        r"\bthe answer is\s*(yes|no)\b",
    ]
    lowered = text.lower()
    for pattern in marker_patterns:
        matches = re.findall(pattern, lowered, flags=re.IGNORECASE)
        if matches:
            return matches[-1].lower()

    matches = re.findall(r"\b(yes|no)\b", lowered, flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return None


def normalize_yes_no(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = text.strip(" .,:;!?\"'`[](){}<>/\n\t")
    match = YES_NO_PATTERN.fullmatch(text)
    if match:
        return match.group(1).lower()
    return None


def first_yes_no(text: str) -> Optional[str]:
    match = YES_NO_PATTERN.search(str(text))
    return match.group(1).lower() if match else None


def last_yes_no(text: str) -> Optional[str]:
    matches = YES_NO_PATTERN.findall(str(text))
    return matches[-1].lower() if matches else None


def strip_generation_specials(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_from_answer_tags(text: str) -> Optional[str]:
    # FAVOR format: <think>...</think>\n<answer>yes</answer>
    tagged = re.findall(r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", text, flags=re.IGNORECASE | re.DOTALL)
    for content in reversed(tagged):
        extracted = first_yes_no(strip_generation_specials(content))
        if extracted:
            return extracted

    open_tag = re.findall(r"<\s*answer\s*>\s*(yes|no)\b", text, flags=re.IGNORECASE | re.DOTALL)
    if open_tag:
        return open_tag[-1].lower()
    return None


def extract_from_post_think(text: str) -> Optional[str]:
    # Thinking format: reasoning text ... </think>\n\nyes
    pieces = re.split(r"<\s*/\s*think\s*>", text, flags=re.IGNORECASE)
    if len(pieces) <= 1:
        return None
    tail = strip_generation_specials(pieces[-1])
    explicit = extract_from_markers(tail)
    if explicit:
        return explicit
    normalized = normalize_yes_no(tail)
    if normalized:
        return normalized
    return first_yes_no(tail)


def extract_from_markers(text: str) -> Optional[str]:
    marker_patterns = [
        r"\bfinal\s+answer\s*(?:is)?\W+(yes|no)\b",
        r"\bthe\s+answer\s+is\W+(yes|no)\b",
        r"\banswer\s*(?:is)?\W+(yes|no)\b",
        r"\btherefore\W+(?:the\s+answer\s+is\W+)?(yes|no)\b",
        r"\bso\W+(?:the\s+answer\s+is\W+)?(yes|no)\b",
    ]
    for pattern in marker_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].lower()
    return None


def extract_from_last_lines(text: str) -> Optional[str]:
    lines = [strip_generation_specials(line) for line in str(text).splitlines()]
    lines = [line for line in lines if line]
    for line in reversed(lines[-4:]):
        normalized = normalize_yes_no(line)
        if normalized:
            return normalized
        if len(line.split()) <= 6:
            extracted = first_yes_no(line)
            if extracted:
                return extracted
    return None


def extract_yes_no_answer(prediction: Any) -> Optional[str]:
    if prediction is None:
        return None
    text = str(prediction).strip()
    if not text:
        return None

    extractors = [
        extract_from_answer_tags,
        extract_from_post_think,
        extract_from_markers,
        extract_from_last_lines,
    ]
    for extractor in extractors:
        extracted = extractor(text)
        if extracted:
            return extracted

    return last_yes_no(strip_generation_specials(text))


def answer_hit(answer: str, prediction: str) -> bool:
    expected = normalize_yes_no(answer)
    if expected is None:
        return False
    extracted = extract_yes_no_answer(prediction)
    return extracted == expected


def get_record_extracted_answer(qa: Dict[str, Any]) -> Optional[str]:
    explicit = normalize_yes_no(qa.get("extracted_answer"))
    if explicit:
        return explicit
    for key in ("predict", "predict_full"):
        extracted = extract_yes_no_answer(qa.get(key))
        if extracted:
            return extracted
    return None


def qa_answer_hit(qa: Dict[str, Any]) -> bool:
    expected = normalize_yes_no(qa.get("answer"))
    extracted = get_record_extracted_answer(qa)
    return expected is not None and extracted == expected


def cal_score(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {"basic_accuracy": 0.0, "halluc_accuracy": 0.0, "accuracy": 0.0}

    basic_acc = 0
    halluc_acc = 0
    pair_acc = 0
    for result in results:
        basic = result.get("basic", {})
        halluc = result.get("hallucination", {})
        basic_ok = qa_answer_hit(basic)
        halluc_ok = qa_answer_hit(halluc)
        basic_acc += int(basic_ok)
        halluc_acc += int(halluc_ok)
        pair_acc += int(basic_ok and halluc_ok)

    total = len(results)
    return {
        "basic_accuracy": basic_acc / total,
        "halluc_accuracy": halluc_acc / total,
        "accuracy": pair_acc / total,
    }


def read_json(path: os.PathLike[str] | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: os.PathLike[str] | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_dtype(dtype_name: str) -> Any:
    name = (dtype_name or "auto").lower()
    if name == "auto":
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _try_from_pretrained(cls: Any, model_path: str, load_kwargs: Dict[str, Any], dtype_value: Any) -> Any:
    variants: List[Dict[str, Any]] = []
    if dtype_value is not None:
        variants.append({**load_kwargs, "dtype": dtype_value})
        variants.append({**load_kwargs, "torch_dtype": dtype_value})
    variants.append(dict(load_kwargs))

    errors: List[str] = []
    for kwargs in variants:
        try:
            return cls.from_pretrained(model_path, **kwargs)
        except TypeError as exc:
            errors.append(f"{cls.__name__}({sorted(kwargs)}): {exc}")
            continue
    raise RuntimeError("; ".join(errors))


def load_favor_model(
    model_path: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any]:
    from transformers import AutoConfig, AutoProcessor

    dtype_value = parse_dtype(dtype)
    config_model_type = None
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config_model_type = getattr(config, "model_type", None)
    except Exception:
        pass
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model_classes: List[Any] = []

    def add_model_class(cls: Any) -> None:
        if cls not in model_classes:
            model_classes.append(cls)

    if config_model_type == "qwen2_5_vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            add_model_class(Qwen2_5_VLForConditionalGeneration)
        except Exception:
            pass
    elif config_model_type == "qwen2_vl":
        try:
            from transformers import Qwen2VLForConditionalGeneration

            add_model_class(Qwen2VLForConditionalGeneration)
        except Exception:
            pass
    elif config_model_type == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration

            add_model_class(Qwen3VLForConditionalGeneration)
        except Exception:
            pass
    elif config_model_type == "qwen3_vl_moe":
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration

            add_model_class(Qwen3VLMoeForConditionalGeneration)
        except Exception:
            pass

    if config_model_type not in {"qwen2_5_vl", "qwen2_vl"}:
        try:
            from transformers import Qwen3VLForConditionalGeneration

            add_model_class(Qwen3VLForConditionalGeneration)
        except Exception:
            pass
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration

            add_model_class(Qwen3VLMoeForConditionalGeneration)
        except Exception:
            pass
    try:
        from transformers import AutoModelForImageTextToText

        add_model_class(AutoModelForImageTextToText)
    except Exception:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        add_model_class(AutoModelForVision2Seq)
    except Exception:
        pass
    try:
        from transformers import AutoModelForCausalLM

        add_model_class(AutoModelForCausalLM)
    except Exception:
        pass

    base_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device_map:
        base_kwargs["device_map"] = device_map
    if attn_implementation:
        base_kwargs["attn_implementation"] = attn_implementation

    errors: List[str] = []
    for cls in model_classes:
        try:
            model = _try_from_pretrained(cls, model_path, base_kwargs, dtype_value)
            model.eval()
            return model, processor
        except Exception as exc:
            errors.append(f"{getattr(cls, '__name__', cls)}: {exc}")

    raise RuntimeError("Could not load FAVOR/Qwen3-VL model. Tried: " + " | ".join(errors))


def get_model_device(model: Any) -> torch.device:
    if hasattr(model, "device") and isinstance(model.device, torch.device):
        if model.device.type != "meta":
            return model.device
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_video_messages(video_input: Any, question: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_input},
                {"type": "text", "text": question},
            ],
        }
    ]


def sample_video_opencv(
    video_path: str,
    num_frames: Optional[int] = None,
    fps: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required when video_decode_backend='opencv'.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None

    if total <= 0:
        cap.release()
        raise RuntimeError(f"Could not determine frame count for video: {video_path}")

    if num_frames is None and fps is not None and source_fps:
        num_frames = int(total / source_fps * fps)
    if num_frames is None:
        num_frames = total
    requested_num_frames = max(1, int(num_frames))
    num_frames = min(requested_num_frames, total)
    indices = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()

    frames: List[np.ndarray] = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not decode sampled frames for video: {video_path}")

    metadata = {
        "total_num_frames": total,
        "fps": source_fps,
        "width": width,
        "height": height,
        "duration": (total / source_fps) if source_fps else None,
        "video_backend": "opencv",
        "requested_num_frames": requested_num_frames,
        "sampled_num_frames": len(frames),
        "frames_indices": indices[: len(frames)],
    }
    return np.stack(frames, axis=0), metadata


def processor_video_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in metadata.items() if key in VIDEO_METADATA_KEYS}


def prepare_qwen_video_inputs(
    processor: Any,
    video_path: str,
    question: str,
    num_frames: Optional[int] = None,
    fps: Optional[float] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    return_metadata: bool = False,
    video_decode_backend: str = "opencv",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if num_frames is not None and fps is not None:
        raise ValueError("num_frames and fps are mutually exclusive for Qwen3-VL video sampling.")

    video_kwargs: Dict[str, Any] = {}
    metadata = None
    if video_decode_backend == "opencv":
        video_input, metadata = sample_video_opencv(
            video_path,
            num_frames=num_frames,
            fps=fps,
        )
        video_kwargs["video_metadata"] = [processor_video_metadata(metadata)]
        video_kwargs["do_sample_frames"] = False
    elif video_decode_backend == "processor":
        video_input = str(video_path)
        metadata = None
    else:
        raise ValueError(f"Unsupported video_decode_backend: {video_decode_backend}")

    messages = build_video_messages(video_input, question)
    template_kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if video_decode_backend == "processor" and num_frames is not None:
        video_kwargs["num_frames"] = num_frames
        video_kwargs["fps"] = None
        video_kwargs["do_sample_frames"] = True
    elif video_decode_backend == "processor" and fps is not None:
        video_kwargs["fps"] = fps
        video_kwargs["num_frames"] = None
        video_kwargs["do_sample_frames"] = True
    if min_pixels is not None or max_pixels is not None:
        video_processor = getattr(processor, "video_processor", None)
        default_size = getattr(video_processor, "size", {}) or {}
        default_min = default_size.get("shortest_edge") if isinstance(default_size, dict) else getattr(default_size, "shortest_edge", None)
        default_max = default_size.get("longest_edge") if isinstance(default_size, dict) else getattr(default_size, "longest_edge", None)
        if min_pixels is None and default_min is None:
            raise ValueError("min_pixels was omitted and processor.video_processor.size has no shortest_edge default.")
        if max_pixels is None and default_max is None:
            raise ValueError("max_pixels was omitted and processor.video_processor.size has no longest_edge default.")
        shortest_edge = int(min_pixels if min_pixels is not None else default_min)
        longest_edge = int(max_pixels if max_pixels is not None else default_max)
        video_kwargs["size"] = {"shortest_edge": shortest_edge, "longest_edge": longest_edge}

    prompt = processor.apply_chat_template(messages, **template_kwargs)
    call_kwargs: Dict[str, Any] = {
        "text": prompt,
        "videos": [[video_input]],
        "return_tensors": "pt",
    }
    if return_metadata:
        call_kwargs["return_metadata"] = True
    if video_kwargs:
        call_kwargs["videos_kwargs"] = video_kwargs

    inputs = dict(processor(**call_kwargs))
    if return_metadata and metadata is not None:
        inputs["video_metadata"] = [metadata]
    return inputs, messages


def strip_model_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in inputs.items() if k not in DROP_MODEL_INPUT_KEYS}


def move_inputs_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def clone_tensor_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def generate_text(
    model: Any,
    processor: Any,
    inputs: Dict[str, Any],
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: float = 0.0,
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    model_inputs = strip_model_inputs(inputs)
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, **gen_kwargs)

    input_len = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    text = decode_generated_outputs(processor, generated_ids)["text"]
    return text, output_ids, generated_ids


def decode_generated_outputs(processor: Any, generated_ids: torch.Tensor) -> Dict[str, str]:
    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    full_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return {
        "text": text,
        "full_text": full_text,
        "extracted_answer": extract_yes_no_answer(text) or extract_yes_no_answer(full_text) or "",
    }


def get_nested_attr(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not hasattr(cur, part):
            raise AttributeError(dotted)
        cur = getattr(cur, part)
    return cur


def find_decoder_layers(model: Any) -> Tuple[str, Sequence[Any]]:
    candidates = [
        "model.language_model.layers",
        "model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "model.text_model.layers",
        "text_model.layers",
        "base_model.model.language_model.layers",
        "base_model.model.layers",
    ]
    for path in candidates:
        try:
            layers = get_nested_attr(model, path)
        except AttributeError:
            continue
        if len(layers) > 0 and hasattr(layers[0], "self_attn"):
            return path, layers
    raise RuntimeError("Could not locate decoder layers with a self_attn module.")


def first_present_int(values: Iterable[Any]) -> Optional[int]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                continue
        try:
            value_int = int(value)
        except Exception:
            continue
        if value_int >= 0:
            return value_int
    return None


def get_video_token_id(model: Any, processor: Any) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    candidates = [
        getattr(config, "video_token_id", None),
        getattr(text_config, "video_token_id", None),
        getattr(processor, "video_token_id", None),
    ]
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        for token in ("<|video_pad|>", "<|video|>"):
            try:
                candidates.append(tokenizer.convert_tokens_to_ids(token))
            except Exception:
                pass
    token_id = first_present_int(candidates)
    if token_id is None:
        raise RuntimeError("Could not determine the Qwen video token id.")
    return token_id


def get_video_token_positions(inputs: Dict[str, Any], video_token_id: int) -> torch.Tensor:
    input_ids = inputs["input_ids"]
    positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise RuntimeError("No video placeholder tokens found in input_ids.")
    return positions


def infer_grid_shape(num_tokens: int) -> Tuple[int, int]:
    if num_tokens <= 0:
        return 1, 1
    root = int(math.sqrt(num_tokens))
    best = (1, num_tokens)
    best_gap = num_tokens - 1
    for h in range(1, root + 1):
        if num_tokens % h == 0:
            w = num_tokens // h
            gap = abs(w - h)
            if gap < best_gap:
                best = (h, w)
                best_gap = gap
    return best


def build_frame_token_map(
    inputs: Dict[str, Any],
    processor: Any,
    video_positions: torch.Tensor,
) -> Dict[str, Any]:
    total_tokens = int(video_positions.numel())
    grid = inputs.get("video_grid_thw")
    if grid is not None:
        grid_cpu = grid.detach().cpu() if torch.is_tensor(grid) else torch.tensor(grid)
        t, h, w = [int(x) for x in grid_cpu.reshape(-1, 3)[0].tolist()]
    else:
        t, h, w = 1, 1, total_tokens

    video_processor = getattr(processor, "video_processor", None)
    merge_size = int(getattr(video_processor, "merge_size", 1) or 1)
    temporal_patch_size = int(getattr(video_processor, "temporal_patch_size", 1) or 1)

    nominal_tokens_per_bin = max(1, (h * w) // max(1, merge_size * merge_size))
    if t * nominal_tokens_per_bin == total_tokens:
        edges = [i * nominal_tokens_per_bin for i in range(t + 1)]
        heatmap_shape = (max(1, h // merge_size), max(1, w // merge_size))
    else:
        edges_np = np.linspace(0, total_tokens, t + 1).round().astype(int)
        edges = edges_np.tolist()
        first_len = max(1, edges[1] - edges[0]) if len(edges) > 1 else total_tokens
        heatmap_shape = infer_grid_shape(first_len)

    frame_slices = []
    for idx in range(t):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        if end > start:
            frame_slices.append((idx, start, end))

    return {
        "num_bins": len(frame_slices),
        "grid_thw": [t, h, w],
        "merge_size": merge_size,
        "temporal_patch_size": temporal_patch_size,
        "heatmap_shape": list(heatmap_shape),
        "frame_slices": frame_slices,
        "total_video_tokens": total_tokens,
    }


def frame_indices_from_metadata(metadata: Any, num_bins: int, temporal_patch_size: int) -> Optional[List[int]]:
    if metadata is None or num_bins <= 0:
        return None
    if isinstance(metadata, (list, tuple)):
        if not metadata:
            return None
        metadata = metadata[0]
        if isinstance(metadata, (list, tuple)):
            if not metadata:
                return None
            metadata = metadata[0]

    frames_indices = getattr(metadata, "frames_indices", None)
    if frames_indices is None and isinstance(metadata, dict):
        frames_indices = metadata.get("frames_indices")
    if frames_indices is None or len(frames_indices) == 0:
        return None

    groups = np.array_split(np.asarray(frames_indices, dtype=int), num_bins)
    selected: List[int] = []
    for group in groups:
        if group.size == 0:
            continue
        selected.append(int(group[group.size // 2]))
    return selected if selected else None


def load_video_frames(
    video_path: str,
    frame_indices: Optional[Sequence[int]] = None,
    count: Optional[int] = None,
) -> List[np.ndarray]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required for visualization frame extraction.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_indices is None:
        if count is None:
            count = 1
        if total > 0:
            frame_indices = np.linspace(0, max(total - 1, 0), count).round().astype(int).tolist()
        else:
            frame_indices = list(range(count))

    frames: List[np.ndarray] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def attention_metrics(frame_ratios: Sequence[float]) -> Dict[str, float]:
    ratios = np.asarray(frame_ratios, dtype=np.float64)
    ratios = ratios[ratios >= 0]
    if ratios.size == 0 or ratios.sum() <= 0:
        return {
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "coefficient_of_variation": 0.0,
            "max_ratio": 0.0,
            "min_ratio": 0.0,
        }
    ratios = ratios / ratios.sum()
    entropy = float(-(ratios * np.log(ratios + 1e-12)).sum())
    normalized_entropy = float(entropy / math.log(len(ratios))) if len(ratios) > 1 else 1.0
    mean = float(ratios.mean())
    std = float(ratios.std())
    return {
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "coefficient_of_variation": std / mean if mean > 0 else 0.0,
        "max_ratio": float(ratios.max()),
        "min_ratio": float(ratios.min()),
    }
