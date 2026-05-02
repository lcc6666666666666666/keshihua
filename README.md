# FAVOR VideoHallucer + Frame Attention Visualization

This project is a Qwen3-VL/FAVOR adaptation. The original `DTR-main/run_inference_dtr.py`
is useful as a reference for VideoHallucer file layout and frame-level attention ratios,
but it cannot be used directly for FAVOR because it is hard-wired to Video-LLaVA,
LLaMA attention modules, and Video-LLaVA `<image>` frame tokens.

## Files

- `favor_utils.py`: shared Qwen3-VL loading, VideoHallucer scoring, video token mapping.
- `run_favor_videohallucer.py`: runs FAVOR on VideoHallucer and writes DTR-compatible `*_predictions.json`.
- `select_cases.py`: selects correct or wrong cases from prediction JSON files.
- `select_contrast_cases.py`: selects cases where Instruct/Thinking are wrong and FAVOR is correct.
- `visualize_frame_attention.py`: generates DTR-style per-frame/per-temporal-bin attention overlays and ratio bars.
- `visualize_qwen_contrast.py`: generates one paper-style comparison figure for Instruct, Thinking, and FAVOR.

## Install

Use the environment that can already load your local model. The visualization path also needs
`matplotlib` and `opencv-python`.

```bash
pip install transformers accelerate torch matplotlib opencv-python tqdm numpy pillow
```

For Qwen3-VL, use a Transformers version that contains `Qwen3VLForConditionalGeneration`.

## 1. Run Benchmark

Default paths in the scripts:

```text
model_path = /data1/lgy/model/FAVOR0.5-8B
data_dir   = /data1/lgy/eval/video_r1/Evaluation/VideoHallucer
```

So the normal command can be:

```bash
python run_favor_videohallucer.py \
  --output_dir outputs/favor_videohallucer \
  --eval_types obj_rel temporal semantic fact nonfact \
  --num_frames 16 \
  --max_new_tokens 16 \
  --continue_on_error
```

Outputs:

```text
outputs/favor_videohallucer/
  obj_rel_predictions.json
  temporal_predictions.json
  semantic_predictions.json
  fact_predictions.json
  nonfact_predictions.json
  favor_evaluation_results.json
```

Each QA item stores:

```text
predict            # cleaned generated text used by the existing scoring flow
predict_full       # full decoded generation with special tokens kept when possible
extracted_answer   # rule-extracted final yes/no answer
```

For Thinking models, use a larger generation budget if you want to keep the reasoning:

```bash
--max_new_tokens 1024
```

`--num_frames` is the number of raw frames sampled by the Qwen3-VL processor. Qwen3-VL
then groups frames by `temporal_patch_size`; the visualization plots the model's actual
`video_grid_thw[0]` temporal bins. If `temporal_patch_size=2`, `--num_frames 16` usually
produces 8 plotted temporal bins.

## 2. Select Cases

```bash
python select_cases.py \
  --pred_dir outputs/favor_videohallucer \
  --output outputs/selected_cases.json \
  --mode both_correct \
  --side hallucination \
  --max_cases 20
```

Useful modes: `both_correct`, `hallucination_correct`, `basic_correct`, `any_wrong`.

## 3. Visualize Attention

```bash
python visualize_frame_attention.py \
  --case_file outputs/selected_cases.json \
  --case_index 0 \
  --output_dir outputs/attention_viz \
  --num_frames 16 \
  --max_new_tokens 16
```

By default this uses only the last actual decoder layer. For a 12-layer decoder, that is
layer `11`; for a 36-layer decoder, that is layer `35`.

Outputs per case:

```text
*_attention.png  # frame overlay, DTR-style ratios
*_ratios.png     # bar chart against a uniform baseline
*_metrics.json   # entropy, normalized entropy, CV, max/min frame ratio
*_attention.npz  # raw token/frame attention arrays
```

The script generates the answer first, then replays generated non-special tokens with the
prompt cache and records selected decoder-layer attention to Qwen3-VL video placeholder
tokens. It normalizes attention within video tokens and aggregates by `video_grid_thw`
temporal bins, so high normalized entropy and low coefficient of variation indicate a
more even temporal distribution.

You can override this with `--layers 11` or `--layers 35`.

## 4. Three-Model Contrast

To make a figure like the DTR paper, first run the same subset with all three models.
For a quick 100-case search on the temporal split:

```bash
python run_favor_videohallucer.py \
  --model_path /data1/lgy/model/Qwen3-VL-8B-Instruct \
  --output_dir outputs/instruct_100 \
  --eval_types temporal \
  --limit_per_type 100 \
  --num_frames 16 \
  --max_new_tokens 1024 \
  --continue_on_error

python run_favor_videohallucer.py \
  --model_path /data1/lgy/model/Qwen3-VL-8B-Thinking \
  --output_dir outputs/thinking_100 \
  --eval_types temporal \
  --limit_per_type 100 \
  --num_frames 16 \
  --max_new_tokens 1024 \
  --continue_on_error

python run_favor_videohallucer.py \
  --model_path /data1/lgy/model/FAVOR0.5-8B \
  --output_dir outputs/favor_100 \
  --eval_types temporal \
  --limit_per_type 100 \
  --num_frames 16 \
  --max_new_tokens 1024 \
  --continue_on_error
```

Then select cases where both Qwen baselines are wrong and FAVOR is correct:

```bash
python select_contrast_cases.py \
  --instruct_pred_dir outputs/instruct_100 \
  --thinking_pred_dir outputs/thinking_100 \
  --favor_pred_dir outputs/favor_100 \
  --output outputs/contrast_cases_100.json \
  --eval_types temporal \
  --side hallucination \
  --max_cases 20
```

Finally draw the same case for all three models using the last decoder layer:

```bash
python visualize_qwen_contrast.py \
  --case_file outputs/contrast_cases_100.json \
  --case_index 0 \
  --output_dir outputs/qwen_contrast_viz \
  --last_n_layers 1 \
  --num_frames 16 \
  --max_new_tokens 1024
```

The comparison output contains:

```text
*_qwen_contrast_attention.png
*_qwen_contrast_metrics.json
*_qwen3_vl_instruct_attention.npz
*_qwen3_vl_thinking_attention.npz
*_favor_attention.npz
```

Use `*_qwen_contrast_attention.png` for the paper-style figure, and use
`normalized_entropy` plus `coefficient_of_variation` in the metrics JSON to support the
claim that FAVOR is more temporally uniform.
