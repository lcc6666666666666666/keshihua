# VideoHallucer Results Summary

## Overall

| Model | Weighted Pair Acc | Weighted Basic Acc | Weighted Halluc Acc | Macro Pair Acc | Macro Basic Acc | Macro Halluc Acc | Pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | 57.04 | 76.61 | 76.00 | 56.86 | 74.32 | 78.08 | 1150 |
| Qwen3-VL-8B-Thinking | 59.48 | 77.39 | 77.22 | 59.82 | 76.15 | 78.82 | 1150 |
| FAVOR0.5-8B | 53.91 | 66.52 | 82.09 | 54.71 | 66.02 | 83.55 | 1150 |

## By Subset

| Model | Subset | Pairs | Pair Acc | Basic Acc | Halluc Acc | Errors | Missing Video |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B-Instruct | obj_rel | 200 | 65.00 | 81.00 | 79.50 | 0 | 0 |
| Qwen3-VL-8B-Instruct | temporal | 176 | 56.82 | 81.82 | 68.18 | 0 | 0 |
| Qwen3-VL-8B-Instruct | semantic | 200 | 73.50 | 88.50 | 84.00 | 0 | 0 |
| Qwen3-VL-8B-Instruct | interaction | 124 | 24.19 | 37.90 | 73.39 | 0 | 0 |
| Qwen3-VL-8B-Instruct | fact | 200 | 35.50 | 78.50 | 52.00 | 0 | 0 |
| Qwen3-VL-8B-Instruct | nonfact | 200 | 71.00 | 78.50 | 91.50 | 0 | 0 |
| Qwen3-VL-8B-Instruct | factdet | 50 | 72.00 | 74.00 | 98.00 | 0 | 0 |
| Qwen3-VL-8B-Thinking | obj_rel | 200 | 71.50 | 78.50 | 91.50 | 0 | 0 |
| Qwen3-VL-8B-Thinking | temporal | 176 | 68.18 | 84.09 | 75.57 | 0 | 0 |
| Qwen3-VL-8B-Thinking | semantic | 200 | 76.00 | 86.00 | 89.00 | 0 | 0 |
| Qwen3-VL-8B-Thinking | interaction | 124 | 18.55 | 31.45 | 74.19 | 0 | 0 |
| Qwen3-VL-8B-Thinking | fact | 200 | 33.50 | 82.50 | 42.50 | 0 | 0 |
| Qwen3-VL-8B-Thinking | nonfact | 200 | 69.00 | 82.50 | 85.00 | 0 | 0 |
| Qwen3-VL-8B-Thinking | factdet | 50 | 82.00 | 88.00 | 94.00 | 0 | 0 |
| FAVOR0.5-8B | obj_rel | 200 | 58.50 | 65.00 | 88.50 | 0 | 0 |
| FAVOR0.5-8B | temporal | 176 | 64.20 | 67.05 | 87.50 | 0 | 0 |
| FAVOR0.5-8B | semantic | 200 | 73.50 | 85.50 | 87.50 | 0 | 0 |
| FAVOR0.5-8B | interaction | 124 | 17.74 | 26.61 | 79.84 | 0 | 0 |
| FAVOR0.5-8B | fact | 200 | 31.00 | 68.00 | 54.50 | 0 | 0 |
| FAVOR0.5-8B | nonfact | 200 | 60.00 | 68.00 | 91.00 | 0 | 0 |
| FAVOR0.5-8B | factdet | 50 | 78.00 | 82.00 | 96.00 | 0 | 0 |
