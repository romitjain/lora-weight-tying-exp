# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Empirical validation experiment testing whether broken LoRA weight tying hurts downstream performance. Trains `Qwen/Qwen2.5-0.5B` on `tatsu-lab/alpaca` under 3 conditions and measures perplexity + structural token generation accuracy.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `peft` is installed from GitHub HEAD (not PyPI) because `ensure_weight_tying` is a recent addition not yet on PyPI.

## Commands

**Training** (~30-45 min per run on 3090/4090, runs sequentially):
```bash
python train.py --run_name broken --seed 42 2>&1 | tee logs/broken.txt
python train.py --run_name fixed --seed 42 2>&1 | tee logs/fixed.txt
python train.py --run_name baseline --seed 42 2>&1 | tee logs/baseline.txt
```

**Generation eval** (only for broken/fixed — baseline has no structural tokens):
```bash
python eval_gen.py --checkpoint_dir ./checkpoints/broken 2>&1 | tee logs/eval_broken.txt
python eval_gen.py --checkpoint_dir ./checkpoints/fixed 2>&1 | tee logs/eval_fixed.txt
```

## Architecture

### Experimental Conditions

| `--run_name` | `modules_to_save` | `ensure_weight_tying` | Purpose |
|---|---|---|---|
| `broken` | `["embed_tokens"]` | `False` | Demonstrates broken tying |
| `fixed` | `["embed_tokens"]` | `True` | Demonstrates correct tying |
| `baseline` | `[]` | N/A | Standard LoRA, no new tokens |

`broken` and `fixed` add 4 structural tokens (`<\|thinking\|>`, `</\|thinking\|>`, `<\|answer\|>`, `</\|answer\|>`) and reformat Alpaca responses to use them. `baseline` trains without token addition.

### `train.py` Flow (7 steps)

1. Load `Qwen/Qwen2.5-0.5B` in bf16
2. Add 4 new special tokens and resize embeddings (skipped for baseline)
3. Load/tokenize 5000 train + 500 eval examples from Alpaca
4. Apply LoRA (`r=16, alpha=32`, targets: `q_proj k_proj v_proj o_proj`)
5. Initial evaluation (perplexity + new-token perplexity)
6. Training loop: 3 epochs, batch=4, grad_accum=4, lr=2e-4, cosine schedule
7. Final eval → save PEFT adapter → reload → merge → save merged model

Checkpoints are saved at `./checkpoints/{run_name}/` (PEFT adapter) and `./checkpoints/{run_name}/merged/` (full merged model).

### `eval_gen.py` Flow

Loads the merged model from `{checkpoint_dir}/merged/`, generates completions for 50 eval prompts with greedy decoding, and checks whether each output matches the structural regex `<|thinking|>...<|answer|>...`. Saves per-example results to `{checkpoint_dir}/generation_results.json`.

### Key Implementation Details

- **Tracking**: `trackio` is imported as `wandb` (drop-in replacement). Project: `lora-weight-tying`.
- **No DataLoader**: batching is done manually via `collate()` which pads to `max_seq_len=512` and sets padding positions to `-100` in labels.
- **Weight-tying diagnostics**: `check_weight_tying()` checks both pointer identity (`data_ptr()`) and value equality (`torch.allclose`). Logs `debug/embed_mean`, `debug/lm_head_mean`, `debug/weight_tied` to trackio.
- **New-token perplexity**: Eval computes a separate perplexity restricted to label positions where the target is one of the 4 new token IDs — this is the key metric for detecting broken tying.
- **Device**: hardcoded to `cuda:0`.
