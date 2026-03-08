# lora-weight-tying-exp

Experiment for the blog [Curious Lora](https://r0m1t.com/curious-lora.html)

## Experiment Results: LoRA Weight Tying

**Model:** Qwen2.5-0.5B
**Dataset:** tatsu-lab/alpaca (5,000 train / 500 eval)
**Hardware:** NVIDIA RTX 3080 Ti
**Seed:** 42

---

## Summary Table

| Metric | `baseline` | `fixed` | `broken` |
|---|---|---|---|
| Initial eval perplexity | 7.16 | 16.55 | 16.55 |
| Final eval perplexity | **4.81** | 5.36 | 5.91 |
| Initial new-token perplexity | N/A | ~2.9 billion | ~2.9 billion |
| Final new-token perplexity | N/A | **1.16** | 3,810.89 |
| Structural token accuracy (strict) | N/A | **49/50 (98%)** | 0/50 (0%) |
| Weight tying intact post-merge | Yes | Yes | **No** |

---

## Per-Condition Breakdown

### baseline

Standard LoRA fine-tuning with no vocabulary extension. No structural tokens added.

| Step | Eval ppl |
|---|---|
| 0 (initial) | 7.16 |
| 200 | 4.84 |
| 400 | 4.81 |
| 600 | 4.79 |
| 800 | 4.80 |
| Final | **4.81** |

| Epoch | Avg train loss |
|---|---|
| 1 | 1.5865 |
| 2 | 1.4887 |
| 3 | 1.4307 |

### fixed (`ensure_weight_tying=True`)

LoRA with 4 new structural tokens added to the vocabulary. `ensure_weight_tying=True` ensures lm_head receives the same gradient updates as embed_tokens throughout training.

| Step | Eval ppl | New-token ppl |
|---|---|---|
| 0 (initial) | 16.55 | ~2,913,696,512 |
| 200 | 4.81 | 1.21 |
| 400 | 5.00 | 1.17 |
| 600 | 4.99 | 1.20 |
| 800 | 5.34 | 1.17 |
| Final | **5.36** | **1.16** |

| Epoch | Avg train loss |
|---|---|
| 1 | 1.6662 |
| 2 | 1.0177 |
| 3 | 0.7179 |

### broken (`ensure_weight_tying=False`)

Same setup as `fixed`, but lm_head is never updated — only embed_tokens receives gradient updates. Weight tying is broken from the first optimizer step.

| Step | Eval ppl | New-token ppl |
|---|---|---|
| 0 (initial) | 16.55 | ~2,913,696,512 |
| 200 | 5.63 | 4,179.75 |
| 400 | 5.67 | 3,780.48 |
| 600 | 5.65 | 4,016.63 |
| 800 | 5.89 | 3,787.11 |
| Final | 5.91 | 3,810.89 |

| Epoch | Avg train loss |
|---|---|
| 1 | 1.8868 |
| 2 | 1.5455 |
| 3 | 1.3807 |

---

## Analysis

### 1. The core bug: lm_head never learns the new tokens

Both `broken` and `fixed` start at the same initial new-token perplexity of ~2.9 billion. This is expected — newly added tokens are randomly initialised, so the model has no idea how to predict them yet.

For `fixed`, new-token perplexity collapses to **1.21 by step 200** (the first eval checkpoint) and stays near 1.16 for the rest of training. The model has essentially memorised the structural format.

For `broken`, new-token perplexity **never drops** — it fluctuates between 3,780 and 4,180 across all three epochs, ending at 3,810. This is because `embed_tokens` (the input side) receives gradient updates and learns the new token embeddings, but `lm_head` (the output side) does not. The model learns to *understand* `<|thinking|>` as input, but its output projection for those token IDs remains at random initialisation. The model literally cannot produce those tokens with any confidence.

### 2. The generation breakdown is total

The structural token accuracy metric (strict end-to-end fullmatch) is the most dramatic result: **0/50 for broken, 49/50 for fixed**.

Looking at the generated outputs for `broken`, the failure modes are varied:
- Some outputs produce `<|thinking|>` but never close it — the model opens the tag, then the lm_head's near-random weights for `</|thinking|>` make it nearly impossible to generate that closing token.
- Some outputs get stuck in repetition loops (e.g., `<|thinking|>She.<|thinking|>She.<|thinking|>...`).
- Some outputs produce garbled multilingual text.
- Some outputs skip structural tokens entirely and generate plain text.

For `fixed`, the model correctly wraps its response in `<|thinking|>...</|thinking|><|answer|>...</|answer|>` in 49 out of 50 cases, with coherent content in both sections.

### 3. Overall perplexity is also hurt

Even on the general (non-structural) eval perplexity, `broken` (5.91) is worse than `fixed` (5.36). The broken weight tying doesn't just hurt the 4 new tokens in isolation — it degrades overall generation quality, likely because the model's output distribution becomes incoherent when a significant fraction of its vocabulary is unpredictable.

`baseline` achieves the best overall perplexity (4.81) because:
- It trains on plain Alpaca text without any structural tokens, so there is no overhead from learning a new response format.
- It uses far fewer trainable parameters (~2.2M vs ~138M for the other two runs, which include `modules_to_save` wrapping the full embedding matrix).

### 4. Training loss is also lower for fixed vs broken

| Epoch | fixed loss | broken loss |
|---|---|---|
| 1 | 1.6662 | 1.8868 |
| 2 | 1.0177 | 1.5455 |
| 3 | 0.7179 | 1.3807 |

`fixed` drives the training loss nearly half that of `broken` by epoch 3 (0.72 vs 1.38). This confirms that the `broken` model is failing to learn the task — the cross-entropy loss on structural token positions stays high because lm_head cannot learn to output them.

### 5. Weight tying survives reload and merge for fixed, not for broken

After saving the PEFT adapter, reloading it, and merging into the base model:
- `fixed`: `embed_tokens` and `lm_head` remain tied (same pointer, same values) at every stage.
- `broken`: The divergence persists through save, reload, and merge. The merged model has `embed_tokens` at a different mean than `lm_head` (5.4121e-05 vs 5.4359e-05), meaning any downstream user loading this merged checkpoint gets a silently broken model with no indication from the config that weight tying has failed.

---

## Conclusion

The experiment provides clear empirical evidence that broken LoRA weight tying is not just a theoretical concern — it causes measurable, severe downstream degradation:

- New-token perplexity stays at ~3,800 (vs 1.16 for the fix) because lm_head never receives updates for the new token IDs.
- Structural generation accuracy (strict fullmatch) collapses from 98% to 0%.
- Overall eval perplexity degrades by ~10% even on non-structural tokens.
- The breakage survives model saving and merging, silently producing a corrupted checkpoint.

Setting `ensure_weight_tying=True` in `LoraConfig` is a one-line fix that fully resolves all of the above.
