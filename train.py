"""
Empirical validation of LoRA weight tying behavior.

Compares 3 conditions on Qwen2.5-0.5B + Alpaca:
  - broken:   modules_to_save=["embed_tokens"], ensure_weight_tying=False
  - fixed:    modules_to_save=["embed_tokens"], ensure_weight_tying=True
  - baseline: standard LoRA (no embedding changes)

Usage:
    python train.py --run_name broken --seed 42
    python train.py --run_name fixed --seed 42
    python train.py --run_name baseline --seed 42
"""

import os
import re
import torch
import argparse
import trackio as wandb
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, PeftModel, get_peft_model

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
NEW_TOKENS = ["<|thinking|>", "</|thinking|>", "<|answer|>", "</|answer|>"]
MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------
# Data formatting
# -------------------------------------------------------------------
def format_example(example, use_structural_tokens=True):
    """Format an Alpaca example into a training string."""
    instruction = example["instruction"]
    inp = example.get("input", "") or ""
    output = example["output"]

    prompt = f"### Instruction:\n{instruction}"
    if inp.strip():
        prompt += f"\n\n### Input:\n{inp}"
    prompt += "\n\n### Response:\n"

    if use_structural_tokens and output.strip():
        # Split at first sentence boundary → thinking / answer
        parts = re.split(r"(?<=[.!?])\s+", output.strip(), maxsplit=1)
        if len(parts) == 2:
            thinking, answer = parts
        else:
            # Single sentence: duplicate into both slots
            thinking = parts[0]
            answer = parts[0]
        response = (
            f"<|thinking|>{thinking}</|thinking|>"
            f"<|answer|>{answer}</|answer|>"
        )
    else:
        response = output

    return prompt + response


# -------------------------------------------------------------------
# Weight-tying diagnostics (mirrors the blog gist)
# -------------------------------------------------------------------
@torch.no_grad()
def check_weight_tying(model, adapter_name="default", step=None):
    """Print and optionally log weight-tying status."""
    emb = model.get_input_embeddings()
    lm = model.get_output_embeddings()

    emb_mean = emb.weight.mean().item()
    lm_mean = lm.weight.mean().item()

    same_ptr = emb.weight.data_ptr() == lm.weight.data_ptr()
    values_close = torch.allclose(emb.weight, lm.weight)
    tied = same_ptr and values_close

    status = "tied" if tied else "BROKEN"
    print(f"  [Weight Tying] embed={emb_mean:.4e}  lm_head={lm_mean:.4e}  → {status}")

    metrics = {
        "debug/embed_mean": emb_mean,
        "debug/lm_head_mean": lm_mean,
        "debug/weight_tied": 1.0 if tied else 0.0,
    }
    if step is not None:
        wandb.log(metrics, step=step)

    return tied


# -------------------------------------------------------------------
# Collation (manual batching — no DataLoader needed)
# -------------------------------------------------------------------
def collate(batch, pad_token_id, max_seq_len):
    """Pad a list of tokenized examples into a batch dict."""
    input_ids, attention_masks = [], []

    for item in batch:
        ids = item["input_ids"][:max_seq_len]
        pad_len = max_seq_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_masks)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # ignore padding in loss

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, eval_data, pad_token_id, new_token_ids, device, max_seq_len):
    """Compute eval loss, perplexity, and new-token-specific perplexity."""
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    total_loss, total_tokens = 0.0, 0
    new_loss, new_count = 0.0, 0
    eval_batch_size = 4

    for i in range(0, len(eval_data), eval_batch_size):
        batch = collate(
            eval_data[i : i + eval_batch_size], pad_token_id, max_seq_len
        )
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

        # Standard causal-LM shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        flat_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        valid = shift_labels != -100
        total_loss += flat_loss[valid].sum().item()
        total_tokens += valid.sum().item()

        # Loss restricted to new-token positions
        if new_token_ids:
            new_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
            for tid in new_token_ids:
                new_mask |= (shift_labels == tid)
            new_mask &= valid
            if new_mask.any():
                new_loss += flat_loss[new_mask].sum().item()
                new_count += new_mask.sum().item()

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    metrics = {"eval/loss": avg_loss, "eval/perplexity": ppl}

    if new_count > 0:
        avg_new = new_loss / new_count
        metrics["eval/new_token_loss"] = avg_new
        metrics["eval/new_token_perplexity"] = torch.exp(torch.tensor(avg_new)).item()
        metrics["eval/new_token_count"] = new_count

    return metrics


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_name",
        required=True,
        choices=["broken", "fixed", "baseline"],
    )
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--num_train", type=int, default=5000)
    parser.add_argument("--num_eval", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f"./checkpoints/{args.run_name}"

    set_seed(args.seed)
    device = "cuda:0"
    use_structural = args.run_name != "baseline"

    # ---- Tracking ----
    wandb.init(project="lora-weight-tying", name=args.run_name)

    print(f"\n{'=' * 60}")
    print(f"  Run: {args.run_name}")
    print(f"{'=' * 60}")

    # ---- 1. Load model & tokenizer ----
    print("\n[1/7] Loading model and tokenizer …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  tie_word_embeddings = {model.config.tie_word_embeddings}")
    check_weight_tying(model)

    # ---- 2. Add new tokens (skip for baseline) ----
    new_token_ids = []
    if use_structural:
        print(f"\n[2/7] Adding {len(NEW_TOKENS)} new special tokens …")
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": NEW_TOKENS}
        )
        model.resize_token_embeddings(len(tokenizer))
        new_token_ids = tokenizer.convert_tokens_to_ids(NEW_TOKENS)
        print(f"  Added {num_added} tokens → vocab size {len(tokenizer)}")
        print(f"  Token IDs: {dict(zip(NEW_TOKENS, new_token_ids))}")
        check_weight_tying(model)
    else:
        print("\n[2/7] Baseline — no token addition")

    # ---- 3. Load & tokenize Alpaca ----
    print("\n[3/7] Loading Alpaca dataset …")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    def tokenize_examples(indices, structural):
        examples = []
        for i in indices:
            text = format_example(ds[int(i)], use_structural_tokens=structural)
            enc = tokenizer(text, truncation=True, max_length=args.max_seq_len)
            examples.append({"input_ids": enc["input_ids"]})
        return examples

    train_data = tokenize_examples(range(args.num_train), use_structural)
    eval_data = tokenize_examples(
        range(args.num_train, args.num_train + args.num_eval), use_structural
    )
    print(f"  Train: {len(train_data)}  Eval: {len(eval_data)}")

    # ---- 4. Apply LoRA ----
    print(f"\n[4/7] Applying LoRA ({args.run_name} config) …")
    modules_to_save = ["embed_tokens"] if use_structural else []
    ensure_tying = args.run_name == "fixed"

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=modules_to_save,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        ensure_weight_tying=ensure_tying,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    check_weight_tying(model)

    # ---- Optimizer & scheduler ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    steps_per_epoch = len(train_data) // args.batch_size
    total_opt_steps = (steps_per_epoch * args.num_epochs) // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, total_opt_steps
    )
    print(f"  Steps/epoch: {steps_per_epoch}  Total opt steps: {total_opt_steps}")

    # ---- 5. Initial eval ----
    print("\n[5/7] Initial evaluation …")
    init_metrics = evaluate(
        model, eval_data, tokenizer.pad_token_id, new_token_ids, device, args.max_seq_len
    )
    wandb.log(init_metrics, step=0)
    print(f"  Eval ppl: {init_metrics['eval/perplexity']:.2f}")
    if "eval/new_token_perplexity" in init_metrics:
        print(f"  New-token ppl: {init_metrics['eval/new_token_perplexity']:.2f}")

    # ---- 6. Training loop ----
    print(f"\n[6/7] Training for {args.num_epochs} epochs …")
    model.train()
    global_step = 0
    opt_step = 0

    for epoch in range(args.num_epochs):
        indices = torch.randperm(len(train_data)).tolist()
        pbar = tqdm(
            range(0, len(indices) - args.batch_size + 1, args.batch_size),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
        )
        epoch_loss, epoch_count = 0.0, 0

        for batch_start in pbar:
            batch_idx = indices[batch_start : batch_start + args.batch_size]
            batch = collate(
                [train_data[j] for j in batch_idx],
                tokenizer.pad_token_id,
                args.max_seq_len,
            )
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()

            epoch_loss += out.loss.item()
            epoch_count += 1
            global_step += 1

            # Optimizer step after grad_accum micro-steps
            if global_step % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                wandb.log(
                    {
                        "train/loss": out.loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm.item(),
                        "train/epoch": epoch + batch_start / len(indices),
                    },
                    step=opt_step,
                )
                pbar.set_postfix(loss=f"{out.loss.item():.4f}", opt=opt_step)

                # Periodic eval
                if opt_step > 0 and opt_step % args.eval_every == 0:
                    eval_m = evaluate(
                        model,
                        eval_data,
                        tokenizer.pad_token_id,
                        new_token_ids,
                        device,
                        args.max_seq_len,
                    )
                    wandb.log(eval_m, step=opt_step)
                    check_weight_tying(model, step=opt_step)
                    msg = f"\n  [Eval @ {opt_step}] ppl={eval_m['eval/perplexity']:.2f}"
                    if "eval/new_token_perplexity" in eval_m:
                        msg += f"  new_tok_ppl={eval_m['eval/new_token_perplexity']:.2f}"
                    print(msg)

        print(f"  Epoch {epoch + 1} avg loss: {epoch_loss / max(epoch_count, 1):.4f}")

    # ---- 7. Final eval, save, reload, merge ----
    print(f"\n[7/7] Final evaluation & save …")
    final_m = evaluate(
        model, eval_data, tokenizer.pad_token_id, new_token_ids, device, args.max_seq_len
    )
    wandb.log({f"final/{k.split('/')[-1]}": v for k, v in final_m.items()}, step=opt_step)
    print(f"  Final ppl: {final_m['eval/perplexity']:.2f}")
    if "eval/new_token_perplexity" in final_m:
        print(f"  Final new-token ppl: {final_m['eval/new_token_perplexity']:.2f}")
    check_weight_tying(model, step=opt_step)

    # Save adapter
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"  Adapter saved to {args.save_dir}")

    # Reload & merge — verify weight tying survives the round-trip
    print("\n  Reloading …")
    reload_base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    if use_structural:
        reload_base.resize_token_embeddings(len(tokenizer))
    reload_peft = PeftModel.from_pretrained(reload_base, args.save_dir)
    print("  After reload:")
    check_weight_tying(reload_peft)

    print("  After merge:")
    merged = reload_peft.merge_and_unload()
    merged.eval()
    check_weight_tying(merged)

    # Save merged model for generation eval
    merged_dir = os.path.join(args.save_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"  Merged model saved to {merged_dir}")

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
