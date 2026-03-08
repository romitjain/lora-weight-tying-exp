"""
Post-training generation evaluation for LoRA weight-tying experiment.

Loads a merged model, generates completions for eval prompts, and measures
structural token accuracy (do generations correctly include <|thinking|>
and <|answer|> tokens in the right order?).

Usage:
    python eval_gen.py --checkpoint_dir ./checkpoints/broken
    python eval_gen.py --checkpoint_dir ./checkpoints/fixed
"""

import os
import re
import json
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

NEW_TOKENS = ["<|thinking|>", "</|thinking|>", "<|answer|>", "</|answer|>"]
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
STRUCTURAL_RE = re.compile(
    r"^\s*<\|thinking\|>.+?</\|thinking\|>\s*<\|answer\|>.+?</\|answer\|>\s*$",
    re.DOTALL,
)


def format_prompt(example):
    """Build the prompt portion (no response) of an Alpaca example."""
    instruction = example["instruction"]
    inp = example.get("input", "") or ""

    prompt = f"### Instruction:\n{instruction}"
    if inp.strip():
        prompt += f"\n\n### Input:\n{inp}"
    prompt += "\n\n### Response:\n"
    return prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--num_train",
        type=int,
        default=5000,
        help="Number of train examples to skip (eval starts after this index)",
    )
    args = parser.parse_args()

    device = "cuda:0"
    merged_dir = os.path.join(args.checkpoint_dir, "merged")

    print(f"Loading merged model from {merged_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval slice of Alpaca
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    eval_examples = [
        ds[i] for i in range(args.num_train, args.num_train + args.num_samples)
    ]

    print(f"Generating {args.num_samples} completions …\n")

    results = []
    correct = 0

    for i, example in enumerate(eval_examples):
        prompt = format_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Stop at </|answer|> so the model doesn't loop after closing the structure
        stop_token_id = tokenizer.convert_tokens_to_ids("</|answer|>")
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=[tokenizer.eos_token_id, stop_token_id],
            )

        generated = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
        )

        has_structure = bool(STRUCTURAL_RE.match(generated))
        if has_structure:
            correct += 1

        results.append(
            {
                "index": i,
                "instruction": example["instruction"],
                "reference": example["output"],
                "generated": generated,
                "has_structure": has_structure,
            }
        )

        # Print the first 10 for quick inspection
        if i < 10:
            tag = "OK" if has_structure else "MISS"
            print(f"--- Example {i + 1} [{tag}] ---")
            print(f"  Instruction: {example['instruction'][:120]}")
            gen_preview = generated.replace("\n", " ")[:200]
            print(f"  Generated:   {gen_preview}")
            print()

    accuracy = correct / len(results) * 100
    print(f"{'=' * 60}")
    print(f"Structural Token Accuracy: {correct}/{len(results)}  ({accuracy:.1f}%)")
    print(f"{'=' * 60}")

    # Persist results
    out_path = os.path.join(args.checkpoint_dir, "generation_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "accuracy_pct": accuracy,
                "correct": correct,
                "total": len(results),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
