"""
Demo 5 — Text Generation with GPT-2 XL (1.5 B parameters).

Loads GPT-2 XL in FP16 on the GPU and generates text from one or more
prompts.  Designed as a heavier workload that is clearly visible in
GPU monitoring tools such as ``nvtop``.

Usage:
    python src/bert_demo/generate.py
    python src/bert_demo/generate.py --prompts "Once upon a time"
    python src/bert_demo/generate.py --max-tokens 300 --runs 3
    python src/bert_demo/generate.py --device cpu
"""

import argparse
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS: List[str] = [
    "The future of artificial intelligence will reshape every industry because",
    "In a world where quantum computers become mainstream, the first thing that changes is",
    "The most important scientific discovery of the 21st century so far has been",
    "When humans finally establish a permanent colony on Mars, the biggest challenge will be",
    "The relationship between consciousness and computation suggests that",
]

MODEL_NAME = "openai-community/gpt2-xl"
SEP = "=" * 70


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 XL text-generation benchmark")
    p.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        metavar="TEXT",
        help="One or more prompts to generate from (defaults to a built-in set of 5)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        dest="max_tokens",
        help="Maximum new tokens to generate per prompt (default: %(default)s)",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=400,
        dest="min_tokens",
        help="Minimum new tokens to generate per prompt (default: %(default)s)",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of prompts to run (default: all prompts)",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Force device (default: auto-detect GPU, fall back to CPU)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve device
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
        dtype = torch.float32
        device_label = "CPU"
        vram_info = ""
    else:
        device = torch.device("cuda")
        dtype = torch.float16
        device_label = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        vram_info = f"  VRAM total : {total_gb:.1f} GB"

    # Resolve prompts
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    if args.runs is not None:
        prompts = prompts[: args.runs]

    print()
    print(SEP)
    print(f"  GPT-2 XL TEXT GENERATION — {len(prompts)} prompt(s)")
    print(SEP)
    print(f"  Device     : {device_label}")
    if vram_info:
        print(vram_info)
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  CUDA avail.: {torch.cuda.is_available()}")
    print(SEP)
    print()

    # Load model
    print(f"Loading {MODEL_NAME} (1.5B params, {dtype})...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    model.eval()
    load_time = time.perf_counter() - t0

    if device.type == "cuda":
        mem_model = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"  Loaded in {load_time:.1f}s  (GPU VRAM: {mem_model:.2f} GB)")
    else:
        print(f"  Loaded in {load_time:.1f}s")
    print()

    # Generate
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attn_mask = torch.ones_like(input_ids)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_tokens,
                min_new_tokens=args.min_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.2,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t1

        n_tokens = output.shape[1] - input_ids.shape[1]
        total_tokens += n_tokens
        total_time += elapsed

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"--- Run {i + 1}/{len(prompts)}  ({n_tokens} tokens in {elapsed:.1f}s = {n_tokens / elapsed:.1f} tok/s) ---")
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print()

    # Summary
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(0) / 1024 ** 3
        peak_info = f"  Peak GPU VRAM    : {peak_gb:.2f} GB"
    else:
        peak_info = ""

    print(SEP)
    print(f"  Total tokens     : {total_tokens}")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  Avg tokens/sec   : {total_tokens / total_time:.1f}")
    if peak_info:
        print(peak_info)
    print(SEP)


if __name__ == "__main__":
    main()
