"""
Demo 1 — Masked Language Modeling with bert-base-uncased.

Replace one or more [MASK] tokens in a sentence and rank the top-k candidates.

Usage:
    python src/bert_demo/masked_lm.py
    python src/bert_demo/masked_lm.py --text "Paris is the [MASK] of France." --top-k 3
"""

import argparse

import torch
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT fill-mask demo")
    parser.add_argument(
        "--text",
        default="The capital of France is [MASK].",
        help="Sentence containing one or more [MASK] tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of candidate predictions to show (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        device_label = torch.cuda.get_device_name(0)
        device = 0
    else:
        device_label = "CPU (no CUDA device found)"
        device = -1

    print(f"Device : {device_label}")
    print("Loading: bert-base-uncased …")

    fill_mask = pipeline(
        "fill-mask",
        model="bert-base-uncased",
        device=device,
    )

    print(f"\nInput  : {args.text}")
    print(f"\nTop {args.top_k} predictions:")

    results = fill_mask(args.text, top_k=args.top_k)

    # The pipeline returns a list of dicts when there is a single mask token.
    # Normalise to a flat list so we handle the single-mask case uniformly.
    if results and isinstance(results[0], dict):
        predictions = results
    else:
        predictions = results[0]

    for rank, pred in enumerate(predictions, start=1):
        score = pred["score"]
        token = pred["token_str"].strip()
        sentence = pred["sequence"]
        print(f"  {rank}. [{score:.4f}]  '{token}'")
        print(f"        → {sentence}")


if __name__ == "__main__":
    main()
