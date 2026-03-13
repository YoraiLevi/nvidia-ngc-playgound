"""
Demo 2 — Sentence Embeddings with bert-base-uncased.

Encodes a set of sentences into fixed-size vectors using mean pooling over the
last hidden state, then prints pairwise cosine similarities so you can see
which sentences BERT considers semantically close.

Usage:
    python src/bert_demo/embeddings.py
    python src/bert_demo/embeddings.py --sentences "Hello world." "Hi there."
"""

import argparse
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_SENTENCES: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy hound.",
    "NVIDIA builds powerful GPUs for AI workloads.",
    "Deep learning requires specialised hardware accelerators.",
    "The weather today is sunny and warm.",
]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings, ignoring padding positions."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT sentence embeddings demo")
    parser.add_argument(
        "--sentences",
        nargs="+",
        default=None,
        metavar="SENTENCE",
        help="Two or more sentences to embed (defaults to the built-in set)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sentences = args.sentences if args.sentences else DEFAULT_SENTENCES

    if len(sentences) < 2:
        raise ValueError("Provide at least two sentences to compare.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device : {device_label}")
    print("Loading: bert-base-uncased …")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    embeddings = mean_pool(output.last_hidden_state, encoded["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    print(f"\nEmbedding shape: {embeddings.shape}  (sentences × hidden_dim)\n")

    print("Pairwise cosine similarity matrix:")
    header = "       " + "".join(f"  [{i+1}]" for i in range(len(sentences)))
    print(header)
    for i in range(len(sentences)):
        row = f"  [{i+1}]  "
        for j in range(len(sentences)):
            sim = torch.dot(embeddings[i], embeddings[j]).item()
            row += f" {sim:+.3f}"
        print(row)

    print("\nSentences:")
    for i, s in enumerate(sentences, start=1):
        print(f"  [{i}] {s}")


if __name__ == "__main__":
    main()
