"""
Demo 3 — Extractive Question Answering with a BERT-base-uncased model.

Uses deepset/bert-base-uncased-squad2 (BERT base fine-tuned on SQuAD 2.0)
to locate the answer span inside a provided context paragraph.

Usage:
    python src/bert_demo/qa.py
    python src/bert_demo/qa.py --question "Who founded NVIDIA?" \
        --context "NVIDIA was founded in 1993 by Jensen Huang."
"""

import argparse
import textwrap
from typing import List, TypedDict

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class Example(TypedDict):
    context: str
    question: str


DEFAULT_EXAMPLES: List[Example] = [
    {
        "context": (
            "NVIDIA Corporation was founded on April 5, 1993, by Jensen Huang, "
            "Chris Malachowsky, and Curtis Priem. The company is headquartered in "
            "Santa Clara, California, and designs graphics processing units (GPUs) "
            "as well as system-on-chip units. NVIDIA's GeForce product line is used "
            "in gaming, while its Tesla and H100 lines target data-centre AI workloads."
        ),
        "question": "Who founded NVIDIA?",
    },
    {
        "context": (
            "BERT (Bidirectional Encoder Representations from Transformers) was "
            "proposed in the paper 'BERT: Pre-training of Deep Bidirectional "
            "Transformers for Language Understanding' by Jacob Devlin, Ming-Wei "
            "Chang, Kenton Lee, and Kristina Toutanova. It was released in "
            "October 2018 and pre-trained on English Wikipedia and BookCorpus "
            "using masked language modelling and next-sentence prediction objectives."
        ),
        "question": "What datasets was BERT pre-trained on?",
    },
    {
        "context": (
            "The NGC (NVIDIA GPU Cloud) catalogue provides GPU-optimised containers, "
            "pre-trained models, and Helm charts. The PyTorch NGC container includes "
            "CUDA, cuDNN, NCCL, and a pre-built PyTorch wheel so that researchers "
            "can start training immediately without compiling dependencies from source. "
            "Access requires a free NVIDIA developer account and an NGC API key."
        ),
        "question": "What do you need to access the NGC catalogue?",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT extractive QA demo")
    parser.add_argument("--question", default=None, help="Question to answer")
    parser.add_argument("--context", default=None, help="Context paragraph")
    return parser.parse_args()


def print_result(example: Example, answer: dict, index: int) -> None:
    wrap = textwrap.fill
    indent = "    "
    print(f"\n{'─' * 60}")
    print(f"Example {index}")
    print(f"{'─' * 60}")
    print(f"Context :\n{wrap(example['context'], width=70, initial_indent=indent, subsequent_indent=indent)}")
    print(f"Question: {example['question']}")
    print(f"Answer  : '{answer['answer']}'  (confidence: {answer['score']:.4f})")


def extract_answer(
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    question: str,
    context: str,
    device: torch.device,
) -> dict:
    """Run extractive QA and return {'answer': str, 'score': float}."""
    inputs = tokenizer(
        question, context,
        return_tensors="pt", truncation=True, max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)

    start_idx = torch.argmax(start_probs).item()
    end_idx = torch.argmax(end_probs).item()

    if end_idx < start_idx:
        return {"answer": "", "score": 0.0}

    answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    score = (start_probs[start_idx] * end_probs[end_idx]).item()
    return {"answer": answer, "score": score}


def main() -> None:
    args = parse_args()

    if args.question and args.context:
        examples: List[Example] = [{"question": args.question, "context": args.context}]
    elif args.question or args.context:
        raise ValueError("Provide both --question and --context together.")
    else:
        examples = DEFAULT_EXAMPLES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device : {device_label}")
    print("Loading: deepset/bert-base-uncased-squad2 …")

    model_name = "deepset/bert-base-uncased-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    model.eval()

    for i, example in enumerate(examples, start=1):
        result = extract_answer(model, tokenizer, example["question"], example["context"], device)
        print_result(example, result, i)

    print(f"\n{'─' * 60}")


if __name__ == "__main__":
    main()
