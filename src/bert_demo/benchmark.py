"""
Benchmark — run BERT-family models and produce a device / memory / timing report.

Runs benchmarks across five task types and prints a summary table showing
which device was used, parameter counts, peak GPU memory, and wall-clock
timings.

Usage:
    python src/bert_demo/benchmark.py
    python src/bert_demo/benchmark.py --device cpu
    python src/bert_demo/benchmark.py --tasks fill-mask sentiment-analysis
    python src/bert_demo/benchmark.py --models bert-base-uncased albert-base-v2
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """Metrics for a single benchmark run."""
    model: str
    task: str
    device_type: str       # "CUDA" or "CPU"
    device_name: str       # e.g. "NVIDIA H100 80GB HBM3"
    params_m: float        # millions of parameters
    load_s: float          # model load wall-clock seconds
    infer_s: float         # inference wall-clock seconds
    peak_gpu_mb: float     # peak CUDA memory (0 when on CPU)
    cpu_rss_delta_mb: float  # RSS delta (approximate, Linux only)
    preview: str           # human-readable sample output
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Device / memory helpers
# ---------------------------------------------------------------------------

class DeviceContext:
    """Resolved device info shared by all benchmarks in a run."""

    def __init__(self, device_arg: str) -> None:
        if device_arg == "cpu" or not torch.cuda.is_available():
            self.pipeline_device = -1
            self.torch_device = torch.device("cpu")
            self.device_type = "CPU"
            self.device_name = "CPU"
            self.on_gpu = False
        else:
            self.pipeline_device = 0
            self.torch_device = torch.device("cuda")
            self.device_type = "CUDA"
            self.device_name = torch.cuda.get_device_name(0)
            self.on_gpu = True


def _gpu_reset() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _gpu_peak_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def _rss_mb() -> float:
    """Current process RSS in MB (Linux only, returns 0 elsewhere)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


# ---------------------------------------------------------------------------
# Sample inputs
# ---------------------------------------------------------------------------

FILL_MASK_TEXT = "The capital of France is {mask}."

QA_CONTEXT = (
    "NVIDIA Corporation was founded on April 5, 1993, by Jensen Huang, "
    "Chris Malachowsky, and Curtis Priem. The company is headquartered in "
    "Santa Clara, California."
)
QA_QUESTION = "Who founded NVIDIA?"

NER_TEXT = (
    "Jensen Huang is the CEO of NVIDIA Corporation in Santa Clara, California."
)

SENTIMENT_TEXT = "This product is absolutely wonderful, I love it!"

EMBEDDING_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy hound.",
    "NVIDIA builds powerful GPUs for AI workloads.",
]


# ---------------------------------------------------------------------------
# Individual benchmark runners
# ---------------------------------------------------------------------------

def _bench_fill_mask(model_name: str, ctx: DeviceContext) -> Result:
    _gpu_reset()
    rss_before = _rss_mb()

    t0 = time.perf_counter()
    pipe = hf_pipeline("fill-mask", model=model_name, device=ctx.pipeline_device)
    load_s = time.perf_counter() - t0

    text = FILL_MASK_TEXT.replace("{mask}", pipe.tokenizer.mask_token)

    t1 = time.perf_counter()
    results = pipe(text, top_k=3)
    infer_s = time.perf_counter() - t1

    params = _count_params(pipe.model)
    gpu_mb = _gpu_peak_mb() if ctx.on_gpu else 0.0
    rss_delta = _rss_mb() - rss_before

    preds = results if isinstance(results[0], dict) else results[0]
    preview_preds = " | ".join(
        f"'{p['token_str'].strip()}' ({p['score']:.3f})" for p in preds[:3]
    )
    preview = f"Input : {text}\n    Top-3: {preview_preds}"

    del pipe
    return Result(
        model_name, "fill-mask", ctx.device_type, ctx.device_name,
        params, load_s, infer_s, gpu_mb, rss_delta, preview,
    )


def _bench_qa(model_name: str, ctx: DeviceContext) -> Result:
    _gpu_reset()
    rss_before = _rss_mb()

    t0 = time.perf_counter()
    pipe = hf_pipeline(
        "question-answering", model=model_name, device=ctx.pipeline_device,
    )
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = pipe(question=QA_QUESTION, context=QA_CONTEXT)
    infer_s = time.perf_counter() - t1

    params = _count_params(pipe.model)
    gpu_mb = _gpu_peak_mb() if ctx.on_gpu else 0.0
    rss_delta = _rss_mb() - rss_before

    preview = (
        f"Q: {QA_QUESTION}\n"
        f"    A: '{result['answer']}' (confidence: {result['score']:.3f})"
    )

    del pipe
    return Result(
        model_name, "question-answering", ctx.device_type, ctx.device_name,
        params, load_s, infer_s, gpu_mb, rss_delta, preview,
    )


def _bench_ner(model_name: str, ctx: DeviceContext) -> Result:
    _gpu_reset()
    rss_before = _rss_mb()

    t0 = time.perf_counter()
    pipe = hf_pipeline(
        "ner", model=model_name, device=ctx.pipeline_device,
        aggregation_strategy="simple",
    )
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    results = pipe(NER_TEXT)
    infer_s = time.perf_counter() - t1

    params = _count_params(pipe.model)
    gpu_mb = _gpu_peak_mb() if ctx.on_gpu else 0.0
    rss_delta = _rss_mb() - rss_before

    entities = ", ".join(
        f"{e['word']} [{e['entity_group']}]" for e in results
    )
    preview = f"Input : {NER_TEXT}\n    Entities: {entities}"

    del pipe
    return Result(
        model_name, "ner", ctx.device_type, ctx.device_name,
        params, load_s, infer_s, gpu_mb, rss_delta, preview,
    )


def _bench_sentiment(model_name: str, ctx: DeviceContext) -> Result:
    _gpu_reset()
    rss_before = _rss_mb()

    t0 = time.perf_counter()
    pipe = hf_pipeline(
        "sentiment-analysis", model=model_name, device=ctx.pipeline_device,
    )
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = pipe(SENTIMENT_TEXT)
    infer_s = time.perf_counter() - t1

    params = _count_params(pipe.model)
    gpu_mb = _gpu_peak_mb() if ctx.on_gpu else 0.0
    rss_delta = _rss_mb() - rss_before

    r = result[0]
    preview = f"Input : {SENTIMENT_TEXT}\n    Label: {r['label']} ({r['score']:.3f})"

    del pipe
    return Result(
        model_name, "sentiment-analysis", ctx.device_type, ctx.device_name,
        params, load_s, infer_s, gpu_mb, rss_delta, preview,
    )


def _bench_embeddings(model_name: str, ctx: DeviceContext) -> Result:
    _gpu_reset()
    rss_before = _rss_mb()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(ctx.torch_device)
    model.eval()
    load_s = time.perf_counter() - t0

    encoded = tokenizer(
        EMBEDDING_SENTENCES, padding=True, truncation=True,
        max_length=512, return_tensors="pt",
    ).to(ctx.torch_device)

    t1 = time.perf_counter()
    with torch.no_grad():
        output = model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    emb = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    emb = F.normalize(emb, p=2, dim=1)
    sim_matrix = emb @ emb.T
    infer_s = time.perf_counter() - t1

    params = _count_params(model)
    gpu_mb = _gpu_peak_mb() if ctx.on_gpu else 0.0
    rss_delta = _rss_mb() - rss_before

    n = len(EMBEDDING_SENTENCES)
    best_i, best_j, best_s = 0, 1, -1.0
    for i in range(n):
        for j in range(i + 1, n):
            s = sim_matrix[i][j].item()
            if s > best_s:
                best_i, best_j, best_s = i, j, s

    lines = [f"Encoded {n} sentences -> {emb.shape[1]}d vectors"]
    for i, sent in enumerate(EMBEDDING_SENTENCES):
        lines.append(f"    [{i+1}] {sent}")
    lines.append(
        f"    Most similar: [{best_i+1}] & [{best_j+1}] "
        f"(cosine = {best_s:.3f})"
    )
    preview = "\n".join(lines)

    del model, tokenizer
    return Result(
        model_name, "embeddings", ctx.device_type, ctx.device_name,
        params, load_s, infer_s, gpu_mb, rss_delta, preview,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCH_REGISTRY: List[tuple] = [
    # (task_key, model_name, runner)
    ("fill-mask",           "bert-base-uncased",                                  _bench_fill_mask),
    ("fill-mask",           "bert-large-uncased",                                 _bench_fill_mask),
    ("fill-mask",           "distilbert-base-uncased",                            _bench_fill_mask),
    ("fill-mask",           "albert-base-v2",                                     _bench_fill_mask),
    ("question-answering",  "deepset/bert-base-uncased-squad2",                   _bench_qa),
    ("ner",                 "dslim/bert-base-NER",                                _bench_ner),
    ("sentiment-analysis",  "nlptown/bert-base-multilingual-uncased-sentiment",   _bench_sentiment),
    ("sentiment-analysis",  "distilbert-base-uncased-finetuned-sst-2-english",    _bench_sentiment),
    ("embeddings",          "bert-base-uncased",                                  _bench_embeddings),
]


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

SEP = "-" * 105

def _print_system_info(ctx: DeviceContext) -> None:
    print()
    print("=" * 105)
    print("  BERT-FAMILY MODEL BENCHMARK")
    print("=" * 105)
    print(f"  Device type : {ctx.device_type}")
    print(f"  Device name : {ctx.device_name}")
    if ctx.on_gpu:
        total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)
        print(f"  GPU memory  : {total:,.0f} MB total")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA avail. : {torch.cuda.is_available()}")
    print("=" * 105)
    print()


def _print_progress(idx: int, total: int, task: str, model: str) -> None:
    print(f"[{idx}/{total}] {task:22s} {model}")


def _print_summary_table(results: List[Result]) -> None:
    print()
    print("=" * 105)
    print("  SUMMARY")
    print("=" * 105)

    header = (
        f"  {'#':<3} {'Model':<52} {'Task':<22} {'Params':>7} "
        f"{'Memory':>9} {'Load':>7} {'Infer':>7}"
    )
    units = (
        f"  {'':3} {'':52} {'':22} {'(M)':>7} "
        f"{'(MB)':>9} {'(s)':>7} {'(s)':>7}"
    )
    print(header)
    print(units)
    print("  " + "-" * 101)

    for i, r in enumerate(results, 1):
        if r.error:
            print(f"  {i:<3} {r.model:<52} {r.task:<22}  ** ERROR: {r.error}")
            continue
        mem = r.peak_gpu_mb if r.peak_gpu_mb > 0 else r.cpu_rss_delta_mb
        mem_label = f"{mem:>8.1f}" if mem > 0 else "     n/a"
        print(
            f"  {i:<3} {r.model:<52} {r.task:<22} "
            f"{r.params_m:>7.1f} {mem_label} {r.load_s:>7.2f} {r.infer_s:>7.4f}"
        )

    print("=" * 105)

    # Memory note
    has_gpu = any(r.peak_gpu_mb > 0 for r in results if not r.error)
    if has_gpu:
        print("  Memory = peak CUDA memory allocated (torch.cuda.max_memory_allocated)")
    else:
        print("  Memory = process RSS delta (approximate, Linux only)")
    print()


def _print_detailed_outputs(results: List[Result]) -> None:
    print(SEP)
    print("  DETAILED OUTPUTS")
    print(SEP)
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r.model}  ({r.task})")
        if r.error:
            print(f"    ERROR: {r.error}")
        else:
            for line in r.preview.split("\n"):
                print(f"    {line}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark BERT-family models — device, memory, and timing report",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Force device (default: auto-detect GPU, fall back to CPU)",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK",
        help="Only run these task types (e.g. fill-mask sentiment-analysis)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Only run benchmarks whose model name contains one of these substrings",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx = DeviceContext(args.device)

    # Filter benchmarks
    benchmarks = BENCH_REGISTRY
    if args.tasks:
        benchmarks = [(t, m, fn) for t, m, fn in benchmarks if t in args.tasks]
    if args.models:
        benchmarks = [
            (t, m, fn) for t, m, fn in benchmarks
            if any(sub in m for sub in args.models)
        ]

    if not benchmarks:
        print("No benchmarks matched the given --tasks / --models filters.")
        sys.exit(1)

    _print_system_info(ctx)

    results: List[Result] = []
    total = len(benchmarks)

    for idx, (task, model_name, runner) in enumerate(benchmarks, 1):
        _print_progress(idx, total, task, model_name)
        try:
            r = runner(model_name, ctx)
            r.device_type = ctx.device_type
            r.device_name = ctx.device_name
            results.append(r)
            print(f"       -> done  (load {r.load_s:.2f}s, infer {r.infer_s:.4f}s)")
        except Exception as exc:
            print(f"       -> ERROR: {exc}")
            results.append(Result(
                model=model_name, task=task,
                device_type=ctx.device_type, device_name=ctx.device_name,
                params_m=0, load_s=0, infer_s=0,
                peak_gpu_mb=0, cpu_rss_delta_mb=0,
                preview="", error=str(exc),
            ))

    _print_summary_table(results)
    _print_detailed_outputs(results)


if __name__ == "__main__":
    main()
