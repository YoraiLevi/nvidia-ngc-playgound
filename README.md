# NVIDIA NGC — Rootless GPU Containers with Podman

A proof-of-concept for running **GPU-accelerated ML workloads without
sudo** using [Podman](https://podman.io/) and the
[NVIDIA NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
Every `podman run` command below executes as a regular, unprivileged user —
no root, no daemon, no docker group.

The demos use **BERT base uncased** as the workload.
Every step is a single copy-pastable shell command.
No custom container image is built — you work directly with the published NGC container.

---

## Why Podman?

| | Docker | Podman |
|---|---|---|
| **Daemon** | `dockerd` runs as root | Daemonless — direct fork/exec |
| **Default privileges** | Requires `sudo` or `docker` group (root-equivalent) | Rootless by default |
| **GPU passthrough** | `--gpus all` via nvidia-container-runtime | `--device nvidia.com/gpu=all` via CDI |

With Docker, adding a user to the `docker` group grants them
**unrestricted root access** to the host.  Podman eliminates this: every
container run is an unprivileged process owned by your user account.
Combined with the NVIDIA Container Toolkit's CDI (Container Device
Interface) support, GPU workloads run without any privilege escalation.

---

## What you will build

| Demo | Script | What it shows |
|---|---|---|
| Masked Language Modeling | `src/bert_demo/masked_lm.py` | Predict the top-k words for a `[MASK]` token |
| Sentence Embeddings | `src/bert_demo/embeddings.py` | Encode sentences and compare them by cosine similarity |
| Question Answering | `src/bert_demo/qa.py` | Locate an answer span inside a context paragraph |
| **Benchmark Report** | `src/bert_demo/benchmark.py` | Run 14 BERT-family models, report device / memory / timing |
| **GPT-2 XL Generation** | `src/bert_demo/generate.py` | Generate text with a 1.5 B-param LLM — visible in `nvtop` |

---

## Repository layout

```
nvidia-ngc-playground/
├── README.md
├── .env.example            ← template for your NGC API key
├── .gitignore
├── pyproject.toml          ← uv project (transformers, accelerate)
└── src/
    └── bert_demo/
        ├── __init__.py
        ├── masked_lm.py
        ├── embeddings.py
        ├── qa.py
        ├── benchmark.py
        └── generate.py
```

Model weights are downloaded from Hugging Face Hub on first run and
cached in a local `models/` directory that is volume-mounted into the
container, so they are only downloaded once.

---

## Shell variants

The `podman run` commands below use **bash** syntax (`$(pwd)`, `\`).
If you are on **Windows PowerShell** substitute as shown:

| bash | PowerShell |
|---|---|
| `$(pwd)` | `${PWD}` |
| `\` (line continuation) | `` ` `` |
| `export VAR=value` | `$env:VAR = "value"` |

**PowerShell example** — each demo step uses the same pattern, just swap
the line-continuation character:

```powershell
podman run --device nvidia.com/gpu=all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:26.02-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/masked_lm.py'
```

> **Note:** GPU passthrough via CDI is Linux-native.  On Windows, run
> Podman inside WSL 2 where the Linux GPU stack is available.

---

## Prerequisites

Before starting, confirm the following are installed:

- **Podman** — `sudo apt install podman` (Debian/Ubuntu), `sudo dnf install podman` (Fedora/RHEL),
  or see the [Podman install guide](https://podman.io/docs/installation)
- **NVIDIA Container Toolkit** — enables GPU passthrough into containers
  ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- **CDI spec generated** — a one-time setup step (requires sudo once):
  ```bash
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  ```
  Verify it worked: `nvidia-ctk cdi list` should show your GPU(s).
- **NVIDIA driver ≥ 570** — required by the 26.02 NGC container (CUDA 13.1)
- An **NVIDIA GPU** with CUDA support
- A free **NVIDIA developer account** at [developer.nvidia.com](https://developer.nvidia.com)

### Verify rootless GPU access

This is the proof — **no sudo, no daemon, just your user account**:

```bash
echo "I am: $(whoami) (uid=$(id -u))"
podman run --rm --device nvidia.com/gpu=all docker.io/nvidia/cuda:12.0.1-base-ubuntu22.04 nvidia-smi
```

You should see your regular username (not `root`) and `nvidia-smi` output
listing your GPU.  If this works, every command in this tutorial will work.

---

## Step 1 — Get an NGC API key

1. Sign in at [https://ngc.nvidia.com](https://ngc.nvidia.com)
2. Click your account name → **Setup** → **Generate API Key**
3. Copy the key — it is shown only once

Keep it safe; you will use it in the next step and again in Step 13.

---

## Step 2 — Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and replace `<paste-your-key-here>` with your actual key:

```
NGC_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> `.env` is listed in `.gitignore` and will never be committed.

---

## Step 3 — Authenticate Podman with NGC

NGC's container registry requires you to log in once per machine.
The username is always the literal string `$oauthtoken`; the password is
your API key.

```bash
podman login nvcr.io --username '$oauthtoken' --password "$(grep NGC_API_KEY .env | cut -d= -f2)"
```

**PowerShell:**

```powershell
$key = (Get-Content .env | Where-Object { $_ -match "NGC_API_KEY" }) -replace "NGC_API_KEY=", ""
podman login nvcr.io --username '$oauthtoken' --password "$key"
```

A successful login prints `Login Succeeded`.

---

## Step 4 — Pull the NGC PyTorch container

The `26.02-py3` image includes CUDA 13.1, cuDNN, and PyTorch 2.11.
It is ~18 GB; this step only needs to run once.

```bash
podman pull nvcr.io/nvidia/pytorch:26.02-py3
```

---

## Step 5 — Install uv locally

[uv](https://docs.astral.sh/uv/) manages this project's Python
dependencies on your host machine and generates the lock file that the
container will consume.

**Linux / macOS / WSL:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows PowerShell (native):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal (or source the updated profile) so that `uv` is on
`PATH`.

---

## Step 6 — Sync the project

Install the project's Python dependencies on the host and generate
`uv.lock`.  The container will read this lock file to install the same
versions inside.

```bash
uv sync
```

You should see uv resolve and install `transformers` and `accelerate`
into a local `.venv`.  A `uv.lock` file is created (or verified) at the
project root.

> **Note:** `torch` is intentionally absent from `pyproject.toml`.
> The NGC container ships a CUDA-optimised PyTorch build; adding a PyPI
> wheel would replace it with a CPU-only version.
> For local runs without the container, use `uv sync --extra local`.

---

## Step 7 — Create the model cache directory

Model weights are large binary files; store them outside the container on
your host so they persist across runs.

```bash
mkdir -p models
```

The directory is mounted at `/root/.cache/huggingface` inside every
`podman run` command below.  Hugging Face Transformers uses that path as
its default download cache.

---

## Step 8 — Demo 1: Masked Language Modeling

`masked_lm.py` loads **bert-base-uncased** and asks it to predict the
word hidden behind a `[MASK]` token.

**Linux / macOS / WSL:**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/masked_lm.py'
```

**PowerShell:**

```powershell
podman run --device nvidia.com/gpu=all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:26.02-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/masked_lm.py'
```

**Expected output (truncated):**

```
Device : NVIDIA H100 80GB HBM3
Loading: bert-base-uncased …

Input  : The capital of France is [MASK].

Top 5 predictions:
  1. [0.6204]  'paris'
        → the capital of france is paris.
  2. [0.0906]  'lille'
        → the capital of france is lille.
  ...
```

**Try a custom sentence:**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/masked_lm.py --text "NVIDIA makes the world'\''s fastest [MASK]." --top-k 3'
```

---

## Step 9 — Demo 2: Sentence Embeddings

`embeddings.py` encodes five sentences into 768-dimensional vectors using
mean pooling over the last hidden state, then prints a cosine-similarity
matrix.  Semantically similar sentence pairs will be close to 1.0.

**Linux / macOS / WSL:**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/embeddings.py'
```

**PowerShell:**

```powershell
podman run --device nvidia.com/gpu=all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:26.02-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/embeddings.py'
```

**Expected output (truncated):**

```
Device : NVIDIA H100 80GB HBM3
Loading: bert-base-uncased …

Embedding shape: torch.Size([5, 768])  (sentences × hidden_dim)

Pairwise cosine similarity matrix:
         [1]   [2]   [3]   [4]   [5]
  [1]  +1.000 +0.972 +0.891 +0.887 +0.871
  [2]  +0.972 +1.000 +0.882 +0.879 +0.860
  ...
```

Sentences [1] and [2] (*"quick brown fox"* and *"fast auburn fox"*) score
close to 1.0 because they describe the same thing in different words.

**Pass your own sentences:**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/embeddings.py --sentences "I love pizza." "Pizza is my favourite food." "The sky is blue."'
```

---

## Step 10 — Demo 3: Question Answering

`qa.py` uses **deepset/bert-base-uncased-squad2** — a `bert-base-uncased`
checkpoint fine-tuned on SQuAD 2.0 — to locate answer spans inside
context paragraphs.

**Linux / macOS / WSL:**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/qa.py'
```

**PowerShell:**

```powershell
podman run --device nvidia.com/gpu=all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:26.02-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/qa.py'
```

**Expected output (truncated):**

```
Device : NVIDIA H100 80GB HBM3
Loading: deepset/bert-base-uncased-squad2 …

────────────────────────────────────────────────────────────
Example 1
────────────────────────────────────────────────────────────
Context :
    NVIDIA Corporation was founded on April 5, 1993, by Jensen Huang …
Question: Who founded NVIDIA?
Answer  : 'Jensen Huang, Chris Malachowsky, and Curtis Priem'  (confidence: 0.9823)
```

**Ask a custom question:**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/qa.py --question "What year was CUDA released?" --context "NVIDIA introduced CUDA in 2006 as a parallel computing platform."'
```

---

## Step 11 — Benchmark Report: all models at once

`benchmark.py` loads **fourteen BERT-family models** across six task types
and prints a summary table with device info, parameter counts, peak GPU
memory, and wall-clock timings.

**Models included:**

| # | Model | Task |
|---|---|---|
| 1 | `bert-base-uncased` | Masked LM |
| 2 | `bert-large-uncased` | Masked LM |
| 3 | `distilbert-base-uncased` | Masked LM |
| 4 | `albert-base-v2` | Masked LM |
| 5 | `roberta-base` | Masked LM |
| 6 | `microsoft/deberta-v3-base` | Masked LM |
| 7 | `deepset/bert-base-uncased-squad2` | Question Answering |
| 8 | `bert-large-uncased-whole-word-masking-finetuned-squad` | Question Answering |
| 9 | `dslim/bert-base-NER` | Named Entity Recognition |
| 10 | `nlptown/bert-base-multilingual-uncased-sentiment` | Sentiment (5-star) |
| 11 | `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment (binary) |
| 12 | `bert-base-uncased` | Sentence Embeddings |
| 13 | `nvidia/dragon-multiturn-query-encoder` | Sentence Embeddings (NVIDIA NGC) |
| 14 | `facebook/bart-large-mnli` | Zero-Shot Classification |

**Linux / macOS / WSL:**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/benchmark.py'
```

**PowerShell:**

```powershell
podman run --device nvidia.com/gpu=all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:26.02-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/benchmark.py'
```

**Force CPU (for comparison):**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/benchmark.py --device cpu'
```

**Run a subset of tasks or models:**

```bash
# Only fill-mask and sentiment tasks
... python src/bert_demo/benchmark.py --tasks fill-mask sentiment-analysis

# Only models whose name contains "distilbert" or "albert"
... python src/bert_demo/benchmark.py --models distilbert albert
```

**Expected output (truncated):**

```
=============================================================================================================
  BERT-FAMILY MODEL BENCHMARK
=============================================================================================================
  Device type : CUDA
  Device name : NVIDIA H100 80GB HBM3
  GPU memory  : 81,559 MB total
  PyTorch     : 2.6.0a0+ecf3bae43e.nv25.1
  CUDA avail. : True
=============================================================================================================

[1/14] fill-mask               bert-base-uncased
       -> done  (load 1.82s, infer 0.0134s)
[2/14] fill-mask               bert-large-uncased
       -> done  (load 2.41s, infer 0.0158s)
...

=============================================================================================================
  SUMMARY
=============================================================================================================
  #   Model                                                Task                   Params   Memory    Load   Infer
                                                                                    (M)     (MB)      (s)      (s)
  -----------------------------------------------------------------------------------------------------
  1   bert-base-uncased                                    fill-mask               109.5    438.2    1.82  0.0134
  2   bert-large-uncased                                   fill-mask               335.1   1340.6    2.41  0.0158
  3   distilbert-base-uncased                              fill-mask                66.4    265.4    0.97  0.0089
  4   albert-base-v2                                       fill-mask                11.7     46.6    0.71  0.0102
  5   roberta-base                                         fill-mask               124.6    498.6    1.74  0.0128
  6   microsoft/deberta-v3-base                            fill-mask                86.9    347.4    1.53  0.0115
  ...
=============================================================================================================
  Memory = peak CUDA memory allocated (torch.cuda.max_memory_allocated)
```

---

## Step 12 — Demo 5: GPT-2 XL Text Generation (1.5 B params)

`generate.py` loads **GPT-2 XL** (1.5 billion parameters) in FP16 and
generates text from a set of prompts.  This is a significantly heavier
workload than the BERT demos — you will see clear GPU utilisation in
monitoring tools like `nvtop`.

**Linux / macOS / WSL:**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/generate.py'
```

**PowerShell:**

```powershell
podman run --device nvidia.com/gpu=all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:26.02-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/generate.py'
```

**Fewer prompts / shorter output:**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/generate.py --runs 2 --max-tokens 200'
```

**Custom prompt:**

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken && python src/bert_demo/generate.py --prompts "Once upon a time in a galaxy far away"'
```

**Expected output (truncated):**

```
======================================================================
  GPT-2 XL TEXT GENERATION — 5 prompt(s)
======================================================================
  Device     : NVIDIA GB10
  VRAM total : 119.7 GB
  PyTorch    : 2.11.0a0+eb65b36
  CUDA avail.: True
======================================================================

Loading openai-community/gpt2-xl (1.5B params, torch.float16)...
  Loaded in 60.8s  (GPU VRAM: 2.97 GB)

--- Run 1/5  (438 tokens in 10.2s = 43.1 tok/s) ---
The future of artificial intelligence will reshape every industry because
of its capabilities and its applications …

--- Run 2/5  (500 tokens in 10.9s = 46.0 tok/s) ---
In a world where quantum computers become mainstream, the first thing
that changes is that people don't want to do quantum stuff …
...

======================================================================
  Total tokens     : 2223
  Total time       : 48.9s
  Avg tokens/sec   : 45.4
  Peak GPU VRAM    : 3.13 GB
======================================================================
```

---

## Step 13 — Optional: Download the NGC BERT model artifact

NVIDIA also hosts a pre-trained **BERTBaseUncased** checkpoint in the NGC
model registry under `nvidia/nemo/bertbaseuncased`.  You can download it
with the **NGC CLI** and load it through the NeMo framework.

### 13a — Install the NGC CLI inside the container

```bash
podman run --device nvidia.com/gpu=all --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c '
    wget -q https://ngc.nvidia.com/downloads/ngccli_linux.zip -O /tmp/ngccli.zip &&
    unzip -q /tmp/ngccli.zip -d /tmp/ngccli &&
    chmod +x /tmp/ngccli/ngc-cli/ngc &&
    export PATH="/tmp/ngccli/ngc-cli:$PATH" &&
    ngc --version
  '
```

### 13b — Configure the NGC CLI and download the model

Replace `<YOUR_API_KEY>` with the value from your `.env` file.

```bash
podman run --device nvidia.com/gpu=all --rm -it \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/models \
  -w /workspace \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c '
    wget -q https://ngc.nvidia.com/downloads/ngccli_linux.zip -O /tmp/ngccli.zip &&
    unzip -q /tmp/ngccli.zip -d /tmp/ngccli &&
    chmod +x /tmp/ngccli/ngc-cli/ngc &&
    export PATH="/tmp/ngccli/ngc-cli:$PATH" &&
    ngc config set --api-key <YOUR_API_KEY> --format_type ascii &&
    ngc registry model download-version "nvidia/nemo/bertbaseuncased:1.0.0rc1" --dest /models
  '
```

The checkpoint is saved under `models/bertbaseuncased_v1.0.0rc1/`.

### 13c — Load the checkpoint with NeMo (inside the container)

Install NeMo on top of the PyTorch container, then load the downloaded
checkpoint:

```bash
podman run --device nvidia.com/gpu=all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/models \
  -w /workspace \
  nvcr.io/nvidia/pytorch:26.02-py3 \
  bash -c '
    pip install nemo_toolkit[nlp] --quiet &&
    python - <<EOF
import nemo.collections.nlp as nemo_nlp
model = nemo_nlp.models.language_modeling.BERTLMModel.restore_from(
    "/models/bertbaseuncased_v1.0.0rc1/bert-base-uncased.nemo"
)
print("Model config:", model.cfg.model_name)
print("Hidden size :", model.bert_model.config.hidden_size)
EOF
  '
```

---

## Troubleshooting

### `nvidia-ctk cdi list` shows 0 devices

The CDI spec has not been generated.  Run once (requires sudo):

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

Then verify: `nvidia-ctk cdi list` should list your GPU(s).

---

### `Error: CDI device ... not found` when running a container

Same cause — the CDI spec is missing or stale.  Regenerate it with the
command above.  If you recently updated your NVIDIA driver, the spec must
be regenerated to pick up the new device paths.

---

### `unauthorized: authentication required` when pulling the container

Your Podman login session has expired.  Re-run Step 3.

---

### `uv: command not found` inside the container

The `curl | sh` installer writes to `~/.local/bin/uv`.  Make sure the
`export PATH` line is present in your `bash -c` string:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

### `Lockfile does not exist` or `uv.lock is out of date`

Run `uv sync` on the host (Step 6) to regenerate `uv.lock`, then re-run
the `podman run` command.

---

### uv tries to install torch and overwrites the NGC wheel

This should not happen because `torch` is excluded from `pyproject.toml`.
If you added it manually, remove it and run `uv sync` again on the host.

---

### Volume mount permission errors (rootless Podman)

In rootless Podman, the container's UID 0 maps to your host user via user
namespaces.  Files you own on the host appear as root-owned inside the
container, and vice versa.  This normally works transparently.

If you hit permission errors on mounted volumes:

1. Ensure the host directories (`models/`, project root) are owned by
   your user.
2. On SELinux systems (Fedora, RHEL), add the `:Z` suffix to volume
   mounts:
   ```bash
   -v "$(pwd)":/workspace:Z \
   -v "$(pwd)/models":/root/.cache/huggingface:Z \
   ```

---

### Short image names fail to resolve

Podman does not default to Docker Hub.  Use fully-qualified image names:

```bash
# Won't work:
podman run nvidia/cuda:12.0.1-base-ubuntu22.04 ...

# Will work:
podman run docker.io/nvidia/cuda:12.0.1-base-ubuntu22.04 ...
```

NGC images (`nvcr.io/...`) already include the full registry path and
work without changes.

---

### `CUDA out of memory` on the embeddings or QA demo

BERT base is small (~440 MB), but if other processes are using the GPU
you may hit OOM.  Free the GPU first, or add `--device cpu` to the script:

```bash
... python src/bert_demo/embeddings.py --device cpu
```

(The scripts default to GPU when available but fall back to CPU
automatically if no CUDA device is detected.)

---

## How uv fits in

```
HOST                                    CONTAINER (nvcr.io/…/pytorch:26.02-py3)
─────────────────────────────────────   ────────────────────────────────────────
uv init / uv add                        curl | sh  →  uv installed to ~/.local/bin
  → pyproject.toml                      uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken
  → uv.lock  ────── local dev only      installs into the container's Python
                                        python src/bert_demo/*.py
                                          imports torch  (pre-installed by NGC)
                                          imports transformers  (installed by uv)
```

- **`uv pip install --system --break-system-packages "transformers>=5.3.0" "accelerate>=1.13.0" sentencepiece tiktoken`** installs those packages
  directly into the container's Python. `--break-system-packages` bypasses PEP 668
  (externally-managed Python), which blocks installs on NGC's Ubuntu 24.04 base.
  Safe in this context because the container is isolated from the host.
- **Why not `uv sync`?** `UV_SYSTEM_PYTHON` does not affect `uv sync`; it only
  applies to `uv pip`. So `uv sync` would create a `.venv` and install there,
  but `python` would run the system interpreter, which lacks the packages. Using
  `uv pip install --system` avoids that mismatch.
- On the host, **`uv sync`** (Step 6) creates `uv.lock` for local development;
  the container does not use the lock file.

---

## References

- [Podman — official site](https://podman.io/)
- [Podman installation guide](https://podman.io/docs/installation)
- [NVIDIA Container Toolkit — CDI support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
- [NGC PyTorch container release notes (26.02)](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-02.html)
- [NGC container registry — PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [NGC model — bertbaseuncased (NeMo)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/bertbaseuncased)
- [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Hugging Face — bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- [uv — Using uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/)
- [NGC CLI user guide](https://docs.ngc.nvidia.com/cli/cmd.html)
