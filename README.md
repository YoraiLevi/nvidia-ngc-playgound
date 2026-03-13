# NVIDIA NGC — BERT Base Uncased in Docker

A step-by-step tutorial for running **BERT base uncased** inside the
[NVIDIA NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
Every step is a single copy-pastable shell command.
No custom Docker image is built — you work directly with the published NGC container.

---

## What you will build

| Demo | Script | What it shows |
|---|---|---|
| Masked Language Modeling | `src/bert_demo/masked_lm.py` | Predict the top-k words for a `[MASK]` token |
| Sentence Embeddings | `src/bert_demo/embeddings.py` | Encode sentences and compare them by cosine similarity |
| Question Answering | `src/bert_demo/qa.py` | Locate an answer span inside a context paragraph |
| **Benchmark Report** | `src/bert_demo/benchmark.py` | Run 9 BERT-family models, report device / memory / timing |

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
        └── benchmark.py
```

Model weights are downloaded from Hugging Face Hub on first run and
cached in a local `models/` directory that is volume-mounted into the
container, so they are only downloaded once.

---

## Shell variants

The `docker run` commands below use **bash** syntax (`$(pwd)`, `\`).
If you are on **Windows PowerShell** substitute as shown:

| bash | PowerShell |
|---|---|
| `$(pwd)` | `${PWD}` |
| `\` (line continuation) | `` ` `` |
| `export VAR=value` | `$env:VAR = "value"` |

**PowerShell example** — each demo step uses the same pattern, just swap
the line-continuation character:

```powershell
docker run --gpus all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:25.01-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/masked_lm.py'
```

---

## Prerequisites

Before starting, confirm the following are installed:

- **Docker** — [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine (Linux)
- **NVIDIA Container Toolkit** — enables GPU passthrough into containers
  ([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- **NVIDIA driver ≥ 570** — required by the 25.01 NGC container
- An **NVIDIA GPU** with CUDA support
- A free **NVIDIA developer account** at [developer.nvidia.com](https://developer.nvidia.com)

Verify your GPU and driver are visible to Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.1-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed in the `nvidia-smi` output.

---

## Step 1 — Get an NGC API key

1. Sign in at [https://ngc.nvidia.com](https://ngc.nvidia.com)
2. Click your account name → **Setup** → **Generate API Key**
3. Copy the key — it is shown only once

Keep it safe; you will use it in the next step and again in Step 12.

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

## Step 3 — Authenticate Docker with NGC

NGC's container registry requires you to log in once per machine.
The username is always the literal string `$oauthtoken`; the password is
your API key.

```bash
docker login nvcr.io --username '$oauthtoken' --password "$(grep NGC_API_KEY .env | cut -d= -f2)"
```

**PowerShell:**

```powershell
$key = (Get-Content .env | Where-Object { $_ -match "NGC_API_KEY" }) -replace "NGC_API_KEY=", ""
docker login nvcr.io --username '$oauthtoken' --password "$key"
```

A successful login prints `Login Succeeded`.

---

## Step 4 — Pull the NGC PyTorch container

The `25.01-py3` image includes CUDA 12.8, cuDNN 9.7, and PyTorch 2.6.
It is ~18 GB; this step only needs to run once.

```bash
docker pull nvcr.io/nvidia/pytorch:25.01-py3
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
`docker run` command below.  Hugging Face Transformers uses that path as
its default download cache.

---

## Step 8 — Demo 1: Masked Language Modeling

`masked_lm.py` loads **bert-base-uncased** and asks it to predict the
word hidden behind a `[MASK]` token.

**Linux / macOS / WSL:**

```bash
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/masked_lm.py'
```

**PowerShell:**

```powershell
docker run --gpus all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:25.01-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/masked_lm.py'
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
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/masked_lm.py --text "NVIDIA makes the world'\''s fastest [MASK]." --top-k 3'
```

---

## Step 9 — Demo 2: Sentence Embeddings

`embeddings.py` encodes five sentences into 768-dimensional vectors using
mean pooling over the last hidden state, then prints a cosine-similarity
matrix.  Semantically similar sentence pairs will be close to 1.0.

**Linux / macOS / WSL:**

```bash
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/embeddings.py'
```

**PowerShell:**

```powershell
docker run --gpus all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:25.01-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/embeddings.py'
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
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/embeddings.py --sentences "I love pizza." "Pizza is my favourite food." "The sky is blue."'
```

---

## Step 10 — Demo 3: Question Answering

`qa.py` uses **deepset/bert-base-uncased-squad2** — a `bert-base-uncased`
checkpoint fine-tuned on SQuAD 2.0 — to locate answer spans inside
context paragraphs.

**Linux / macOS / WSL:**

```bash
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/qa.py'
```

**PowerShell:**

```powershell
docker run --gpus all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:25.01-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/qa.py'
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
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/qa.py --question "What year was CUDA released?" --context "NVIDIA introduced CUDA in 2006 as a parallel computing platform."'
```

---

## Step 11 — Benchmark Report: all models at once

`benchmark.py` loads **nine BERT-family models** across five task types
and prints a summary table with device info, parameter counts, peak GPU
memory, and wall-clock timings.

**Models included:**

| # | Model | Task |
|---|---|---|
| 1 | `bert-base-uncased` | Masked LM |
| 2 | `bert-large-uncased` | Masked LM |
| 3 | `distilbert-base-uncased` | Masked LM |
| 4 | `albert-base-v2` | Masked LM |
| 5 | `deepset/bert-base-uncased-squad2` | Question Answering |
| 6 | `dslim/bert-base-NER` | Named Entity Recognition |
| 7 | `nlptown/bert-base-multilingual-uncased-sentiment` | Sentiment (5-star) |
| 8 | `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment (binary) |
| 9 | `bert-base-uncased` | Sentence Embeddings |

**Linux / macOS / WSL:**

```bash
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/benchmark.py'
```

**PowerShell:**

```powershell
docker run --gpus all --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}/models:/root/.cache/huggingface" `
  -w /workspace `
  -e UV_SYSTEM_PYTHON=1 `
  nvcr.io/nvidia/pytorch:25.01-py3 `
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/benchmark.py'
```

**Force CPU (for comparison):**

```bash
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/root/.cache/huggingface \
  -w /workspace \
  -e UV_SYSTEM_PYTHON=1 \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet && export PATH="$HOME/.local/bin:$PATH" && uv pip install --system --break-system-packages transformers accelerate && python src/bert_demo/benchmark.py --device cpu'
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

[1/9] fill-mask               bert-base-uncased
       -> done  (load 1.82s, infer 0.0134s)
[2/9] fill-mask               bert-large-uncased
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
  ...
=============================================================================================================
  Memory = peak CUDA memory allocated (torch.cuda.max_memory_allocated)
```

---

## Step 12 — Optional: Download the NGC BERT model artifact

NVIDIA also hosts a pre-trained **BERTBaseUncased** checkpoint in the NGC
model registry under `nvidia/nemo/bertbaseuncased`.  You can download it
with the **NGC CLI** and load it through the NeMo framework.

### 12a — Install the NGC CLI inside the container

```bash
docker run --gpus all --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c '
    wget -q https://ngc.nvidia.com/downloads/ngccli_linux.zip -O /tmp/ngccli.zip &&
    unzip -q /tmp/ngccli.zip -d /tmp/ngccli &&
    chmod +x /tmp/ngccli/ngc-cli/ngc &&
    export PATH="/tmp/ngccli/ngc-cli:$PATH" &&
    ngc --version
  '
```

### 12b — Configure the NGC CLI and download the model

Replace `<YOUR_API_KEY>` with the value from your `.env` file.

```bash
docker run --gpus all --rm -it \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/models \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.01-py3 \
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

### 12c — Load the checkpoint with NeMo (inside the container)

Install NeMo on top of the PyTorch container, then load the downloaded
checkpoint:

```bash
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/models":/models \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.01-py3 \
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

### `docker: Error response from daemon: could not select device driver`

The NVIDIA Container Toolkit is not installed or not configured.
Follow the [official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then restart the Docker daemon:

```bash
sudo systemctl restart docker
```

---

### `unauthorized: authentication required` when pulling the container

Your Docker login session has expired.  Re-run Step 3.

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
the `docker run` command.

---

### uv tries to install torch and overwrites the NGC wheel

This should not happen because `torch` is excluded from `pyproject.toml`.
If you added it manually, remove it and run `uv sync` again on the host.

---

### Windows: `invalid volume specification` path errors

Use `${PWD}` (not `$(pwd)`) in PowerShell, and make sure Docker Desktop
has **"Use the WSL 2 based engine"** or **"Expose daemon on tcp"** enabled
in Settings → General.

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
HOST                                    CONTAINER (nvcr.io/…/pytorch:25.01-py3)
─────────────────────────────────────   ────────────────────────────────────────
uv init / uv add                        curl | sh  →  uv installed to ~/.local/bin
  → pyproject.toml                      uv pip install --system --break-system-packages transformers accelerate
  → uv.lock  ────── local dev only      installs into the container's Python
                                        python src/bert_demo/*.py
                                          imports torch  (pre-installed by NGC)
                                          imports transformers  (installed by uv)
```

- **`uv pip install --system --break-system-packages transformers accelerate`** installs those packages
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

- [NGC PyTorch container release notes (25.01)](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-01.html)
- [NGC container registry — PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [NGC model — bertbaseuncased (NeMo)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/bertbaseuncased)
- [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Hugging Face — bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- [uv — Using uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/)
- [NGC CLI user guide](https://docs.ngc.nvidia.com/cli/cmd.html)
