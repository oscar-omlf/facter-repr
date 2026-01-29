# Reproducing FACTER (ICML 2025): Fairness-Aware Conformal Thresholding + Prompt Engineering

This repository is an **independent reimplementation** and **reproducibility study** of:

**FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair LLM-Based Recommender Systems**  
Arya Fayyazi, Mehdi Kamal, Massoud Pedram (ICML 2025)  
[Paper](https://openreview.net/pdf?id=edN2rEemj6) (OpenReview)  
[Code](https://github.com/AryaFayyazi/FACTER) (Github)

FACTER is a post-hoc wrapper around a black-box LLM recommender that:

1) calibrates a **fairness threshold** using conformal prediction (offline), and
2) iteratively performs **prompt repair** when fairness violations are detected (online), without retraining the LLM.

This repo contains dataset builders and an end-to-end experiment runner for:

- MovieLens-1M (`ml-1m`) (visit https://grouplens.org/datasets/movielens/1m/ for details)
- Amazon Movies & TV 5-core + metadata (`amazon`) (visit https://jmcauley.ucsd.edu/data/amazon/ for details)

## What you can reproduce

This repository is intended to reproduce the full pipeline behavior (dataset construction, baselines, offline calibration, online monitoring, and MLflow logging).
Exact metric parity with the reproduction paper may depend on model choice, compute budget (GPU vs CPU), and prompt/model availability.

## Repository structure

*High-level map* (selected):

```text
.
├─ src/facter/                 # Library code
│  ├─ data/                    # Download + dataset building utilities
│  ├─ eval/                    # Metrics + baselines + counterfactual evaluation
│  ├─ fairness/                # Offline calibration + online monitoring/scoring
│  ├─ models/                  # HF generator/ranker + embedders + model registry
│  ├─ prompting/               # Prompt repair
│  └─ tracking/                # MLflow logging wrappers
├─ scripts/                    # CLI entrypoints
│  ├─ download_data.py         # Download raw data
│  ├─ build_dataset.py         # Build processed dataset
│  ├─ run_facter.py            # Main experiment runner
│  ├─ mlflow_run_prune.ipynb   # Jupyter notebook to analyse and prune MLflow runs
│  └─ results_analysis.ipynb   # Jupyter notebook to extract and analyse results
├─ data/                       # Raw/processed data + caches (created at runtime)
├─ mlruns.zip                  # MLflow run archive (zip) to be used by scripts/results_analysis.ipynb
└─ mlflow.db                   # Default MLflow store (SQLite)
```


## Setup (conda)

Conda is the recommended setup. Docker/Compose is provided as an optional alternative.

### 1) Create environment
```bash
conda env create -f environment.yml
conda activate facter-repro
```

### 2) Install this repo
From the repo root:
```bash
pip install -e .
```

### 3) Test the installation
```bash
pytest
```

## Download and preprocess data
### Download raw data

MovieLens-1M:
```bash
python scripts/download_data.py --dataset ml-1m
```

Amazon:
```bash
python scripts/download_data.py --dataset amazon
```

### Build the processed split + prompts

You must build the processed dataset for each dataset you want to run.

This produces a deterministic sample of interactions, then a 70/30 calibration/test split, and then per-row prompts.

MovieLens-1M:
```bash
python scripts/build_dataset.py \
  --dataset ml-1m \
  --seed 0 \
  --n 2500 \
  --n_candidates 40 \
  --relevance_mode future_window \
  --relevance_window 10
```

Amazon Movies & TV:
```bash
python scripts/build_dataset.py \
  --dataset amazon \
  --seed 0 \
  --n 3750 \
  --n_candidates 40 \
  --relevance_mode future_window \
  --relevance_window 10
```

Outputs:
- MovieLens-1M:
  - `data/processed/ml-1m/cal/dataset.jsonl`
  - `data/processed/ml-1m/test/dataset.jsonl`
  - `data/processed/ml-1m/meta.json`
- Amazon:
  - `data/processed/amazon/cal/dataset.jsonl`
  - `data/processed/amazon/test/dataset.jsonl`
  - `data/processed/amazon/meta.json`

## Running experiments

The main entrypoint is `scripts/run_facter.py`.
This repo runs local models downloaded from HugggingFace for the LLM ranker/generator.

GPU is recommended. CPU runs are supported but can be very slow.

#### Model selection
You can choose a model in two ways:

1) **Recommended:** pick a short name via `--base_model`. (Available options: `llama3`, `llama2`, `mistral`)
  This resolves to a HuggingFace `model_id` via the registry in `src/facter/models/model_registry.py` (`BASE_MODELS`).

2) **Override:** pass a Hugging Face `--model_id` directly.
  If provided, it overrides `--base_model`.

Example (use a base model preset):
```bash
python scripts/run_facter.py \
  --base_model llama3 \
  --seeds 0 \
  --protected_attrs gender,age,occupation \
  --max_iterations 3 \
  --progress \
  --predict_mode open \
  --datasets ml-1m,amazon \
  --baseline_prompts both
```

Example (override with an explicit HF model id):
```bash
python scripts/run_facter.py \
  --base_model mistral \
  --model_id mistralai/Mistral-7B-Instruct-v0.2 \
  --seeds 0 \
  --protected_attrs gender,age,occupation \
  --max_iterations 5 \
  --progress \
  --predict_mode open \
  --datasets ml-1m \
  --baseline_prompts both 
```

This will:
- run a Zero-Shot baseline (rank without fairness prompt repair),
- run a Fair baseline (rank with static fairness prompt),
- run offline conformal calibration to compute Q_alpha^(0),
- run online iterations with violation-triggered prompt repair + threshold update,
- log results to MLflow (default: `sqlite:///./mlflow.db`),
- save per-example tables under `data/processed/<dataset>/runs/*.parquet`.

### Hyperparameters (FACTER paper defaults):

- `--tau_rho 0.90`
- `--lambda_fairness 0.7`
- `--gamma 0.95`
- `--buffer_size 50`
- `--alpha 0.10` (note: some sections also discuss values as {0.90..0.98})

## Tracking (MLflow)
This repo logs:
- run parameters (seed, hyperparameters, dataset config)
- metrics (baseline + per-iteration FACTER summary)
- a JSON summary artifact

To inspect runs:
```bash
mlflow ui --backend-store-uri "sqlite:///./mlflow.db"
```

Then open the MLflow UI and filter by experiment/run to view metrics and logged artifacts.

## Caching and performance notes
- The HF ranker caches rankings in `data/cache/ranker/` keyed by (system_prompt, user_prompt, candidates).
- The SentenceTransformer embedder caches embeddings in `data/cache/embeddings/`.
- Running large LLMs locally typically requires a GPU with sufficient VRAM.

### Hugging Face access / `HF_TOKEN`

Some models are gated on Hugging Face and require an access-approved account and a token:

- https://huggingface.co/meta-llama/Llama-2-7b
- https://huggingface.co/meta-llama/Meta-Llama-3-8B

Other models (e.g. Mistral) are typically ungated:

- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

If you use a gated model and do not set `HF_TOKEN`, downloading the weights will fail.

Set `HF_TOKEN` on your host before starting the container:

* **Linux/macOS**

  ```bash
  export HF_TOKEN=hf_...
  ```
* **Windows PowerShell**

  ```powershell
  $env:HF_TOKEN="hf_..."
  ```

* **Windows (cmd.exe)**

  ```bat
  set HF_TOKEN=hf_...
  ```

## Setup Docker / Compose (persistent environment)

Docker is optional. If you prefer a conda-native workflow, you can ignore this section.

### Prerequisites

* Docker + Docker Compose installed.
* For using gated models the HF_TOKEN environment variable must be set as described above.

### First run (build image + create the environment)

The first time you run this, Docker will:

* pull the base image,
* create the conda environment from `environment.yml`,
* install this repo inside the image (`pip install -e .`),
* start a persistent container you can `exec` into.

Start the GPU environment:

```bash
docker compose --profile gpu up -d --build
docker compose exec env-gpu bash
```

You are now inside the container (conda env active). Run the same experiment commands shown in the sections above, `python scripts/build_dataset.py ...`, and `python scripts/run_facter.py ....`

### Repeated runs (reuse the already-built image/container)

On subsequent runs, you do not need to rebuild anything.

Start (GPU):

```bash
docker compose --profile gpu up -d
docker compose exec env-gpu bash
```

Stop:

```bash
docker compose down
```


## Contributions
...

## License
This repository is for academic, non-commercial use only. For other uses, please contact the authors.