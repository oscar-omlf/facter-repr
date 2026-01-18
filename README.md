# Reproducing FACTER (ICML 2025): Fairness-Aware Conformal Thresholding + Prompt Engineering

This repository is an **independent reimplementation** and **reproducibility study** of:

**FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair LLM-Based Recommender Systems**  
Arya Fayyazi, Mehdi Kamal, Massoud Pedram (ICML 2025)  
Paper (arXiv): https://arxiv.org/abs/2502.02966

FACTER is a post-hoc wrapper around a black-box LLM recommender that:
1) calibrates a **fairness threshold** using conformal prediction (offline), and  
2) iteratively performs **prompt repair** when fairness violations are detected (online), without retraining the LLM.

> Note: This repo focuses on the MovieLens-1M reproduction path first. Amazon support and UP5 are planned/partial.

...

## Repository structure


## Setup (conda)
> NOTE: Ideally we will want to add Docker and/or Singularity for maximum reproducibility.

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
### Download MovieLens-1M
```bash
python scripts/download_data.py --dataset ml-1m
```

### Build the processed split + prompts
This produces a deterministic sample of interactions, then a 70/30 calibration/test split, then per-row prompts.
```bash
python scripts/build_ml1m_dataset.py \
  --seed 42 \
  --n 2500 \
  --n_candidates 100
```

Outputs:
- `data/processed/ml-1m/cal/dataset.jsonl`
- `data/processed/ml-1m/test/dataset.jsonl`
- `data/processed/ml-1m/meta.json`

## Running experiments (MovieLens-1M)
### End-to-end FACTER (offline + online)
You must provide a Hugging Face `model_id` for the LLM ranker. Example:
```bash
python scripts/run_facter.py \
   --model_id meta-llama/Meta-Llama-3-8B-Instruct \
   --seed 42 \
   --protected_attrs gender,age,occupation \
   --max_iterations 3 \
   --progress \
   --predict_mode open \
   --datasets ml-1m,amazon
```

This will:
- run a Zero-Shot baseline (rank without fairness prompt repair),
- run offline conformal calibration to compute Q_alpha^(0),
- run online iterations with violation-triggered prompt repair + threshold update,
- log results to MLflow (default: `sqlite:///./mlflow.db`),
- save a per-example table to `data/processed/ml-1m/runs/*.parquet`.

### Hyperparameters (paper defaults):
The default values I am using (TODO: Triple check):
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
- TODO: Add more?

To inspect runs:
```bash
mlflow ui --backend-store-uri sqlite:///./mlflow.db
```

## Caching and performance notes
- The HF ranker caches rankings in `data/cache/ranker/` keyed by (system_prompt, user_prompt, candidates).
- The SentenceTransformer embedder caches embeddings in `data/cache/embeddings/`.
- Running large LLMs locally typically requires a GPU with sufficient VRAM.

If you use gated models (e.g., certain LLaMA weights), you may need:
```bash
export HF_TOKEN=...
```

## Contributions
...

## License
...