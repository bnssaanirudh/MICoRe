# MICoRe: Causal Representation Learning

This repository contains the implementation of **MICoRe**, a research-grade framework for identifiable causal representation learning.

## 🚀 Features
- **iVAE-based Identifiability**: Learns ground-truth causal factors from multi-environment data.
- **NOTEARS-MLP DAG Learning**: Recovers the structural causal model (SCM) between latents.
- **Minimal Intervention Regularization**: Enforces sparse soft interventions for better generalization.
- **Comprehensive Metrics**: Includes DCI, SHD, and OOD evaluation.

## 📁 Structure
- `data/`: Dataset generators and loaders.
- `models/`: iVAE, NOTEARS, and unified MICoRe modules.
- `training/`: Trainer and complex loss functions.
- `evaluation/`: Metrics and visualization tools.
- `experiments/`: Scripts for running full pipelines and ablations.
- `results/`: Paper draft and experiment logs.

## 🛠️ Setup
```bash
pip install -r requirements.txt
```

## 🧪 Running Experiments
Run the full pipeline on the synthetic 3DIdent-like dataset:
```bash
python main.py --dataset 3dident --epochs 100 --viz
```

Run on the Pendulum dataset:
```bash
python main.py --dataset pendulum --epochs 50 --viz
```

## 📊 Evaluation
Results are printed to the console and visualizations (graph, latents) are displayed if `--viz` is used.
Check `results/paper_draft.md` for the theoretical background and methodology.
