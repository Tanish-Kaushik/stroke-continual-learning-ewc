[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Continual%20Learning-purple)](https://arxiv.org/abs/1612.00796)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

# Continual Learning for Ischemic Stroke Severity Assessment

**Target Professor:** Kavitha Muthu Subash (Nagasaki University)  
**Research Theme:** Pattern Recognition and Understanding  
**Aligns with funded grant:** *"A data-saving and self-supervised deep learning system for continuous ischemic stroke assessment"* (2024–2027)

## 📌 Problem Statement
Conventional stroke severity models suffer from **catastrophic forgetting** when new patient data arrives (e.g., new hospital, new imaging protocol). They forget previously learned knowledge. This project implements **Elastic Weight Consolidation (EWC)** combined with **Experience Replay** to maintain performance across sequential tasks without forgetting.

## 🧠 Methodology
- **Tasks:** Synthetic data → 30% real → 100% real (simulated)
- **Model:** ResNet-18 adapted for binary stroke severity (mild/severe)
- **Continual Learning Strategies:** Baseline (no CL), EWC only, Replay only, EWC+Replay
- **Evaluation Metrics:** Average accuracy, Backward Transfer (BWT), Fisher Information heatmap

## 📊 Results

|        Strategy         | Backward Transfer (BWT) | Forgetting |
|-------------------------|-------------------------|------------|
| Baseline (fine-tuning)  |         -35.2%          |   Severe   |
|        EWC only         |         -14.8%          |  Moderate  |
|      Replay only        |          -8.3%          |    Low     |
|    **EWC + Replay**     |        **-2.1%**        |**Near-zero**|

## 📊 Visualizations

| Average Accuracy | Backward Transfer | Fisher Heatmap |
|----------------|------------------|----------------|
| ![Avg Acc](./avg_accuracy.png) | ![BWT](./backward_transfer.png) | ![Fisher](./fisher_heatmap.png) |

Higher bars indicate filters that EWC identified as **critical** for stroke features – these are heavily protected during new learning.

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stroke-continual-learning.git
   cd stroke-continual-learning
