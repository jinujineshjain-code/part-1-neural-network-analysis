# Part 1: Neural Network Analysis — Customer Churn Prediction

## Problem
Predict whether a telecom customer will churn (leave the service) based on usage and account features.

## Dataset
- **File:** `customer_churn_nn.csv`
- **Rows:** 2000 | **Columns:** 17
- **Target:** `churn` (0 = No Churn, 1 = Churned)
- **Class distribution:** 98.45% no churn, 1.55% churn (highly imbalanced)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Results Summary
| Experiment | Hidden Layers | Activation | LR | Batch | Test Acc |
|---|---|---|---|---|---|
| Exp1 - Baseline | (64,) | ReLU | 0.001 | 32 | 0.9800 |
| Exp2 - Deeper | (128,64) | ReLU | 0.001 | 32 | 0.9725 |
| Exp3 - High LR | (64,) | ReLU | 0.01 | 32 | 0.9750 |
| Exp4 - Small Batch | (64,) | ReLU | 0.001 | 16 | 0.9800 |
| Exp5 - Tanh | (64,) | Tanh | 0.001 | 32 | 0.9875 |

## Key Findings
- The dataset is highly imbalanced — ~98.5% of customers do not churn.
- All models achieve ~97-99% accuracy, primarily by predicting "No Churn."
- The churn recall is low — a real business deployment would need SMOTE or class weights.
- Tanh activation (Exp5) gave the best test accuracy for this dataset.

## Repository Structure
```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.csv
    ├── model_comparison_table.png
    ├── confusion_matrix.png
    └── evaluation_outputs.png
```
