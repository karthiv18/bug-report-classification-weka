# Bug Report Classification using Weka

## Overview
This project implements a bug report classifier using machine learning techniques. The task is to classify issue reports as either **bug** or **non-bug** using textual features.

## Dataset
- Source: Caffe GitHub repository
- Size: 40 instances (20 bug, 20 non-bug)
- Preprocessed to remove formatting inconsistencies

## Methodology
- Feature extraction: TF-IDF (top 1000 words)
- Classifiers:
  - Naive Bayes (baseline)
  - Random Forest (proposed)
  - SVM (SMO)

- Evaluation:
  - 10-fold stratified cross-validation
  - Primary metric: Macro F1
  - Statistical test: 5×2cv paired t-test

## Results

| Classifier      | Macro F1 | Accuracy |
|----------------|---------|----------|
| Naive Bayes    | NaN     | 50%      |
| Random Forest  | 0.4949  | 50%      |
| SVM            | NaN     | 50%      |

## Key Findings
- Random Forest produced the most reliable performance
- Naive Bayes and SVM showed degenerate behaviour
- Macro F1 is essential when evaluating such models

## How to Run

```bash
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".:weka.jar:." BugReportClassifier final.csv
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".:weka.jar:." StatisticalTest final.csv
