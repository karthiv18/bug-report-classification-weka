# Bug Report Classification using Machine Learning

## Overview
This project presents an intelligent software engineering tool for automatically classifying bug reports based on their textual descriptions. Efficient bug classification helps developers prioritise issues and improve software maintenance.

The system compares a baseline approach with more advanced machine learning techniques to improve classification performance.

## Models Compared
- Naive Bayes (Baseline)
- Random Forest (Proposed)
- Support Vector Machine (SVM)

TF-IDF is used to convert textual bug reports into numerical feature vectors.

## Motivation
Naive Bayes assumes independence between words, which is often unrealistic in bug reports. More advanced models such as Random Forest and SVM are expected to perform better by capturing complex relationships in the data.

## Features
- Proper machine learning pipeline using Weka
- TF-IDF feature extraction
- 10-fold cross-validation
- Multiple evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score (Weighted and Macro)
- Confusion matrix analysis
- Statistical significance testing using 5x2 cross-validation paired t-test

## Dataset
The dataset contains bug report descriptions and labels:

- Column 1: Text (bug report description)
- Column 2: Label (bug / non-bug)

## How to Run

### Compile
```bash
javac -cp ".:weka.jar" src/BugReportClassifier.java
javac -cp ".:weka.jar" src/StatisticalTest.java
