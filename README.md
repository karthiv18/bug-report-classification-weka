# Bug Report Classification using Machine Learning

## Overview
This project implements an intelligent software engineering tool for classifying bug reports using machine learning.

We compare:
- Naive Bayes (baseline)
- Random Forest (proposed)
- Support Vector Machine (SVM)

TF-IDF is used for feature extraction.

## Features
- Proper ML pipeline using Weka
- 10-fold cross-validation
- Multiple evaluation metrics (Accuracy, F1-score, Precision, Recall)
- Statistical significance testing (5x2cv paired t-test)

## Dataset
The dataset contains bug report text and labels:
- Column 1: Text
- Column 2: Label (bug / non-bug)

## How to Run

### Compile
```bash
javac -cp ".:weka.jar" src/BugReportClassifier.java
javac -cp ".:weka.jar" src/StatisticalTest.java
