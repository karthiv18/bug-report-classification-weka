# Bug Report Classification using Weka

## Overview

This project investigates automated bug report classification as a binary text classification problem. The objective is to distinguish genuine bug reports from non-bug issues using machine learning techniques applied to textual data.

The study compares a baseline Naive Bayes model against two proposed classifiers: Random Forest and Support Vector Machine (SVM), using TF-IDF feature representation.

---

## Dataset

* **Source:** Caffe GitHub repository
* **Size:** 40 instances (20 bug, 20 non-bug)
* **Preprocessing:** Cleaned to remove formatting inconsistencies and ensure compatibility with Weka

---

## Methodology

### Feature Extraction

* TF-IDF representation
* Top 1000 terms used as features

### Classifiers

* **Naive Bayes** (baseline)
* **Random Forest** (proposed)
* **Support Vector Machine (SMO)** (proposed)

### Evaluation Strategy

* 10-fold stratified cross-validation
* Primary metric: **Macro F1**
* Statistical test: **5×2 cross-validation paired t-test**

---

## Results

| Model         | Macro F1 | Accuracy | Kappa |
| ------------- | -------- | -------- | ----- |
| Naive Bayes   | NaN      | 50%      | 0.0   |
| Random Forest | 0.4949   | 50%      | 0.0   |
| SVM (SMO)     | NaN      | 50%      | 0.0   |

---

## Key Findings

* Random Forest achieved the highest macro F1 score (0.4949), indicating more balanced classification performance.
* Naive Bayes and SVM exhibited **degenerate behaviour**, predicting a single class for all instances.
* This resulted in **undefined (NaN) macro F1 scores**, as precision and recall for one class became zero.
* Accuracy alone is misleading in such scenarios; macro F1 provides a more reliable evaluation.

---

## Statistical Significance

The 5×2 cross-validation paired t-test could not be computed meaningfully.
Naive Bayes produced undefined (NaN) macro F1 scores, resulting in undefined variance.

Therefore, no valid statistical comparison could be performed.

---

## Project Structure

```
bug-report-classification-weka/
│
├── BugReportClassifier.java     
├── StatisticalTest.java        
├── final.csv                   
├── results.txt                
├── weka.jar                 
└── README.md                  
```

---

## How to Run

### Compile

```bash id="c1m8pn"
javac -cp ".:weka.jar" BugReportClassifier.java
javac -cp ".:weka.jar" StatisticalTest.java
```

### Run Classification

```bash id="w8v2zx"
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".:weka.jar:." BugReportClassifier final.csv
```

### Run Statistical Test

```bash id="q4k7lm"
java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".:weka.jar:." StatisticalTest final.csv
```

---

## Key Insight

The NaN macro F1 scores arise because one class receives zero predictions, making precision and recall undefined. This highlights a limitation of macro-averaged metrics under degenerate classifier behaviour.

Random Forest avoids this issue and demonstrates more stable performance on this dataset.

---

## References

* Dietterich, T. (1998). Approximate Statistical Tests for Comparing Classification Algorithms
* Weka Documentation: https://www.cs.waikato.ac.nz/ml/weka/
