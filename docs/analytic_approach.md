# Analytic Approach

## 1. Overview

This project follows a supervised machine learning approach to analyze the relationships between health indicators and diabetes risk. The objective is not only to build predictive models with good performance but also to understand how different health indicators influence the prediction outcomes. Therefore, Explainable Artificial Intelligence (XAI) is incorporated into the analytical workflow to improve model transparency and interpretability.

The overall workflow follows the CRISP-DM methodology, beginning with business understanding and ending with model interpretation and discussion.

---

## 2. Analytical Pipeline

The project follows the pipeline below:

```text
Business Understanding
        ↓
Dataset Understanding
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis (EDA)
        ↓
Statistical Analysis
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Machine Learning Modeling
        ↓
Model Evaluation
        ↓
Explainable AI (SHAP)
        ↓
Discussion & Conclusions
```

Each stage contributes to answering the research questions while ensuring the analytical process remains systematic, reproducible, and interpretable.

---

## 3. Data Understanding

The CDC Diabetes Health Indicators dataset is first examined to understand its structure, variable definitions, data types, and target distribution. Basic data quality checks are performed to identify duplicated records, missing values, and potential inconsistencies before further analysis.

Understanding the characteristics of the dataset provides the foundation for selecting suitable preprocessing techniques and machine learning models.

---

## 4. Data Preparation

The dataset is prepared for analysis through several preprocessing steps, including data validation, duplicate checking, data type verification, and feature inspection. Since the dataset is already encoded numerically, only minimal preprocessing is expected.

The cleaned dataset is then stored as the processed version for all subsequent analyses to ensure consistency throughout the project.

---

## 5. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is conducted to investigate the distributions of health indicators and their relationships with diabetes status.

Descriptive statistics and visualization techniques are used to:

* Understand the distribution of each variable.
* Examine the balance of the target variable.
* Explore associations between individual health indicators and diabetes.
* Identify potential patterns and trends for feature selection.

The findings from EDA provide evidence for later statistical analysis and machine learning modeling.

---

## 6. Statistical Analysis

Statistical methods are employed to determine whether observed differences between groups are statistically significant.

Depending on the variable type, appropriate statistical tests will be applied to evaluate the relationships between health indicators and diabetes status. These analyses provide additional evidence beyond visualization and help identify variables that may contribute meaningfully to predictive modeling.

---

## 7. Machine Learning Modeling

Multiple supervised machine learning algorithms are developed and compared to predict diabetes risk.

The models are trained using the processed dataset and evaluated under the same data partition strategy to ensure fair comparison.

Performance is assessed using multiple evaluation metrics, allowing the strengths and limitations of each algorithm to be analyzed systematically.

---

## 8. Explainable Artificial Intelligence

Although machine learning models can achieve high predictive performance, their predictions are often difficult to interpret.

To improve transparency, SHAP (SHapley Additive exPlanations) is employed to explain both global and local model behavior.

Global explanations identify the overall importance of each health indicator, while local explanations demonstrate how individual features influence predictions for specific observations.

This approach enables a deeper understanding of the factors contributing to diabetes risk and enhances the interpretability of the predictive models.

---

## 9. Expected Analytical Outcomes

The analytical approach is expected to produce:

* A comprehensive understanding of the CDC Diabetes Health Indicators dataset.
* Statistical evidence of relationships between health indicators and diabetes.
* Comparative performance of multiple machine learning models.
* Explainable AI insights into the contribution of individual health indicators.
* Practical findings that support interpretable and evidence-based diabetes risk analysis.
