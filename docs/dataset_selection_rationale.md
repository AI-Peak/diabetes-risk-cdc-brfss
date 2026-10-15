# Dataset Selection Rationale

## 1. Overview

Selecting an appropriate dataset is a critical step in ensuring that the analytical findings are meaningful, reliable, and applicable to real-world scenarios. For this project, the **CDC Diabetes Health Indicators** dataset was selected because it contains a comprehensive collection of demographic, lifestyle, and health-related variables that are relevant to diabetes risk analysis.

The dataset originates from the **Behavioral Risk Factor Surveillance System (BRFSS)** conducted by the Centers for Disease Control and Prevention (CDC), making it one of the most widely used public health survey datasets for diabetes research.

---

## 2. Reasons for Selecting This Dataset

The dataset was selected for several reasons:

* It contains a large number of observations, allowing robust statistical analysis and machine learning experiments.
* The variables represent multiple aspects of an individual's health, including physiological conditions, lifestyle behaviors, healthcare access, demographic characteristics, and socioeconomic status.
* The dataset is publicly available and widely used in academic research, supporting reproducibility and comparison with previous studies.
* The variables are already structured in tabular format, reducing the complexity of preprocessing while preserving sufficient analytical depth.
* The dataset is suitable for both predictive modeling and interpretable machine learning techniques such as SHAP.

---

## 3. Use of the Original Imbalanced Dataset

This project intentionally uses the **original CDC Diabetes Health Indicators dataset** rather than the pre-balanced version.

Maintaining the original class distribution allows the analytical results to better reflect real-world diabetes prevalence. Although class imbalance may increase the difficulty of classification, it provides a more realistic evaluation of model performance and better represents practical deployment scenarios.

To address the imbalance, model performance will be evaluated using multiple classification metrics rather than relying solely on overall accuracy.

---

## 4. Strengths of the Dataset

The selected dataset provides several advantages:

* Large sample size suitable for statistical inference and machine learning.
* Rich collection of demographic, behavioral, and health-related variables.
* Reliable source from a nationally recognized public health survey.
* Minimal preprocessing requirements due to standardized variable encoding.
* Appropriate for Explainable Artificial Intelligence (XAI) studies.
* Supports both statistical analysis and predictive analytics within a single dataset.

---

## 5. Limitations of the Dataset

Despite its strengths, several limitations should be acknowledged:

* The data are collected through self-reported surveys, which may introduce reporting bias.
* The dataset is cross-sectional and does not capture temporal changes in health conditions.
* Some important clinical measurements, laboratory test results, and genetic information are unavailable.
* The original dataset exhibits class imbalance, which may influence model learning and evaluation.
* The dataset represents the U.S. population and may not fully generalize to other countries or healthcare systems.

---

## 6. Suitability for This Project

The CDC Diabetes Health Indicators dataset aligns well with the objectives of this project. It enables comprehensive exploratory data analysis, statistical investigation of diabetes-related factors, predictive modeling using machine learning algorithms, and interpretation through Explainable Artificial Intelligence.

Overall, the dataset provides an appropriate balance between data quality, analytical complexity, and interpretability, making it suitable for an end-to-end analytics project focused on diabetes risk analysis.
