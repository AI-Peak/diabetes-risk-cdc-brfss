# Data Understanding Report
## CDC Diabetes Health Indicators (BRFSS 2015)

### Methodology: CRISP-DM (Data Understanding)
**Author:** Senior Data Analytics Engineer  
**Date:** 2026-07-20  
**Project:** Diabetes-Analytics

---

## 1. Introduction and Objectives
This document presents the **Data Understanding** phase of the CRISP-DM methodology for the **CDC Diabetes Health Indicators** dataset, sourced from the 2015 Behavioral Risk Factor Surveillance System (BRFSS). The primary objective is to inspect, summarize, and validate the dataset before any downstream data preparation, feature engineering, or modeling tasks. 

Importantly, this analysis is performed on the **original, imbalanced dataset** to accurately reflect real-world epidemiological diabetes prevalence within the surveyed population. No balancing techniques (e.g., SMOTE, oversampling, or undersampling) have been applied.

---

## 2. Dataset Overview and Structural Characteristics
The dataset represents a subset of survey responses focusing on behavioral risk factors, healthcare access, demographics, and clinical conditions related to diabetes.

* **Dataset Shape:** (253680, 22)
* **Number of Rows (Observations):** 253,680
* **Number of Columns (Variables):** 22

### Dataset Overview Table
| Metric | Value |
| :--- | :--- |
| **Number of Observations (Rows)** | 253,680 |
| **Number of Attributes (Columns)** | 22 |
| **Data Format** | Comma-Separated Values (CSV) |
| **File Location** | `data/raw/diabetes_binary_health_indicators_BRFSS2015.csv` |

The attributes are represented in pandas as 64-bit floating-point numbers (`float64`), which reflect coded response values from the BRFSS questionnaire.

---

## 3. Variable Classification
The 22 variables in the dataset span multiple analytical domains, including clinical indicators, physical measurements, lifestyles, demographics, and socioeconomic factors. 

Below is the structured variable classification schema:

| Variable | Type | Category | Description |
| :--- | :--- | :--- | :--- |
| `Diabetes_binary` | Binary | **Target** | Diabetes status (0 = no diabetes, 1 = prediabetes or diabetes) |
| `HighBP` | Binary | **Health Condition** | High blood pressure indicator (0 = no high BP, 1 = high BP) |
| `HighChol` | Binary | **Health Condition** | High cholesterol indicator (0 = no high cholesterol, 1 = high cholesterol) |
| `CholCheck` | Binary | **Healthcare Access** | Cholesterol check in past 5 years (0 = no check, 1 = check) |
| `BMI` | Numerical | **Physical Measurement** | Body Mass Index (BMI) |
| `Smoker` | Binary | **Lifestyle** | Smoked at least 100 cigarettes in lifetime (0 = no, 1 = yes) |
| `Stroke` | Binary | **Health Condition** | Ever told you had a stroke (0 = no, 1 = yes) |
| `HeartDiseaseorAttack` | Binary | **Health Condition** | Coronary heart disease or myocardial infarction (0 = no, 1 = yes) |
| `PhysActivity` | Binary | **Lifestyle** | Physical activity in past 30 days excluding work (0 = no, 1 = yes) |
| `Fruits` | Binary | **Lifestyle** | Consume fruit 1 or more times per day (0 = no, 1 = yes) |
| `Veggies` | Binary | **Lifestyle** | Consume vegetables 1 or more times per day (0 = no, 1 = yes) |
| `HvyAlcoholConsump` | Binary | **Lifestyle** | Heavy alcohol consumption (0 = no, 1 = yes) |
| `AnyHealthcare` | Binary | **Healthcare Access** | Have any health care coverage (0 = no, 1 = yes) |
| `NoDocbcCost` | Binary | **Healthcare Access** | Could not see doctor because of cost in past 12 months (0 = no, 1 = yes) |
| `GenHlth` | Ordinal | **General Health** | Self-reported general health scale (1 = excellent to 5 = poor) |
| `MentHlth` | Numerical | **General Health** | Days of poor mental health in past 30 days (0-30) |
| `PhysHlth` | Numerical | **General Health** | Days of poor physical health in past 30 days (0-30) |
| `DiffWalk` | Binary | **Health Condition** | Serious difficulty walking or climbing stairs (0 = no, 1 = yes) |
| `Sex` | Binary | **Demographic** | Biological sex (0 = female, 1 = male) |
| `Age` | Ordinal | **Demographic** | 13-level age category (1 = 18-24 to 13 = 80+) |
| `Education` | Ordinal | **Socioeconomic** | Education level scale (1 = never attended school to 6 = college graduate) |
| `Income` | Ordinal | **Socioeconomic** | Income scale (1 = <$10,000 to 8 = $75,000+) |

### Variables Summary by Type
- **Binary Variables (15):** Indicator variables coded as `0` or `1`.
- **Numerical Variables (3):** Continuous or discrete ratio/interval measures (`BMI`, `MentHlth`, `PhysHlth`).
- **Ordinal Variables (4):** Categorical scales representing progression (e.g., age bracket, education, income, general health).

---

## 4. Missing Values Analysis
To assess data quality and completeness, a missingness check was performed across all 22 attributes.

| Column | Missing Count | Missing Percentage |
| :--- | :---: | :---: |
| `Diabetes_binary` | 0 | 0.0000% |
| `HighBP` | 0 | 0.0000% |
| `HighChol` | 0 | 0.0000% |
| `CholCheck` | 0 | 0.0000% |
| `BMI` | 0 | 0.0000% |
| `Smoker` | 0 | 0.0000% |
| `Stroke` | 0 | 0.0000% |
| `HeartDiseaseorAttack` | 0 | 0.0000% |
| `PhysActivity` | 0 | 0.0000% |
| `Fruits` | 0 | 0.0000% |
| `Veggies` | 0 | 0.0000% |
| `HvyAlcoholConsump` | 0 | 0.0000% |
| `AnyHealthcare` | 0 | 0.0000% |
| `NoDocbcCost` | 0 | 0.0000% |
| `GenHlth` | 0 | 0.0000% |
| `MentHlth` | 0 | 0.0000% |
| `PhysHlth` | 0 | 0.0000% |
| `DiffWalk` | 0 | 0.0000% |
| `Sex` | 0 | 0.0000% |
| `Age` | 0 | 0.0000% |
| `Education` | 0 | 0.0000% |
| `Income` | 0 | 0.0000% |

**Key Observation:** There are **zero missing values** (nulls or NaNs) detected in the raw dataset. This completeness suggests that the dataset has undergone preliminary extraction and formatting prior to our receipt. Consequently, no imputation strategies are required at this stage.

---

## 5. Duplicate Records Evaluation
The dataset was scanned for identical row profiles to evaluate the degree of duplicate records.

- **Duplicate Rows Count:** 24206
- **Duplicate Percentage:** 9.5419%

**Epidemiological Context:** In a large-scale survey consisting of 253,680 respondents and 22 coded categorical or ordinal variables, it is mathematically expected to observe identical response profiles (duplicates) without it implying data entry errors. For example, two respondents may share the exact same profile: female, aged 50-54, college graduate, high income, non-smoker, with high blood pressure, etc. Removing these records would artificiality skew the underlying sample distribution and reduce statistical power. Therefore, **in accordance with CRISP-DM guidelines, duplicate records are retained** for subsequent analysis.

---

## 6. Target Variable Distribution
The target variable, `Diabetes_binary`, represents the diabetes status of the respondent, where `0` indicates no diabetes, and `1` indicates prediabetes or diabetes.

| Target Class | Description | Count | Percentage |
| :---: | :--- | :---: | :---: |
| `0.0` | No Diabetes | 218,334 | 86.07% |
| `1.0` | Prediabetes or Diabetes | 35,346 | 13.93% |

### Class Imbalance Diagnostics
The target variable exhibits a pronounced **class imbalance**, with approximately **86.07%** of the sample classified as without reported diabetes (`0`) and only **13.93%** classified as having prediabetes or diabetes (`1`). This distribution aligns with real-world epidemiological statistics where diabetes is a prevalent but minority health condition in the general population. 

A visualization of this distribution has been saved as `results/data_understanding/target_distribution.png`. Downstream classification models will need to account for this imbalance (e.g., using screening-oriented machine learning evaluation or appropriate evaluation metrics such as F1-score, Precision-Recall AUC, rather than raw accuracy), but the raw data itself must remain unweighted and unadjusted in this phase to preserve real-world prevalence rates.

---

## 7. Descriptive Statistics for Numerical Variables
Descriptive statistics were computed for the three numerical attributes (`BMI`, `MentHlth`, `PhysHlth`) to characterize their central tendency, dispersion, and spread.

| Statistic | BMI | MentHlth (Mental Health) | PhysHlth (Physical Health) |
| :--- | :---: | :---: | :---: |
| **Count** | 253,680 | 253,680 | 253,680 |
| **Mean** | 28.3824 | 3.1848 | 4.2421 |
| **Std. Dev.** | 6.6087 | 7.4128 | 8.7180 |
| **Minimum** | 12.0 | 0.0 | 0.0 |
| **25th Percentile (Q1)** | 24.0 | 0.0 | 0.0 |
| **Median (Q2)** | 27.0 | 0.0 | 0.0 |
| **75th Percentile (Q3)** | 31.0 | 2.0 | 3.0 |
| **Maximum** | 98.0 | 30.0 | 30.0 |
| **Range** | 86.0 | 30.0 | 30.0 |

### Distributional Analysis:
1. **BMI:** The average Body Mass Index (BMI) is **28.38**, which lies within the "Overweight" category (BMI 25.0 to 29.9) according to WHO guidelines. The minimum BMI is 12.0 and the maximum is 98.0. The standard deviation of 6.61 suggests significant variability, and a median of 27.0 implies a slight right skew due to extreme outliers on the higher end.
2. **Mental Health (`MentHlth`):** Represents the number of days in the past 30 days the respondent rated their mental health as "not good". The mean is **3.18 days** with a standard deviation of **7.41**. With Q1 = 0.0, median = 0.0, and Q3 = 2.0 (meaning over 75% of respondents report 2 or fewer bad days in the past 30 days), this distribution is heavily zero-inflated and right-skewed. The maximum is 30 days.
3. **Physical Health (`PhysHlth`):** Represents the number of days in the past 30 days the respondent rated their physical health as "not good". Similar to mental health, it is zero-inflated, showing a mean of **4.24 days**, standard deviation of **8.71**, and a median of **0.0 days**. The maximum is 30 days.

---

## 8. Unique Values and Categorical Support
For every column, the distinct values and unique counts were logged to verify the domain bounds of nominal and ordinal variables.

| Attribute | Unique Count | Distinct Coded Values / Range |
| :--- | :---: | :--- |
| `Diabetes_binary` | 2 | [0, 1] |
| `HighBP` | 2 | [0, 1] |
| `HighChol` | 2 | [0, 1] |
| `CholCheck` | 2 | [0, 1] |
| `BMI` | 84 | [12, ..., 98] (Total: 84 values) |
| `Smoker` | 2 | [0, 1] |
| `Stroke` | 2 | [0, 1] |
| `HeartDiseaseorAttack` | 2 | [0, 1] |
| `PhysActivity` | 2 | [0, 1] |
| `Fruits` | 2 | [0, 1] |
| `Veggies` | 2 | [0, 1] |
| `HvyAlcoholConsump` | 2 | [0, 1] |
| `AnyHealthcare` | 2 | [0, 1] |
| `NoDocbcCost` | 2 | [0, 1] |
| `GenHlth` | 5 | [1, 2, 3, 4, 5] |
| `MentHlth` | 31 | [0, ..., 30] (Total: 31 values) |
| `PhysHlth` | 31 | [0, ..., 30] (Total: 31 values) |
| `DiffWalk` | 2 | [0, 1] |
| `Sex` | 2 | [0, 1] |
| `Age` | 13 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] |
| `Education` | 6 | [1, 2, 3, 4, 5, 6] |
| `Income` | 8 | [1, 2, 3, 4, 5, 6, 7, 8] |

---

## 9. Data Range Validation Findings
A range validation check was conducted on critical variables to ensure they conform to the BRFSS standard coding schemes.

| Variable | Expected Range | Actual Min | Actual Max | Violations Count | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `BMI` | Non-negative value | 12.0 | 98.0 | 0 | **PASSED** |
| `MentHlth` | 0 to 30 days | 0.0 | 30.0 | 0 | **PASSED** |
| `PhysHlth` | 0 to 30 days | 0.0 | 30.0 | 0 | **PASSED** |
| `GenHlth` | 1 to 5 scale | 1.0 | 5.0 | 0 | **PASSED** |
| `Age` | 1 to 13 category scale | 1.0 | 13.0 | 0 | **PASSED** |
| `Education` | 1 to 6 category scale | 1.0 | 6.0 | 0 | **PASSED** |
| `Income` | 1 to 8 category scale | 1.0 | 8.0 | 0 | **PASSED** |

**Validation Assessment:**
All investigated variables adhered strictly to their expected boundary conditions:
- `BMI` values are strictly positive, ranging from 12 to 98.
- Mental and physical health days (`MentHlth`, `PhysHlth`) are bounded exactly within [0, 30].
- Coded scale variables for general health (`GenHlth`), Age (`Age`), Education (`Education`), and Income (`Income`) fall strictly within their categorical ranges defined by the CDC.
- No values outside the expected coded ranges were detected in the examined variables.

---

## 10. Key Observations and Recommendations for Modeling
1. **Sample Positive-Class Proportion:** The positive-class proportion of **13.93%** describes the unweighted BRFSS sample analyzed in this project. It should not be interpreted as a weighted national prevalence estimate. Preserving the natural sample distribution is essential for training models that reflect the analyzed sample balance.
2. **Missing Values:** No imputation is necessary, as there are zero missing records.
3. **Repeated Feature Profiles:** Exact repeated rows cannot be verified as repeated observations of the same respondent because the dataset does not provide a respondent identifier. They may represent different respondents sharing the same discretized demographic, lifestyle, and health profile. These repeated feature profiles were retained to preserve the original sample frequencies and class distribution.
4. **Scale Differences:** Downstream machine learning models (especially distance-based or gradient-descent algorithms like SVMs, KNNs, or Logistic Regression) will require scaling or standardization due to the difference in scales between variables like `BMI` (12–98) and binary indicators (0–1).
5. **Class Imbalance Handling:** For modeling, since data rebalancing is forbidden in this phase, evaluation metrics must prioritize **F1-Score, Recall, Precision, and Precision-Recall AUC (PR-AUC)** over accuracy. A naive baseline predicting "No reported diabetes" for all cases would achieve 86.07% accuracy but 0% recall, making such a result unsuitable for the screening-oriented objective defined in this project.

---
*Report automatically generated by the Data Understanding Module.*
