# Dataset Description

## 1. Dataset Overview

This project uses the **CDC Diabetes Health Indicators** dataset, which is derived from the **Behavioral Risk Factor Surveillance System (BRFSS) 2015** conducted by the Centers for Disease Control and Prevention (CDC). The BRFSS is one of the largest health-related telephone surveys in the United States, collecting information on chronic diseases, health conditions, lifestyle behaviors, and access to healthcare services.

Each record in the dataset represents one survey respondent, while each feature describes a demographic characteristic, health condition, or lifestyle behavior associated with that individual.

---

## 2. Dataset Source

* **Dataset Name:** CDC Diabetes Health Indicators
* **Original Source:** Behavioral Risk Factor Surveillance System (BRFSS) 2015
* **Organization:** Centers for Disease Control and Prevention (CDC)
* **Distribution Platform:** UCI Machine Learning Repository / Kaggle
* **Dataset Type:** Structured tabular dataset
* **Prediction Task:** Binary classification

---

## 3. Dataset Characteristics

| Attribute                | Description                              |
| ------------------------ | ---------------------------------------- |
| Number of observations   | 253,680                                  |
| Number of input features | 21                                       |
| Target variable          | Diabetes_binary                          |
| Data type                | Tabular                                  |
| Feature types            | Binary, ordinal, and numerical           |
| Missing values           | None                                     |
| Duplicate records        | To be verified during data preprocessing |
| Class distribution       | Imbalanced (original dataset)            |

---

## 4. Target Variable

The prediction target is **Diabetes_binary**, which indicates whether an individual has diabetes.

| Value | Description |
| ----- | ----------- |
| 0     | No reported diabetes |
| 1     | Prediabetes or diabetes |


This project uses the **original imbalanced dataset**, preserving the natural distribution of diabetes cases to better reflect real-world conditions.

---

## 5. Feature Categories

The dataset contains variables that can be grouped into several categories.

### 5.1 Health Conditions

* HighBP
* HighChol
* CholCheck
* Stroke
* HeartDiseaseorAttack
* DiffWalk
* GenHlth
* MentHlth
* PhysHlth

### 5.2 Lifestyle Behaviors

* Smoker
* PhysActivity
* Fruits
* Veggies
* HvyAlcoholConsump

### 5.3 Physical Measurement

* BMI

### 5.4 Healthcare Access

* AnyHealthcare
* NoDocbcCost

### 5.5 Demographic and Socioeconomic Information

* Sex
* Age
* Education
* Income

---

## 6. Data Types

The dataset contains three primary variable types.

| Variable Type | Examples                         |
| ------------- | -------------------------------- |
| Binary        | HighBP, HighChol, Smoker, Stroke |
| Ordinal       | GenHlth, Age, Education, Income  |
| Numerical     | BMI, MentHlth, PhysHlth          |

---

## 7. Data Quality

Based on the dataset documentation, no missing values are expected. Nevertheless, data quality verification will be performed during preprocessing to confirm:

* Missing values
* Duplicate records
* Invalid feature values
* Data type consistency
* Target class distribution

These validation steps ensure that the processed dataset is suitable for subsequent statistical analysis and machine learning.

---

## 8. Intended Use in This Project

The dataset will be used throughout the entire analytical workflow, including:

* Data Understanding
* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Statistical Analysis
* Feature Engineering
* Machine Learning Modeling
* Model Evaluation
* Explainable Artificial Intelligence (SHAP)

Using a single dataset throughout the project ensures consistency between exploratory analysis, predictive modeling, and model interpretation.
