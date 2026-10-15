# Data Dictionary

## Overview

This document describes the variables contained in the **CDC Diabetes Health Indicators** dataset. Each variable is summarized by its data type, description, and possible values. The dictionary serves as a reference throughout data preprocessing, exploratory data analysis (EDA), statistical analysis, and machine learning modeling.

| Variable             | Type            | Category             | Description                                                      | Values                                  |
| -------------------- | --------------- | -------------------- | ---------------------------------------------------------------- | --------------------------------------- |
| Diabetes_binary      | Binary (Target) | Target               | Diabetes status                                                  | 0 = No reported diabetes, 1 = Prediabetes or diabetes |
| HighBP               | Binary          | Health Condition     | High blood pressure                                              | 0 = No, 1 = Yes                         |
| HighChol             | Binary          | Health Condition     | High cholesterol                                                 | 0 = No, 1 = Yes                         |
| CholCheck            | Binary          | Health Condition     | Cholesterol checked within the past 5 years                      | 0 = No, 1 = Yes                         |
| BMI                  | Numerical       | Physical Measurement | Body Mass Index                                                  | Integer                                 |
| Smoker               | Binary          | Lifestyle            | Has smoked at least 100 cigarettes in lifetime                   | 0 = No, 1 = Yes                         |
| Stroke               | Binary          | Health Condition     | History of stroke                                                | 0 = No, 1 = Yes                         |
| HeartDiseaseorAttack | Binary          | Health Condition     | History of coronary heart disease or heart attack                | 0 = No, 1 = Yes                         |
| PhysActivity         | Binary          | Lifestyle            | Physical activity during the past 30 days (excluding work)       | 0 = No, 1 = Yes                         |
| Fruits               | Binary          | Lifestyle            | Consumes fruit one or more times per day                         | 0 = No, 1 = Yes                         |
| Veggies              | Binary          | Lifestyle            | Consumes vegetables one or more times per day                    | 0 = No, 1 = Yes                         |
| HvyAlcoholConsump    | Binary          | Lifestyle            | Heavy alcohol consumption                                        | 0 = No, 1 = Yes                         |
| AnyHealthcare        | Binary          | Healthcare Access    | Has any form of healthcare coverage                              | 0 = No, 1 = Yes                         |
| NoDocbcCost          | Binary          | Healthcare Access    | Could not see a doctor because of cost                           | 0 = No, 1 = Yes                         |
| GenHlth              | Ordinal         | Health Status        | Self-rated general health                                        | 1 = Excellent ... 5 = Poor              |
| MentHlth             | Numerical       | Health Status        | Number of days with poor mental health during the past 30 days   | 0–30                                    |
| PhysHlth             | Numerical       | Health Status        | Number of days with poor physical health during the past 30 days | 0–30                                    |
| DiffWalk             | Binary          | Health Condition     | Serious difficulty walking or climbing stairs                    | 0 = No, 1 = Yes                         |
| Sex                  | Binary          | Demographic          | Biological sex                                                   | 0 = Female, 1 = Male                    |
| Age                  | Ordinal         | Demographic          | Age group                                                        | 1–13 (youngest to oldest age groups)    |
| Education            | Ordinal         | Socioeconomic        | Highest level of education completed                             | 1–6 (lowest to highest education level) |
| Income               | Ordinal         | Socioeconomic        | Annual household income category                                 | 1–8 (lowest to highest income category) |

---

## Variable Categories

The variables are grouped into the following categories to facilitate analysis.

| Category                  | Variables                                                           |
| ------------------------- | ------------------------------------------------------------------- |
| Target                    | Diabetes_binary                                                     |
| Health Conditions         | HighBP, HighChol, CholCheck, Stroke, HeartDiseaseorAttack, DiffWalk |
| Lifestyle Behaviors       | Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump            |
| Physical Measurement      | BMI                                                                 |
| Healthcare Access         | AnyHealthcare, NoDocbcCost                                          |
| General Health Status     | GenHlth, MentHlth, PhysHlth                                         |
| Demographic Information   | Sex, Age                                                            |
| Socioeconomic Information | Education, Income                                                   |

---

## Notes

* Binary variables are encoded as **0** and **1**, where **1** generally indicates the presence of a condition or behavior.
* Ordinal variables represent ordered categories and should be interpreted according to their ranking rather than as continuous measurements.
* BMI, MentHlth, and PhysHlth are numerical variables and may require descriptive statistical analysis before modeling.
* Age, Education, and Income are ordinal categories rather than exact numerical values.
* The target variable (**Diabetes_binary**) is imbalanced because this project intentionally uses the original dataset to preserve the real-world distribution of diabetes cases.
