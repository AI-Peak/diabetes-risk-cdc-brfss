-- =========================================================================
-- 03_EDA.SQL
-- Author: Senior SQL Server Data Analyst
-- Project: Diabetes-Analytics
-- Target DB: Microsoft SQL Server
-- Description: Exploratory Data Analysis (EDA) on the CDC Diabetes dataset.
--              Includes metadata tags for Python automation.
-- =========================================================================

USE DiabetesAnalytics;
GO

-- =========================================================================
-- SECTION 1: DATASET OVERVIEW & TARGET DISTRIBUTION
-- =========================================================================

-- Query 1: Target Variable (Diabetes_binary) Distribution
-- SAVE_AS: eda_target_distribution.csv
SELECT 
    Diabetes_binary,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage
FROM diabetes_health_indicators
GROUP BY Diabetes_binary;
GO


-- =========================================================================
-- SECTION 2: DEMOGRAPHIC ANALYSIS
-- =========================================================================

-- Query 2: Sex vs Diabetes
-- Sex: 0 = Female, 1 = Male
-- SAVE_AS: sex_vs_diabetes.csv
SELECT 
    Sex,
    CASE Sex WHEN 0 THEN 'Female' WHEN 1 THEN 'Male' END AS Sex_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY Sex
ORDER BY Sex;
GO

-- Query 3: Age vs Diabetes
-- Age: 13-level category (1 = 18-24, 2 = 25-29, ..., 13 = 80 or older)
-- SAVE_AS: age_vs_diabetes.csv
SELECT 
    Age,
    CASE Age 
        WHEN 1 THEN '18-24' WHEN 2 THEN '25-29' WHEN 3 THEN '30-34' 
        WHEN 4 THEN '35-39' WHEN 5 THEN '40-44' WHEN 6 THEN '45-49' 
        WHEN 7 THEN '50-54' WHEN 8 THEN '55-59' WHEN 9 THEN '60-64' 
        WHEN 10 THEN '65-69' WHEN 11 THEN '70-74' WHEN 12 THEN '75-79' 
        WHEN 13 THEN '80+' 
    END AS Age_Range,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY Age
ORDER BY Age;
GO

-- Query 4: Education vs Diabetes
-- Education: 1 = Never attended school/kindergarten, ..., 6 = College Graduate+
-- SAVE_AS: education_vs_diabetes.csv
SELECT 
    Education,
    CASE Education 
        WHEN 1 THEN 'Never attended/Kindergarten'
        WHEN 2 THEN 'Elementary (1-8)'
        WHEN 3 THEN 'Some High School'
        WHEN 4 THEN 'High School Graduate'
        WHEN 5 THEN 'Some College'
        WHEN 6 THEN 'College Graduate+'
    END AS Education_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY Education
ORDER BY Education;
GO

-- Query 5: Income vs Diabetes
-- Income: 1 = Less than $10,000, ..., 8 = $75,000 or more
-- SAVE_AS: income_vs_diabetes.csv
SELECT 
    Income,
    CASE Income 
        WHEN 1 THEN '< $10k' WHEN 2 THEN '$10k-$15k' WHEN 3 THEN '$15k-$20k' 
        WHEN 4 THEN '$20k-$25k' WHEN 5 THEN '$25k-$35k' WHEN 6 THEN '$35k-$50k' 
        WHEN 7 THEN '$50k-$75k' WHEN 8 THEN '>= $75k' 
    END AS Income_Range,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY Income
ORDER BY Income;
GO


-- =========================================================================
-- SECTION 3: HEALTH CONDITION ANALYSIS
-- =========================================================================

-- Query 6: HighBP vs Diabetes
-- HighBP: 0 = No high blood pressure, 1 = High blood pressure
-- SAVE_AS: highbp_vs_diabetes.csv
SELECT 
    HighBP,
    CASE HighBP WHEN 0 THEN 'No High BP' WHEN 1 THEN 'High BP' END AS HighBP_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY HighBP
ORDER BY HighBP;
GO

-- Query 7: HighChol vs Diabetes
-- HighChol: 0 = No high cholesterol, 1 = High cholesterol
-- SAVE_AS: highchol_vs_diabetes.csv
SELECT 
    HighChol,
    CASE HighChol WHEN 0 THEN 'No High Chol' WHEN 1 THEN 'High Chol' END AS HighChol_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY HighChol
ORDER BY HighChol;
GO

-- Query 8: Stroke vs Diabetes
-- Stroke: 0 = No stroke history, 1 = History of stroke
-- SAVE_AS: stroke_vs_diabetes.csv
SELECT 
    Stroke,
    CASE Stroke WHEN 0 THEN 'No Stroke History' WHEN 1 THEN 'Stroke History' END AS Stroke_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY Stroke
ORDER BY Stroke;
GO

-- Query 9: HeartDiseaseorAttack vs Diabetes
-- HeartDiseaseorAttack: 0 = No coronary heart disease or MI, 1 = History of CHD/MI
-- SAVE_AS: heartdisease_vs_diabetes.csv
SELECT 
    HeartDiseaseorAttack,
    CASE HeartDiseaseorAttack WHEN 0 THEN 'No Heart Disease' WHEN 1 THEN 'Heart Disease' END AS HeartDisease_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY HeartDiseaseorAttack
ORDER BY HeartDiseaseorAttack;
GO

-- Query 10: DiffWalk vs Diabetes
-- DiffWalk: 0 = No difficulty walking/climbing stairs, 1 = Having difficulty walking/climbing stairs
-- SAVE_AS: diffwalk_vs_diabetes.csv
SELECT 
    DiffWalk,
    CASE DiffWalk WHEN 0 THEN 'No Diff Walking' WHEN 1 THEN 'Diff Walking' END AS DiffWalk_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY DiffWalk
ORDER BY DiffWalk;
GO

-- Query 11: GenHlth vs Diabetes
-- GenHlth: Self-reported general health scale (1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor)
-- SAVE_AS: genhlth_vs_diabetes.csv
SELECT 
    GenHlth,
    CASE GenHlth 
        WHEN 1 THEN 'Excellent' WHEN 2 THEN 'Very Good' WHEN 3 THEN 'Good' 
        WHEN 4 THEN 'Fair' WHEN 5 THEN 'Poor' 
    END AS GenHlth_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY GenHlth
ORDER BY GenHlth;
GO

-- Query 12: BMI statistics by Diabetes (Numerical variable)
-- SAVE_AS: bmi_statistics.csv
SELECT 
    Diabetes_binary,
    AVG(CAST(BMI AS FLOAT)) AS Mean_BMI,
    STDEV(BMI) AS StdDev_BMI,
    MIN(BMI) AS Min_BMI,
    MAX(BMI) AS Max_BMI
FROM diabetes_health_indicators
GROUP BY Diabetes_binary;
GO

-- Query 13: MentHlth statistics by Diabetes (Numerical variable, days of poor mental health in past 30 days)
-- SAVE_AS: menthlth_statistics.csv
SELECT 
    Diabetes_binary,
    AVG(CAST(MentHlth AS FLOAT)) AS Mean_MentHlth,
    STDEV(MentHlth) AS StdDev_MentHlth,
    MIN(MentHlth) AS Min_MentHlth,
    MAX(MentHlth) AS Max_MentHlth
FROM diabetes_health_indicators
GROUP BY Diabetes_binary;
GO

-- Query 14: PhysHlth statistics by Diabetes (Numerical variable, days of poor physical health in past 30 days)
-- SAVE_AS: physhlth_statistics.csv
SELECT 
    Diabetes_binary,
    AVG(CAST(PhysHlth AS FLOAT)) AS Mean_PhysHlth,
    STDEV(PhysHlth) AS StdDev_PhysHlth,
    MIN(PhysHlth) AS Min_PhysHlth,
    MAX(PhysHlth) AS Max_PhysHlth
FROM diabetes_health_indicators
GROUP BY Diabetes_binary;
GO


-- =========================================================================
-- SECTION 4: LIFESTYLE ANALYSIS
-- =========================================================================

-- Query 15: Smoking vs Diabetes
-- Smoker: Have smoked at least 100 cigarettes (0 = No, 1 = Yes)
-- SAVE_AS: smoker_vs_diabetes.csv
SELECT 
    Smoker,
    CASE Smoker WHEN 0 THEN 'Non-Smoker' WHEN 1 THEN 'Smoker' END AS Smoker_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY Smoker
ORDER BY Smoker;
GO

-- Query 16: Physical Activity vs Diabetes
-- PhysActivity: Physical activity in past 30 days, excluding work (0 = No, 1 = Yes)
-- SAVE_AS: physactivity_vs_diabetes.csv
SELECT 
    PhysActivity,
    CASE PhysActivity WHEN 0 THEN 'Inactive' WHEN 1 THEN 'Active' END AS PhysActivity_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY PhysActivity
ORDER BY PhysActivity;
GO

-- Query 17: Fruit Consumption vs Diabetes
-- Fruits: Consume Fruit 1 or more times per day (0 = No, 1 = Yes)
-- SAVE_AS: fruits_vs_diabetes.csv
SELECT 
    Fruits,
    CASE Fruits WHEN 0 THEN 'No Daily Fruit' WHEN 1 THEN 'Daily Fruit' END AS Fruits_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY Fruits
ORDER BY Fruits;
GO

-- Query 18: Vegetable Consumption vs Diabetes
-- Veggies: Consume Vegetables 1 or more times per day (0 = No, 1 = Yes)
-- SAVE_AS: veggies_vs_diabetes.csv
SELECT 
    Veggies,
    CASE Veggies WHEN 0 THEN 'No Daily Veggies' WHEN 1 THEN 'Daily Veggies' END AS Veggies_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY Veggies
ORDER BY Veggies;
GO

-- Query 19: Heavy Alcohol Consumption vs Diabetes
-- HvyAlcoholConsump: Adult men >14 drinks/week, adult women >7 drinks/week (0 = No, 1 = Yes)
-- SAVE_AS: hvyalcoholconsump_vs_diabetes.csv
SELECT 
    HvyAlcoholConsump,
    CASE HvyAlcoholConsump WHEN 0 THEN 'No Heavy Drinking' WHEN 1 THEN 'Heavy Drinking' END AS Alcohol_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY HvyAlcoholConsump
ORDER BY HvyAlcoholConsump;
GO


-- =========================================================================
-- SECTION 5: HEALTHCARE ACCESS
-- =========================================================================

-- Query 20: AnyHealthcare vs Diabetes
-- AnyHealthcare: Has any healthcare coverage (0 = No, 1 = Yes)
-- SAVE_AS: anyhealthcare_vs_diabetes.csv
SELECT 
    AnyHealthcare,
    CASE AnyHealthcare WHEN 0 THEN 'No Healthcare' WHEN 1 THEN 'Healthcare' END AS Healthcare_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY AnyHealthcare
ORDER BY AnyHealthcare;
GO

-- Query 21: NoDocbcCost vs Diabetes
-- NoDocbcCost: Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? (0 = No, 1 = Yes)
-- SAVE_AS: nodocbccost_vs_diabetes.csv
SELECT 
    NoDocbcCost,
    CASE NoDocbcCost WHEN 0 THEN 'No Cost Barrier' WHEN 1 THEN 'Cost Barrier' END AS CostBarrier_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY NoDocbcCost
ORDER BY NoDocbcCost;
GO

-- Query 22: CholCheck vs Diabetes
-- CholCheck: Cholesterol check in past 5 years (0 = No, 1 = Yes)
-- SAVE_AS: cholcheck_vs_diabetes.csv
SELECT 
    CholCheck,
    CASE CholCheck WHEN 0 THEN 'No Chol Check' WHEN 1 THEN 'Chol Check' END AS CholCheck_Label,
    COUNT(*) AS Frequency,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage,
    CAST(SUM(CAST(Diabetes_binary AS FLOAT)) * 100.0 / COUNT(*) AS DECIMAL(5, 2)) AS DiabetesRate_Pct
FROM diabetes_health_indicators
GROUP BY CholCheck
ORDER BY CholCheck;
GO
