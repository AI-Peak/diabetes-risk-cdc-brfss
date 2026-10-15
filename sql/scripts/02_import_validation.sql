-- =========================================================================
-- 02_IMPORT_VALIDATION.SQL
-- Author: Senior SQL Server Data Analyst
-- Project: Diabetes-Analytics
-- Target DB: Microsoft SQL Server
-- Description: Queries to validate the integrity of the imported data.
--              Includes metadata tags for Python automation.
-- =========================================================================

USE DiabetesAnalytics;
GO

-- Query 1: Total row count
-- SAVE_AS: total_row_count.csv
SELECT COUNT(*) AS TotalRows 
FROM diabetes_health_indicators;
GO


-- Query 2: Distribution of Diabetes_binary
-- SAVE_AS: target_distribution.csv
SELECT 
    Diabetes_binary,
    COUNT(*) AS RecordCount,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS DECIMAL(5, 2)) AS Percentage
FROM diabetes_health_indicators
GROUP BY Diabetes_binary;
GO


-- Query 3: NULL value count for every column
-- SAVE_AS: missing_values.csv
SELECT 
    SUM(CASE WHEN Diabetes_binary IS NULL THEN 1 ELSE 0 END) AS NullCount_Diabetes_binary,
    SUM(CASE WHEN HighBP IS NULL THEN 1 ELSE 0 END) AS NullCount_HighBP,
    SUM(CASE WHEN HighChol IS NULL THEN 1 ELSE 0 END) AS NullCount_HighChol,
    SUM(CASE WHEN CholCheck IS NULL THEN 1 ELSE 0 END) AS NullCount_CholCheck,
    SUM(CASE WHEN BMI IS NULL THEN 1 ELSE 0 END) AS NullCount_BMI,
    SUM(CASE WHEN Smoker IS NULL THEN 1 ELSE 0 END) AS NullCount_Smoker,
    SUM(CASE WHEN Stroke IS NULL THEN 1 ELSE 0 END) AS NullCount_Stroke,
    SUM(CASE WHEN HeartDiseaseorAttack IS NULL THEN 1 ELSE 0 END) AS NullCount_HeartDiseaseorAttack,
    SUM(CASE WHEN PhysActivity IS NULL THEN 1 ELSE 0 END) AS NullCount_PhysActivity,
    SUM(CASE WHEN Fruits IS NULL THEN 1 ELSE 0 END) AS NullCount_Fruits,
    SUM(CASE WHEN Veggies IS NULL THEN 1 ELSE 0 END) AS NullCount_Veggies,
    SUM(CASE WHEN HvyAlcoholConsump IS NULL THEN 1 ELSE 0 END) AS NullCount_HvyAlcoholConsump,
    SUM(CASE WHEN AnyHealthcare IS NULL THEN 1 ELSE 0 END) AS NullCount_AnyHealthcare,
    SUM(CASE WHEN NoDocbcCost IS NULL THEN 1 ELSE 0 END) AS NullCount_NoDocbcCost,
    SUM(CASE WHEN GenHlth IS NULL THEN 1 ELSE 0 END) AS NullCount_GenHlth,
    SUM(CASE WHEN MentHlth IS NULL THEN 1 ELSE 0 END) AS NullCount_MentHlth,
    SUM(CASE WHEN PhysHlth IS NULL THEN 1 ELSE 0 END) AS NullCount_PhysHlth,
    SUM(CASE WHEN DiffWalk IS NULL THEN 1 ELSE 0 END) AS NullCount_DiffWalk,
    SUM(CASE WHEN Sex IS NULL THEN 1 ELSE 0 END) AS NullCount_Sex,
    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS NullCount_Age,
    SUM(CASE WHEN Education IS NULL THEN 1 ELSE 0 END) AS NullCount_Education,
    SUM(CASE WHEN Income IS NULL THEN 1 ELSE 0 END) AS NullCount_Income
FROM diabetes_health_indicators;
GO


-- Query 4: Duplicate row count
-- SAVE_AS: duplicate_summary.csv
WITH RankedRows AS (
    SELECT 
        *,
        ROW_NUMBER() OVER(
            PARTITION BY 
                Diabetes_binary, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, 
                HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, 
                AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, 
                Sex, Age, Education, Income
            ORDER BY (SELECT NULL)
        ) AS RowNum
    FROM diabetes_health_indicators
)
SELECT COUNT(*) AS DuplicateCount
FROM RankedRows
WHERE RowNum > 1;
GO


-- Query 5: Unique value count for every column
-- SAVE_AS: unique_value_counts.csv
SELECT 
    COUNT(DISTINCT Diabetes_binary) AS Unique_Diabetes_binary,
    COUNT(DISTINCT HighBP) AS Unique_HighBP,
    COUNT(DISTINCT HighChol) AS Unique_HighChol,
    COUNT(DISTINCT CholCheck) AS Unique_CholCheck,
    COUNT(DISTINCT BMI) AS Unique_BMI,
    COUNT(DISTINCT Smoker) AS Unique_Smoker,
    COUNT(DISTINCT Stroke) AS Unique_Stroke,
    COUNT(DISTINCT HeartDiseaseorAttack) AS Unique_HeartDiseaseorAttack,
    COUNT(DISTINCT PhysActivity) AS Unique_PhysActivity,
    COUNT(DISTINCT Fruits) AS Unique_Fruits,
    COUNT(DISTINCT Veggies) AS Unique_Veggies,
    COUNT(DISTINCT HvyAlcoholConsump) AS Unique_HvyAlcoholConsump,
    COUNT(DISTINCT AnyHealthcare) AS Unique_AnyHealthcare,
    COUNT(DISTINCT NoDocbcCost) AS Unique_NoDocbcCost,
    COUNT(DISTINCT GenHlth) AS Unique_GenHlth,
    COUNT(DISTINCT MentHlth) AS Unique_MentHlth,
    COUNT(DISTINCT PhysHlth) AS Unique_PhysHlth,
    COUNT(DISTINCT DiffWalk) AS Unique_DiffWalk,
    COUNT(DISTINCT Sex) AS Unique_Sex,
    COUNT(DISTINCT Age) AS Unique_Age,
    COUNT(DISTINCT Education) AS Unique_Education,
    COUNT(DISTINCT Income) AS Unique_Income
FROM diabetes_health_indicators;
GO


-- Query 6: Minimum and Maximum values of specific columns
-- SAVE_AS: key_features_min_max.csv
SELECT 
    MIN(BMI) AS Min_BMI, MAX(BMI) AS Max_BMI,
    MIN(Age) AS Min_Age, MAX(Age) AS Max_Age,
    MIN(Education) AS Min_Education, MAX(Education) AS Max_Education,
    MIN(Income) AS Min_Income, MAX(Income) AS Max_Income,
    MIN(GenHlth) AS Min_GenHlth, MAX(GenHlth) AS Max_GenHlth,
    MIN(MentHlth) AS Min_MentHlth, MAX(MentHlth) AS Max_MentHlth,
    MIN(PhysHlth) AS Min_PhysHlth, MAX(PhysHlth) AS Max_PhysHlth
FROM diabetes_health_indicators;
GO


-- Query 7: Range validation for all variables (count of out-of-range rows)
-- SAVE_AS: range_validation.csv
SELECT 
    SUM(CASE WHEN Diabetes_binary NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_Diabetes_binary,
    SUM(CASE WHEN HighBP NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_HighBP,
    SUM(CASE WHEN HighChol NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_HighChol,
    SUM(CASE WHEN CholCheck NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_CholCheck,
    SUM(CASE WHEN BMI <= 0 THEN 1 ELSE 0 END) AS Invalid_BMI,
    SUM(CASE WHEN Smoker NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_Smoker,
    SUM(CASE WHEN Stroke NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_Stroke,
    SUM(CASE WHEN HeartDiseaseorAttack NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_HeartDiseaseorAttack,
    SUM(CASE WHEN PhysActivity NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_PhysActivity,
    SUM(CASE WHEN Fruits NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_Fruits,
    SUM(CASE WHEN Veggies NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_Veggies,
    SUM(CASE WHEN HvyAlcoholConsump NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_HvyAlcoholConsump,
    SUM(CASE WHEN AnyHealthcare NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_AnyHealthcare,
    SUM(CASE WHEN NoDocbcCost NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_NoDocbcCost,
    SUM(CASE WHEN GenHlth NOT BETWEEN 1 AND 5 THEN 1 ELSE 0 END) AS Invalid_GenHlth,
    SUM(CASE WHEN MentHlth NOT BETWEEN 0 AND 30 THEN 1 ELSE 0 END) AS Invalid_MentHlth,
    SUM(CASE WHEN PhysHlth NOT BETWEEN 0 AND 30 THEN 1 ELSE 0 END) AS Invalid_PhysHlth,
    SUM(CASE WHEN DiffWalk NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_DiffWalk,
    SUM(CASE WHEN Sex NOT BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS Invalid_Sex,
    SUM(CASE WHEN Age NOT BETWEEN 1 AND 13 THEN 1 ELSE 0 END) AS Invalid_Age,
    SUM(CASE WHEN Education NOT BETWEEN 1 AND 6 THEN 1 ELSE 0 END) AS Invalid_Education,
    SUM(CASE WHEN Income NOT BETWEEN 1 AND 8 THEN 1 ELSE 0 END) AS Invalid_Income
FROM diabetes_health_indicators;
GO
