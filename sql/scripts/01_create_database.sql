-- ==========================================
-- 01_CREATE_DATABASE.SQL
-- Author: Senior SQL Server Data Analyst
-- Project: Diabetes-Analytics
-- Target DB: Microsoft SQL Server
-- ==========================================

-- 1. Create the DiabetesAnalytics database if it does not exist
IF DB_ID('DiabetesAnalytics') IS NULL
BEGIN
    CREATE DATABASE DiabetesAnalytics;
END;
GO

-- Switch context to the database
USE DiabetesAnalytics;
GO

-- 2. Create the main table for diabetes health indicators if it does not exist.
-- TINYINT is chosen for all fields because all indicators are discrete integers within 0-255:
-- Binary flags: 0-1
-- BMI range: 12-98
-- Other indices (MentHlth, PhysHlth, GenHlth, Age, Education, Income): 0-30
IF OBJECT_ID('dbo.diabetes_health_indicators', 'U') IS NULL
BEGIN
    CREATE TABLE diabetes_health_indicators (
        Diabetes_binary TINYINT NOT NULL,
        HighBP TINYINT NOT NULL,
        HighChol TINYINT NOT NULL,
        CholCheck TINYINT NOT NULL,
        BMI TINYINT NOT NULL,
        Smoker TINYINT NOT NULL,
        Stroke TINYINT NOT NULL,
        HeartDiseaseorAttack TINYINT NOT NULL,
        PhysActivity TINYINT NOT NULL,
        Fruits TINYINT NOT NULL,
        Veggies TINYINT NOT NULL,
        HvyAlcoholConsump TINYINT NOT NULL,
        AnyHealthcare TINYINT NOT NULL,
        NoDocbcCost TINYINT NOT NULL,
        GenHlth TINYINT NOT NULL,
        MentHlth TINYINT NOT NULL,
        PhysHlth TINYINT NOT NULL,
        DiffWalk TINYINT NOT NULL,
        Sex TINYINT NOT NULL,
        Age TINYINT NOT NULL,
        Education TINYINT NOT NULL,
        Income TINYINT NOT NULL
    );
END;
GO

