-- =========================================================================
-- 00_IMPORT_DATA.SQL
-- Author: Senior SQL Server Data Analyst
-- Project: Diabetes-Analytics
-- Target DB: Microsoft SQL Server
-- Description: BULK INSERT template for populating diabetes_health_indicators
--              from processed CSV data.
-- =========================================================================

USE DiabetesAnalytics;
GO

-- Replace <PATH_TO_CSV> with the absolute path to data/processed/diabetes_cleaned.csv on your server.
-- Example: 'C:\path\to\Diabetes-Analytics\data\processed\diabetes_cleaned.csv'

BULK INSERT diabetes_health_indicators
FROM '<PATH_TO_CSV>'
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
);
GO
