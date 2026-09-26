-- ============================================================================
-- churn_analysis_queries.sql
-- Business-facing SQL analysis on the customer_churn table.
-- Compatible with MySQL 8+ / PostgreSQL / SQLite (minor syntax tweaks noted).
--
-- To load the CSV into a table first (MySQL example):
--
--   CREATE TABLE customer_churn (
--       customerID          VARCHAR(20),
--       gender              VARCHAR(10),
--       SeniorCitizen       INT,
--       Partner             VARCHAR(5),
--       Dependents          VARCHAR(5),
--       tenure              INT,
--       PhoneService        VARCHAR(5),
--       MultipleLines       VARCHAR(20),
--       InternetService     VARCHAR(15),
--       OnlineSecurity      VARCHAR(20),
--       OnlineBackup        VARCHAR(20),
--       DeviceProtection    VARCHAR(20),
--       TechSupport         VARCHAR(20),
--       StreamingTV         VARCHAR(20),
--       StreamingMovies     VARCHAR(20),
--       Contract            VARCHAR(20),
--       PaperlessBilling    VARCHAR(5),
--       PaymentMethod       VARCHAR(30),
--       MonthlyCharges      DECIMAL(8,2),
--       TotalCharges        DECIMAL(10,2),
--       Churn               VARCHAR(5)
--   );
--
--   LOAD DATA LOCAL INFILE 'data/raw/customer_churn.csv'
--   INTO TABLE customer_churn
--   FIELDS TERMINATED BY ',' ENCLOSED BY '"'
--   LINES TERMINATED BY '\n'
--   IGNORE 1 ROWS;
-- ============================================================================


-- 1. Overall churn rate
SELECT
    COUNT(*)                                              AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)        AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn;


-- 2. Churn rate by contract type (month-to-month customers are usually highest risk)
SELECT
    Contract,
    COUNT(*)                                              AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)        AS churned,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- 3. Churn rate by internet service type
SELECT
    InternetService,
    COUNT(*)                                              AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                         AS avg_monthly_charges
FROM customer_churn
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- 4. Tenure buckets vs churn (early-life customers churn the most)
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-1 yr'
        WHEN tenure <= 24 THEN '1-2 yr'
        WHEN tenure <= 48 THEN '2-4 yr'
        ELSE '4+ yr'
    END AS tenure_group,
    COUNT(*)                                              AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY tenure_group
ORDER BY MIN(tenure);


-- 5. Payment method vs churn
SELECT
    PaymentMethod,
    COUNT(*)                                              AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- 6. Revenue at risk: total monthly revenue currently sitting with predicted-risk segments
--    (Month-to-month + Fiber optic + no tech support = classic high-risk combo)
SELECT
    COUNT(*)                                              AS high_risk_customers,
    ROUND(SUM(MonthlyCharges), 2)                         AS monthly_revenue_at_risk,
    ROUND(SUM(MonthlyCharges) * 12, 2)                    AS annualized_revenue_at_risk
FROM customer_churn
WHERE Contract = 'Month-to-month'
  AND InternetService = 'Fiber optic'
  AND TechSupport = 'No'
  AND Churn = 'No';   -- customers still active today but matching the risk profile


-- 7. Add-on services vs churn (does bundling reduce churn?)
SELECT
    OnlineSecurity,
    TechSupport,
    COUNT(*)                                              AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
WHERE InternetService != 'No'
GROUP BY OnlineSecurity, TechSupport
ORDER BY churn_rate_pct DESC;


-- 8. Top 20 highest-value customers currently at churn risk (for a retention call list)
SELECT
    customerID,
    Contract,
    InternetService,
    tenure,
    MonthlyCharges,
    TotalCharges
FROM customer_churn
WHERE Churn = 'No'
  AND Contract = 'Month-to-month'
ORDER BY MonthlyCharges DESC
LIMIT 20;


-- 9. Average tenure and spend, churned vs retained
SELECT
    Churn,
    COUNT(*)                                              AS customers,
    ROUND(AVG(tenure), 1)                                 AS avg_tenure_months,
    ROUND(AVG(MonthlyCharges), 2)                         AS avg_monthly_charges,
    ROUND(AVG(TotalCharges), 2)                           AS avg_total_charges
FROM customer_churn
GROUP BY Churn;


-- 10. Senior citizens vs churn
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS customer_type,
    COUNT(*)                                              AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY customer_type;
