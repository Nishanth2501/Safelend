-- Installments: late/shortfall signals
WITH src AS (
  SELECT * FROM 'data/interim/installments_payments_clean.parquet'
),
feat AS (
  SELECT
    SK_ID_CURR,
    GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0) AS late_days,
    GREATEST(AMT_INSTALMENT - AMT_PAYMENT, 0)         AS shortfall
  FROM src
)
SELECT
  SK_ID_CURR,
  COUNT(*)                                        AS INST_N_PAYMENTS,
  AVG(CASE WHEN late_days > 0 THEN 1 ELSE 0 END)  AS INST_LATE_SHARE,
  AVG(late_days)                                   AS INST_LATE_MEAN,
  SUM(shortfall)                                   AS INST_SHORTFALL_SUM,
  AVG(shortfall)                                   AS INST_SHORTFALL_MEAN
FROM feat
GROUP BY SK_ID_CURR;