-- POS_CASH monthly history → applicant aggregates
WITH src AS (
  SELECT * FROM 'data/interim/pos_cash_balance_clean.parquet'
),
hist AS (
  SELECT * FROM src WHERE COALESCE(MONTHS_BALANCE, -999999) <= 0
)
SELECT
  SK_ID_CURR,
  COUNT(*)                                    AS POS_N_MONTHS,
  MAX(MONTHS_BALANCE)                         AS POS_LAST_MONTH,
  AVG(SK_DPD)                                 AS POS_DPD_MEAN,
  MAX(SK_DPD)                                 AS POS_DPD_MAX,
  AVG(SK_DPD_DEF)                             AS POS_DPD_DEF_MEAN,
  MAX(SK_DPD_DEF)                             AS POS_DPD_DEF_MAX,
  AVG(CASE WHEN NAME_CONTRACT_STATUS='Active' THEN 1 ELSE 0 END)     AS POS_PCT_ACTIVE,
  AVG(CASE WHEN NAME_CONTRACT_STATUS='Completed' THEN 1 ELSE 0 END)  AS POS_PCT_COMPLETED
FROM hist
GROUP BY SK_ID_CURR;