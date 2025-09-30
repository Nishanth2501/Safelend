-- Credit card behavior: utilization & payment discipline
WITH src AS (
  SELECT * FROM 'data/interim/credit_card_balance_clean.parquet'
),
hist AS (
  SELECT * FROM src WHERE COALESCE(MONTHS_BALANCE, -999999) <= 0
),
util AS (
  SELECT
    SK_ID_CURR,
    CASE 
      WHEN AMT_CREDIT_LIMIT_ACTUAL > 0 THEN AMT_BALANCE * 1.0 / AMT_CREDIT_LIMIT_ACTUAL
      ELSE NULL
    END AS util,
    AMT_CREDIT_LIMIT_ACTUAL,
    AMT_PAYMENT_CURRENT,
    AMT_INST_MIN_REGULARITY,
    SK_DPD,
    SK_DPD_DEF,
    CNT_DRAWINGS_CURRENT
  FROM hist
)
SELECT
  SK_ID_CURR,
  AVG(util)                                      AS CC_UTIL_MEAN,
  MAX(util)                                      AS CC_UTIL_MAX,
  AVG(AMT_CREDIT_LIMIT_ACTUAL)                   AS CC_LIMIT_MEAN,
  AVG(CASE 
        WHEN AMT_INST_MIN_REGULARITY IS NOT NULL AND AMT_INST_MIN_REGULARITY > 0 
        THEN LEAST(AMT_PAYMENT_CURRENT / AMT_INST_MIN_REGULARITY, 5.0)
        ELSE NULL
      END)                                       AS CC_MINPAY_RATIO_MEAN,
  AVG(CASE 
        WHEN AMT_INST_MIN_REGULARITY IS NOT NULL AND AMT_INST_MIN_REGULARITY > 0 
        THEN CASE WHEN AMT_PAYMENT_CURRENT >= AMT_INST_MIN_REGULARITY THEN 1 ELSE 0 END
        ELSE NULL
      END)                                       AS CC_PCT_MONTHS_PAID,
  SUM(CNT_DRAWINGS_CURRENT)                       AS CC_DRAWINGS_COUNT,
  MAX(SK_DPD)                                     AS CC_DPD_MAX,
  MAX(SK_DPD_DEF)                                 AS CC_DPD_DEF_MAX
FROM util
GROUP BY SK_ID_CURR;