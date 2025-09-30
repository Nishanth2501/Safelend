-- Previous applications before current application
WITH src AS (
  SELECT * FROM 'data/interim/previous_application_clean.parquet'
),
hist AS (
  SELECT * FROM src WHERE COALESCE(DAYS_DECISION, -999999) <= 0
)
SELECT
  SK_ID_CURR,
  COUNT(*)                                                      AS PREV_N_APPS,
  SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS PREV_N_APPROVED,
  SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END) AS PREV_N_REFUSED,
  AVG(AMT_CREDIT)                                               AS PREV_AMT_CREDIT_MEAN,
  SUM(AMT_CREDIT)                                               AS PREV_AMT_CREDIT_SUM,
  AVG(AMT_ANNUITY)                                              AS PREV_AMT_ANNUITY_MEAN,
  AVG(CNT_PAYMENT)                                              AS PREV_CNT_PAYMENT_MEAN,
  MAX(DAYS_DECISION)                                            AS PREV_DAYS_LAST_DECISION_MAX
FROM hist
GROUP BY SK_ID_CURR;