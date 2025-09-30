-- Bureau: one row per SK_ID_CURR summarizing prior credits
WITH src AS (
  SELECT * FROM 'data/interim/bureau_clean.parquet'
)
, hist AS (
  -- DAYS_CREDIT <= 0 means recorded before/at application
  SELECT * FROM src WHERE COALESCE(DAYS_CREDIT, -999999) <= 0
)
SELECT
  SK_ID_CURR,
  COUNT(*)                                      AS BUR_N_CREDITS,
  SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS BUR_N_ACTIVE,
  AVG(AMT_CREDIT_SUM)                           AS BUR_AMT_CREDIT_SUM_MEAN,
  SUM(AMT_CREDIT_SUM)                           AS BUR_AMT_CREDIT_SUM_SUM,
  SUM(AMT_CREDIT_SUM_DEBT)                      AS BUR_DEBT_SUM,
  SUM(AMT_CREDIT_SUM_OVERDUE)                   AS BUR_OVERDUE_SUM,
  MAX(CREDIT_DAY_OVERDUE)                       AS BUR_OVERDUE_DAYS_MAX,
  AVG(CREDIT_DAY_OVERDUE)                       AS BUR_OVERDUE_DAYS_MEAN,
  AVG(DAYS_CREDIT)                              AS BUR_DAYS_CREDIT_MEAN,
  MAX(DAYS_CREDIT_UPDATE)                       AS BUR_DAYS_CREDIT_UPDATE_MAX
FROM hist
GROUP BY SK_ID_CURR;