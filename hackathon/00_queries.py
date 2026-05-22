# =============================================================================
# 09_queries.py — Day-of queries, in execution order
# Each block = one Databricks notebook cell.
# Pattern: spark.sql("""...""") for exploration, .toPandas() before modeling.
# Replace TABLE_NAME with the real table at 10am.
# =============================================================================

TABLE_NAME = "`hackathon`.`data`.`pophealth_pdf_train_patientlevel`"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: EDA & PREP
# ─────────────────────────────────────────────────────────────────────────────

# 1. Schema inspection
display(spark.sql(f"DESCRIBE TABLE {TABLE_NAME}"))

# ─────────────────────────────────────────────────────────────────────────────
# EDA SUMMARY — run this single cell for a full snapshot before modeling
# ─────────────────────────────────────────────────────────────────────────────
def eda_summary(table):
    def q(sql): return spark.sql(sql).collect()
    def f(v):   return float(v) if v is not None else 0.0
    def s(v):   return str(v)   if v is not None else "Unknown"

    # --- cohort & target ---
    row = q(f"SELECT COUNT(*) AS n, AVG(CAST(Outcome AS DOUBLE)) AS rate, SUM(CAST(Outcome AS DOUBLE)) AS events FROM {table}")[0]
    print(f"{'='*55}")
    print(f"  COHORT & TARGET")
    print(f"  Total rows   : {row.n:,}")
    print(f"  Outcome rate : {100*f(row.rate):.2f}%  ({int(f(row.events)):,} events)")

    # --- subgroup sizes + event rates ---
    print(f"\n  SUBGROUP SIZES & EVENT RATES")
    subs = q(f"""
        SELECT subgroup, COUNT(*) AS n, AVG(CAST(Outcome AS DOUBLE)) AS rate FROM (
          SELECT Outcome, 'Glaucoma'          AS subgroup FROM {table} WHERE cms_ccw_glaucoma_36 = 1
          UNION ALL
          SELECT Outcome, 'Diabetic'                      FROM {table} WHERE cms_ccw_diabetes_36 = 1
          UNION ALL
          SELECT Outcome, 'Glaucoma+Diabetic'             FROM {table}
            WHERE cms_ccw_glaucoma_36=1 AND cms_ccw_diabetes_36=1
          UNION ALL
          SELECT Outcome, 'CKD'                           FROM {table} WHERE cms_ccw_chronic_kidney_disease_36=1
          UNION ALL
          SELECT Outcome, 'Heart Failure'                 FROM {table} WHERE cms_ccw_heart_failure_36=1
        ) GROUP BY subgroup ORDER BY rate DESC
    """)
    for r in subs:
        print(f"    {s(r.subgroup):<22} n={r.n:>7,}  event_rate={100*f(r.rate):.2f}%")

    # --- deprivation vs outcome ---
    print(f"\n  ADI QUINTILE vs OUTCOME RATE")
    adis = q(f"""
        SELECT ADINatRankQuintile AS quintile, COUNT(*) AS n, AVG(CAST(Outcome AS DOUBLE)) AS rate
        FROM {table} GROUP BY quintile ORDER BY quintile
    """)
    for r in adis:
        print(f"    Quintile {s(r.quintile)}  n={r.n:>7,}  event_rate={100*f(r.rate):.2f}%")

    # --- race/ethnicity ---
    print(f"\n  RACE/ETHNICITY vs OUTCOME RATE")
    races = q(f"""
        SELECT DerivedRaceEthnicity AS grp, COUNT(*) AS n, AVG(CAST(Outcome AS DOUBLE)) AS rate
        FROM {table} GROUP BY grp ORDER BY rate DESC
    """)
    for r in races:
        print(f"    {s(r.grp):<35} n={r.n:>7,}  event_rate={100*f(r.rate):.2f}%")

    # --- sex ---
    print(f"\n  SEX vs OUTCOME RATE")
    sexes = q(f"SELECT sex, COUNT(*) AS n, AVG(CAST(Outcome AS DOUBLE)) AS rate FROM {table} GROUP BY sex ORDER BY rate DESC")
    for r in sexes:
        print(f"    {s(r.sex):<10} n={r.n:>7,}  event_rate={100*f(r.rate):.2f}%")

    # --- missing values ---
    print(f"\n  MISSING VALUES (key columns)")
    miss = q(f"""
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN ADINatrank IS NULL THEN 1 END)                   AS null_adi,
          SUM(CASE WHEN CAHPIPercentile IS NULL THEN 1 END)              AS null_cahpi,
          SUM(CASE WHEN straight_line_distance_miles IS NULL THEN 1 END) AS null_dist,
          SUM(CASE WHEN DerivedRaceEthnicity IS NULL THEN 1 END)         AS null_race,
          SUM(CASE WHEN Outcome IS NULL THEN 1 END)                      AS null_target
        FROM {table}
    """)[0]
    n = miss.n or 1
    for col in ["null_adi", "null_cahpi", "null_dist", "null_race", "null_target"]:
        val = getattr(miss, col) or 0
        print(f"    {col:<20} {val:>6,}  ({100*val/n:.1f}%)")

    # --- date range ---
    print(f"\n  DATE RANGE")
    dates = q(f"SELECT MIN(EncMonth) AS mn, MAX(EncMonth) AS mx, COUNT(DISTINCT EncMonth) AS n_months FROM {table}")[0]
    print(f"    {s(dates.mn)} → {s(dates.mx)}  ({dates.n_months} distinct months)")
    print(f"{'='*55}")

eda_summary(TABLE_NAME)

# ─────────────────────────────────────────────────────────────────────────────

# PHASE 2: FEATURE ENGINEERING

# ─────────────────────────────────────────────────────────────────────────────

# 12. Build feature view — cohort: glaucoma OR diabetic patients only
# Focused column set; Python one-hot encodes the 3 string cols after .toPandas()
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW features AS
SELECT
  patientID,
  EncMonth,
  Outcome,

  -- ── Primary cohort flags ──────────────────────────────────────────────────
  cms_ccw_glaucoma_36                                    AS flag_glaucoma,
  cms_ccw_diabetes_36                                    AS flag_diabetes,
  CASE WHEN cms_ccw_glaucoma_36 = 1
            AND cms_ccw_diabetes_36 = 1 THEN 1 ELSE 0 END AS flag_glaucoma_diabetic,

  -- ── Eye-related comorbidities ─────────────────────────────────────────────
  cms_ccw_sensory_blindness_and_visual_impairment_36     AS flag_blindness,
  cms_ccw_cataract_36                                    AS flag_cataract,

  -- ── Key comorbidities (glaucoma / diabetes risk factors) ─────────────────
  cms_ccw_hypertension_36                                AS flag_hypertension,
  cms_ccw_obesity_36                                     AS flag_obesity,
  cms_ccw_chronic_kidney_disease_36                      AS flag_ckd,
  cms_ccw_heart_failure_36                               AS flag_hf,
  cms_ccw_ischemic_heart_disease_36                      AS flag_ihd,
  cms_ccw_atrial_fibrillation_36                         AS flag_afib,
  cms_ccw_depression_36                                  AS flag_depression,
  cms_ccw_anxiety_disorders_36                           AS flag_anxiety,
  cms_ccw_tobacco_use_36                                 AS flag_tobacco,
  GREATEST(cms_ccw_lung_cancer_36, cms_ccw_colorectal_cancer_36,
           cms_ccw_female_male_breast_cancer_36)         AS flag_cancer,

  -- ── Comorbidity burden ────────────────────────────────────────────────────
  last_12months_diagnosis_count_ccw,
  last_12months_chronic_count,

  -- ── Deprivation / SES ─────────────────────────────────────────────────────
  ADINatrank,
  ADIStaterank,
  CAHPIPercentile,
  straight_line_distance_miles,
  LOG(1 + straight_line_distance_miles)                  AS distance_log,
  CASE WHEN ADINatrank >= 80 THEN 1 ELSE 0 END           AS high_deprivation,
  CASE WHEN CAHPIPercentile <= 25 THEN 1 ELSE 0 END      AS low_cahpi,
  CASE WHEN straight_line_distance_miles > 10
       THEN 1 ELSE 0 END                                 AS far_from_care,
  ADINatRankQuintile,
  CAHPIQuartile,

  -- ── Office visit utilization ──────────────────────────────────────────────
  last_12months_officevisit_count,
  last_6months_officevisit_count,
  last_12months_officevisit_distinct_specialty,
  last_6months_officevisit_distinct_specialty,
  last_6months_officevisit_count
    / (last_12months_officevisit_count + 0.001)          AS op_recency_ratio,
  CASE WHEN last_12months_officevisit_distinct_specialty >= 3
       THEN 1 ELSE 0 END                                 AS high_specialty_breadth,

  -- ── Specialty visits ──────────────────────────────────────────────────────
  Specialty_internal_medicine,
  Specialty_family_practice,
  Specialty_nephrology,
  Specialty_cardiology,
  Specialty_cardiology
    / (last_12months_officevisit_count + 0.001)          AS cardio_visit_ratio,

  -- ── Demographics (one-hot encoded in Python) ──────────────────────────────
  DerivedRaceEthnicity,
  MaritalStatus,
  sex

FROM {TABLE_NAME}
WHERE Outcome IS NOT NULL
  AND (cms_ccw_glaucoma_36 = 1 OR cms_ccw_diabetes_36 = 1)  -- cohort filter
""")
print("features view created")

# ─────────────────────────────────────────────────────────────────────────────

# 13. Confirm temporal split boundaries
display(spark.sql("""
SELECT
  MIN(EncMonth) AS earliest,
  MAX(EncMonth) AS latest,
  COUNT(DISTINCT DATE_TRUNC('month', EncMonth)) AS n_months
FROM features
"""))

# ─────────────────────────────────────────────────────────────────────────────

# 14. Feature–target correlation
display(spark.sql("""
SELECT
  CORR(ADINatrank,                      Outcome) AS r_adi,
  CORR(CAHPIPercentile,                 Outcome) AS r_cahpi,
  CORR(last_12months_officevisit_count, Outcome) AS r_op12,
  CORR(flag_glaucoma,                   Outcome) AS r_glaucoma,
  CORR(flag_diabetes,                   Outcome) AS r_diabetes,
  CORR(flag_glaucoma_diabetic,          Outcome) AS r_glaucoma_diabetic,
  CORR(Specialty_cardiology,            Outcome) AS r_cardio,
  CORR(straight_line_distance_miles,    Outcome) AS r_distance,
  CORR(last_12months_chronic_count,     Outcome) AS r_chronic_count
FROM features
"""))

# ─────────────────────────────────────────────────────────────────────────────

# 15. Pull feature table into pandas for modeling
df = spark.sql("SELECT * FROM features").toPandas()
print(f"Shape: {df.shape}, positive rate: {df['Outcome'].mean():.3f}")
# → hand df to 01_features.py build_features() from here


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: VALIDATION SUMMARY
# Run after 10_model_building.py cell 14 registers the scores view.
# ─────────────────────────────────────────────────────────────────────────────

def validation_summary():
    def q(sql): return spark.sql(sql).collect()
    def f(v):   return float(v) if v is not None else 0.0
    def i(v):   return int(v)   if v is not None else 0

    SEP = "=" * 60
    sep = "-" * 60

    print(SEP)
    print("  PHASE 3 — VALIDATION SUMMARY")
    print(SEP)

    # ── 16. Overall confusion matrix ─────────────────────────────────────────
    cm = q("""
        SELECT
          SUM(CASE WHEN y_true=1 AND y_pred=1 THEN 1 END) AS TP,
          SUM(CASE WHEN y_true=0 AND y_pred=1 THEN 1 END) AS FP,
          SUM(CASE WHEN y_true=1 AND y_pred=0 THEN 1 END) AS FN,
          SUM(CASE WHEN y_true=0 AND y_pred=0 THEN 1 END) AS TN
        FROM scores
    """)[0]
    tp, fp, fn, tn = i(cm.TP), i(cm.FP), i(cm.FN), i(cm.TN)
    n_total = tp + fp + fn + tn
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)
    npv  = tn / max(tn + fn, 1)
    print(f"\n[16] OVERALL CONFUSION MATRIX  (n={n_total:,})")
    print(f"       Predicted+  Predicted-")
    print(f"  Actual+   {tp:>6}      {fn:>6}   (sensitivity {sens:.3f})")
    print(f"  Actual-   {fp:>6}      {tn:>6}   (specificity {spec:.3f})")
    print(f"  PPV {ppv:.3f}  NPV {npv:.3f}")

    # ── 17. Subgroup confusion matrices ──────────────────────────────────────
    sgs = q("""
        SELECT subgroup, COUNT(*) AS n, AVG(y_true) AS prev,
          SUM(CASE WHEN y_true=1 AND y_pred=1 THEN 1 END) AS TP,
          SUM(CASE WHEN y_true=0 AND y_pred=1 THEN 1 END) AS FP,
          SUM(CASE WHEN y_true=1 AND y_pred=0 THEN 1 END) AS FN,
          SUM(CASE WHEN y_true=0 AND y_pred=0 THEN 1 END) AS TN
        FROM scores GROUP BY subgroup
    """)
    print(f"\n[17] SUBGROUP CONFUSION MATRICES")
    print(f"  {'Subgroup':<22} {'n':>5} {'Prev':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Sens':>6} {'PPV':>6}")
    print(f"  {sep}")
    for r in sgs:
        tp_s, fp_s, fn_s, tn_s = i(r.TP), i(r.FP), i(r.FN), i(r.TN)
        s_sens = tp_s / max(tp_s + fn_s, 1)
        s_ppv  = tp_s / max(tp_s + fp_s, 1)
        print(f"  {str(r.subgroup):<22} {i(r.n):>5,} {f(r.prev):>6.3f} {tp_s:>5} {fp_s:>5} {fn_s:>5} {tn_s:>5} {s_sens:>6.3f} {s_ppv:>6.3f}")

    # ── 18. Decile lift table ─────────────────────────────────────────────────
    deciles = q("""
        WITH d AS (SELECT *, NTILE(10) OVER (ORDER BY y_prob DESC) AS decile FROM scores)
        SELECT decile, COUNT(*) AS n, SUM(y_true) AS n_events, AVG(y_true) AS event_rate,
               AVG(y_true) / (SELECT AVG(y_true) FROM scores) AS lift
        FROM d GROUP BY decile ORDER BY decile
    """)
    print(f"\n[18] DECILE LIFT TABLE")
    print(f"  {'Decile':>7} {'n':>6} {'Events':>7} {'EventRate':>10} {'Lift':>6}")
    print(f"  {sep}")
    for r in deciles:
        print(f"  {i(r.decile):>7} {i(r.n):>6,} {i(r.n_events):>7} {f(r.event_rate):>10.4f} {f(r.lift):>6.2f}x")

    # ── 19. Top-decile capture ────────────────────────────────────────────────
    cap = q("""
        WITH d AS (SELECT *, NTILE(10) OVER (ORDER BY y_prob DESC) AS decile FROM scores)
        SELECT SUM(CASE WHEN decile=1 THEN y_true END) / SUM(y_true) AS capture FROM d
    """)[0]
    print(f"\n[19] TOP-DECILE CAPTURE RATE: {f(cap.capture):.3f}  ({f(cap.capture)*100:.1f}% of all events in top 10% of scores)")

    # ── 20. Equity — race/ethnicity ───────────────────────────────────────────
    races = q("""
        SELECT DerivedRaceEthnicity, COUNT(*) AS n,
               AVG(y_true) AS actual_rate, AVG(y_prob) AS mean_prob,
               AVG(y_true) - AVG(y_prob) AS cal_gap
        FROM scores GROUP BY DerivedRaceEthnicity ORDER BY actual_rate DESC
    """)
    print(f"\n[20] EQUITY — RACE/ETHNICITY")
    print(f"  {'Group':<32} {'n':>6} {'ActualRate':>11} {'MeanScore':>10} {'CalGap':>8}")
    print(f"  {sep}")
    for r in races:
        flag = " ⚠" if abs(f(r.cal_gap)) > 0.02 else ""
        print(f"  {str(r.DerivedRaceEthnicity):<32} {i(r.n):>6,} {f(r.actual_rate):>11.4f} {f(r.mean_prob):>10.4f} {f(r.cal_gap):>+8.4f}{flag}")

    # ── 21. Equity — ADI quintile ─────────────────────────────────────────────
    adis = q("""
        SELECT ADINatRankQuintile, COUNT(*) AS n,
               AVG(y_true) AS actual_rate, AVG(y_prob) AS mean_prob,
               AVG(y_true) - AVG(y_prob) AS cal_gap
        FROM scores GROUP BY ADINatRankQuintile ORDER BY ADINatRankQuintile
    """)
    print(f"\n[21] EQUITY — ADI DEPRIVATION QUINTILE  (5 = most deprived)")
    print(f"  {'Quintile':>9} {'n':>6} {'ActualRate':>11} {'MeanScore':>10} {'CalGap':>8}")
    print(f"  {sep}")
    for r in adis:
        flag = " ⚠" if abs(f(r.cal_gap)) > 0.02 else ""
        print(f"  {str(r.ADINatRankQuintile):>9} {i(r.n):>6,} {f(r.actual_rate):>11.4f} {f(r.mean_prob):>10.4f} {f(r.cal_gap):>+8.4f}{flag}")

    # ── 22. Calibration bins ──────────────────────────────────────────────────
    bins = q("""
        WITH b AS (SELECT *, NTILE(10) OVER (ORDER BY y_prob) AS bin FROM scores)
        SELECT bin, AVG(y_prob) AS mean_pred, AVG(y_true) AS mean_actual, COUNT(*) AS n
        FROM b GROUP BY bin ORDER BY bin
    """)
    print(f"\n[22] CALIBRATION  (predicted vs actual by score decile)")
    print(f"  {'Bin':>4} {'n':>6} {'MeanPredicted':>14} {'MeanActual':>11} {'Gap':>8}")
    print(f"  {sep}")
    for r in bins:
        gap = f(r.mean_actual) - f(r.mean_pred)
        flag = " ⚠" if abs(gap) > 0.05 else ""
        print(f"  {i(r.bin):>4} {i(r.n):>6,} {f(r.mean_pred):>14.4f} {f(r.mean_actual):>11.4f} {gap:>+8.4f}{flag}")

    print(f"\n{SEP}")
    print("  END OF VALIDATION SUMMARY")
    print(SEP)

validation_summary()
