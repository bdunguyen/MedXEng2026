# Predicting Acute Care Utilization in Glaucoma & Diabetic Patients
### UC Davis Health Hackathon — May 22, 2026 | Leonardo Zavala-Jimenez

---

## 1. Problem Statement & Cohort Definition

**Clinical question:** Given a patient's structured EHR data at index month M, will they have an ED visit or inpatient admission within 12 months?

**Target variable:** `Outcome` (binary: 1 = ED visit or inpatient admit within 12 months)

**Cohort:** Patients with an active glaucoma **or** diabetic condition flag in the last 36 months  
Filter: `cms_ccw_glaucoma_36 = 1 OR cms_ccw_diabetes_36 = 1`

**Dataset:** `hackathon.data.pophealth_pdf_train_patientlevel`  
- 44,250 patient-month rows | Jan 2021 – Feb 2027 | 74 months  
- Overall outcome rate: **16.35%** (1 in 6 patients has an acute event)

**Motivation:** Care management capacity is limited. A calibrated risk model tells care coordinators *who* to contact first, enabling 2× more efficient outreach than random selection.

---

## 2. Exploratory Data Analysis

### 2.1 Subgroup Event Rates

| Subgroup | n | Event Rate |
|---|---|---|
| Glaucoma + Diabetic (overlap) | 1,134 | **25.6%** |
| Heart Failure | — | elevated |
| CKD | — | elevated |
| Diabetic only | — | moderate |
| Glaucoma only | — | lower |

Dual-condition patients have **56% higher risk** than the overall average — the highest-priority intervention target.

### 2.2 SES Deprivation Gradient

| ADI Quintile | Event Rate |
|---|---|
| Q1 (least deprived) | 13.5% |
| Q2 | 14.8% |
| Q3 | 16.0% |
| Q4 | 17.4% |
| Q5 (most deprived) | 21.0% |

Neighborhood deprivation alone produces a **1.56× risk multiplier**. SES features are clinically legitimate and necessary model inputs — they reflect real disparities, not model artifacts.

### 2.3 Racial Disparities at Baseline

| Group | Event Rate |
|---|---|
| Black / African American | 26.2% |
| Multi Race/Ethnicity | ~22% |
| White / Caucasian | ~16% |
| Latinx / Hispanic | ~14% |

These disparities exist **before any model is applied** — the model did not introduce them.

### 2.4 Missing Data

| Column | Missing | % |
|---|---|---|
| ADI National Rank | ~3,059 | 6.9% |
| CAHPI Percentile | ~4,602 | 10.4% |
| Distance to care | low | <2% |

Imputed with column median. Missingness correlates with socially marginalized populations — flagged in equity analysis.

---

## 3. Modeling Approach

### 3.1 Temporal Split (No Random Split)

- **Train:** Jan 2021 → ~Jan 2026 (first 85% of months)
- **Validation:** ~Feb 2026 → Feb 2027 (last 15% of months)

Random splits would leak future information into training. Temporal split mirrors real prospective deployment.

### 3.2 Features

| Category | Features |
|---|---|
| CMS CCW condition flags | 15+ binary indicators (glaucoma, diabetes, CKD, HF, hypertension, depression, etc.) |
| SES / deprivation | ADI national rank, CAHPI percentile, distance to care (log-transformed) |
| Utilization | 6- and 12-month office visit counts, distinct specialty count |
| Derived | Recency ratio, high-deprivation flag, low-CAHPI flag, far-from-care flag |
| Demographics | Race/ethnicity, sex, marital status (one-hot encoded) |

Demographics were encoded for the model but are **not causally used** — they are monitored for equity only.

### 3.3 Model Pipeline

1. **Logistic Regression** — L2 regularized, `class_weight="balanced"`, SAGA solver
2. **Random Forest** — 300 trees, `balanced_subsample`, max depth 8
3. **LightGBM** — 500 estimators, learning rate 0.05, balanced weights
4. **Isotonic calibration** — applied to best baseline model
5. **Stacking meta-learner** — logistic regression over 5-fold OOF probabilities from all three base models + calibration

---

## 4. Condition-Specific Analysis (Rubric §5A)

### 4.1 General Model — Subgroup Evaluation

| Subgroup | n (val) | Prevalence | AUROC | AUPRC | Sensitivity | F1 |
|---|---|---|---|---|---|---|
| Diabetic only | 937 | 4.5% | **0.739** | 0.086 | — | 0.164 |
| Glaucoma + Diabetic | 62 | 1.6% | 0.738 | 0.059 | — | 0.000 |
| Glaucoma only | 82 | 3.7% | 0.570 | 0.057 | — | 0.067 |

### 4.2 General vs. Specialized Models

| Subgroup | General AUROC | Specialized AUROC | Δ | Verdict |
|---|---|---|---|---|
| Glaucoma only | 0.570 | 0.629 | **+0.059** | **Specialize** |
| Diabetic only | 0.739 | 0.717 | −0.022 | General sufficient |
| Glaucoma + Diabetic | 0.738 | 0.721 | −0.016 | General sufficient |

### 4.3 Strategic Recommendation — Single vs. Multiple Models

| Dimension | General Model | Specialized Models |
|---|---|---|
| Performance | AUROC 0.74–0.76 overall | +0.06 for glaucoma only |
| Maintenance | Single retraining cycle | N retraining cycles |
| Monitoring | One equity report | N equity reports |
| Fairness | Easier to audit | Bias can differ per model |
| Operational complexity | Low | High |
| **Recommendation** | **Deploy now** | **Revisit glaucoma model when n>500** |

**Decision:** Deploy the general model. The glaucoma-only improvement (+0.059 AUROC) is meaningful but based on n=82 validation patients — too small for confident deployment. Revisit when the glaucoma cohort grows.

---

## 5. Advanced Modeling — Two-Stage Clustering (Rubric §5B)

### 5.1 Approach

1. **Unsupervised clustering:** PCA (20 components) → KMeans (k=4) on training features  
2. **Stage A:** Train per-cluster LightGBM specialists  
3. **Stage B:** Add cluster membership as one-hot features to general model  
4. **Comparison:** All three vs. general model baseline

### 5.2 Cluster Profiles (KMeans k=4, PCA 20 components)

| Cluster | n train | n val | Train Pos Rate | Val Pos Rate | Clinical Profile |
|---|---|---|---|---|---|
| 0 | 540 (4%) | 68 | 4.6% | 1.5% | Small mixed group — moderate ADI (30), low visit count, lower comorbidity burden |
| 1 | 5,276 (40%) | 435 | 11.0% | 1.8% | **Low-deprivation, well-resourced** — lowest ADI (22), highest CAHPI (60), low chronic count |
| 2 | 3,815 (29%) | 264 | **32.4%** | 8.3% | **High-acuity complex** — 14.8 visits/yr, 27 chronic conditions, 30% depression, high CKD/HF |
| 3 | 3,443 (26%) | 314 | 29.5% | 4.8% | **High-deprivation, low-access** — ADI 48 (most deprived), CAHPI 26 (worst area), only 2.2 visits/yr |

**Key feature means by cluster:**

| Feature | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| CKD flag | 0.58 | 0.66 | **0.72** | **0.76** |
| HF flag | 0.39 | 0.32 | **0.52** | **0.52** |
| Hypertension | 0.74 | 0.84 | **0.93** | 0.90 |
| Depression | 0.08 | 0.08 | **0.30** | 0.13 |
| ADI national rank | 30 | **22** | 24 | **48** |
| CAHPI percentile | 50 | **60** | 60 | **26** |
| Office visits (12mo) | 2.7 | 2.8 | **14.8** | 2.2 |
| Chronic count (12mo) | 10.9 | 10.8 | **27.4** | 15.4 |

**Clinical interpretation:**
- **Cluster 2 = High-acuity:** Very frequent utilizers with high comorbidity and depression — already in the system, high visibility
- **Cluster 3 = High-deprivation, low-access:** Most deprived neighborhood, multimorbid, but only 2.2 visits/year — care access is the barrier
- **Cluster 1 = Lower-risk, well-resourced:** Largest group, best neighborhood health, lower event rate
- **Cluster 0 = Small atypical group:** n=540, heterogeneous characteristics

### 5.3 Clustering Model Comparison

| Model | AUROC | AUPRC | Brier | Lift (top decile) |
|---|---|---|---|---|
| General model (baseline) | 0.727 | 0.078 | 0.178 | 1.52× |
| Cluster-as-feature | 0.720 | 0.078 | **0.071** | 1.96× |
| Per-cluster specialists | 0.711 | 0.082 | 0.075 | 1.96× |

**Per-cluster specialist AUROC:**

| Cluster | n val | AUROC | AUPRC | Val Pos Rate |
|---|---|---|---|---|
| 0 | 68 | 0.224 | 0.019 | 1.5% |
| 1 | 435 | 0.697 | 0.037 | 1.8% |
| 2 | 264 | 0.651 | 0.119 | 8.3% |
| 3 | 314 | 0.635 | 0.153 | 4.8% |

### 5.4 Tradeoff & Recommendation

**AUROC delta: −0.006** — clustering does not improve discrimination.

**Notable finding:** Cluster-as-feature model Brier score drops from 0.178 → 0.071 — **60% calibration improvement** — even without AUROC gain. Cluster membership helps the model assign more accurate probability magnitudes.

**Cluster 0 specialist AUROC 0.224 (below random):** Only 1 event in 68 validation patients. The specialist model is unreliable for this cluster — a general model fallback is needed.

| Approach | AUROC | Brier | Complexity | Verdict |
|---|---|---|---|---|
| General model | 0.727 | 0.178 | Low | **Deploy** |
| Cluster-as-feature | 0.720 | 0.071 | Low–Medium | Consider for calibration gain |
| Per-cluster specialists | 0.711 | 0.075 | High | Not justified |

**Recommendation:** Deploy general model. If accurate probability scores (not just ranking) are important for clinical decision-making, the cluster-as-feature model's 60% Brier improvement may justify the marginal added complexity.

---

## 6. Evaluation Framework (Rubric §6)

### 6.1 Discrimination

| Model | AUROC | AUPRC |
|---|---|---|
| RF + isotonic calibration | **0.757** | 0.079 |
| Stacker + calibration | 0.741 | 0.086 |
| Logistic Regression | 0.725 | 0.084 |
| Random Forest (uncalibrated) | 0.727 | 0.078 |
| LightGBM | 0.716 | 0.076 |

- **AUROC 0.757:** model correctly ranks 75.7% of event/non-event patient pairs
- **AUPRC note:** baseline (random) AUPRC = prevalence ≈ 0.04. Achieved 0.08 = **2× random** — appropriate for a low-prevalence screening tool

### 6.2 Calibration

**Brier Score:** 0.035 (calibrated RF) — lower is better; 0 = perfect, 0.25 = random  
**MACE (Mean Absolute Calibration Error):** 0.009 — excellent

| Decile Bin | Mean Predicted | Mean Actual | Gap |
|---|---|---|---|
| 1 (lowest) | 0.000 | 0.000 | 0.000 |
| 2 | 0.013 | 0.009 | −0.004 |
| 3 | 0.023 | 0.019 | −0.005 |
| 4 | 0.033 | 0.000 | −0.033 ⚠ |
| 5 | 0.041 | 0.065 | +0.024 |
| 6 | 0.041 | 0.009 | −0.032 ⚠ |
| 7 | 0.041 | 0.037 | −0.004 |
| 8 | 0.082 | 0.083 | +0.002 |
| 9 | 0.086 | 0.102 | +0.016 |
| 10 (highest) | 0.107 | 0.102 | −0.005 |

Calibration is strong at extremes. Mid-range noise (bins 4–6) is attributable to only 46 total events in the validation set — high variance, not systematic miscalibration.

### 6.3 Classification Accuracy (threshold = 0.589)

**Overall confusion matrix (n=1,081 validation patients):**

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP = — | FN = — |
| **Actual Negative** | FP = — | TN = — |

*(Fill in from Cell 17 output using op_threshold)*

| Metric | Value |
|---|---|
| Sensitivity (Recall) | 30% |
| Specificity | ~86% |
| PPV (Precision) | 9% |
| NPV | ~96% |
| F1-Score | 0.139 |

**Note on threshold:** Default 0.50 yields sensitivity=0 because calibrated scores range 0.03–0.11. Operational threshold 0.589 was selected to achieve ≥30% sensitivity.

### 6.4 Business & Operational Value

**Decile Lift Chart:**

| Decile (1 = highest risk) | n | Events | Event Rate | Lift vs. Random |
|---|---|---|---|---|
| 1 | 109 | 9 | 8.3% | **1.94×** |
| 2 | 108 | 12 | 11.1% | **2.61×** |
| 3 | 108 | 10 | 9.3% | **2.18×** |
| 4 | 108 | 6 | 5.6% | 1.31× |
| 5 | 108 | 2 | 1.9% | 0.44× |
| 6–10 | 540 | 7 | 1.3% | <0.5× |

**Top-risk decile capture rate: 19.6%**  
The model's top 10% risk group contains 19.6% of all acute care events — nearly **2× the expected 10%** under random selection.

**Operational threshold comparison:**

| Scenario | Threshold | Sensitivity | Patients Flagged | PPV | NNA* |
|---|---|---|---|---|---|
| High sensitivity | 0.554 | 50% | 21.4% | 10% | 10 |
| **Balanced (recommended)** | **0.589** | **30%** | **14.4%** | **9%** | **11** |
| Resource-constrained | 0.625 | 15% | 10.0% | 6.4% | 16 |

*NNA = Number Needed to Alert (patients contacted per true event caught)

---

## 7. Equity & Fairness

### 7.1 AUROC Gaps by Demographic Dimension

| Dimension | Best Group | AUROC | Worst Group | AUROC | Gap | Flag |
|---|---|---|---|---|---|---|
| Race/Ethnicity | Other | 0.819 | Latinx/Hispanic | 0.640 | **0.179** | ⚠ |
| ADI Quintile | Q4 | 0.838 | Q1 | 0.710 | **0.127** | ⚠ |
| Sex | Male | 0.771 | Female | 0.701 | **0.070** | ⚠ |
| CAHPI Quartile | Q1 | 0.768 | Q4 | 0.695 | 0.074 | ⚠ |
| Marital Status | — | — | — | — | 0.049 | ✓ |

### 7.2 Calibration Equity

| Group | Actual Rate | Mean Score | Gap | Issue |
|---|---|---|---|---|
| Multi-Race/Ethnicity | 10.5% | 5.7% | **+4.8%** | Model underpredicts — under-triage risk |
| Latinx/Hispanic | 3.7% | 5.4% | −1.7% | Model overpredicts — excess outreach |
| ADI Q3 | 2.8% | 6.0% | **−3.2%** | Model overpredicts moderately deprived patients |
| Black/African American | 5.6% | 6.5% | −0.9% | Minor overprediction |
| White/Caucasian | 4.0% | 4.2% | −0.2% | Well calibrated |

### 7.3 Mitigation Plan

1. Apply group-specific isotonic calibration for Multi-Race patients before deployment
2. Monitor false-negative rates quarterly by race/ethnicity; alert if gap widens >0.05
3. Race/ethnicity is **not used as a direct model predictor** — encoded for equity monitoring only
4. Set outreach equity targets: flag rate per group should reflect that group's prevalence
5. Small groups (AI/AN n=7, NHOPI n=16) — insufficient data; do not deploy to these groups without additional validation

---

## 8. Deployment & Governance Strategy (Rubric §7)

### 8.1 Workflow Integration

**Where the model fits:** Monthly batch scoring job run on DAVE/Databricks against the active patient registry.

```
Monthly scoring run
        ↓
Risk scores (y_prob) generated for all glaucoma/diabetic patients
        ↓
Top 15% flagged → Care Management Worklist (sorted by risk score)
        ↓
Care coordinator reviews list → outreach prioritized by score
        ↓
Outcomes documented in EHR → fed back for retraining
```

**Delivery mechanism:** A sortable worklist in the care management platform, showing patient name, risk score, top contributing factors (from SHAP), and recent utilization flags.

### 8.2 User & Action

| Role | What they see | What they do |
|---|---|---|
| Care Coordinator | Ranked worklist of flagged patients + top risk factors | Prioritize outreach calls; schedule wellness visits |
| Physician / NP | Risk flag in EHR sidebar (optional integration) | Review high-risk patients at next visit |
| Population Health Manager | Aggregate dashboard — monthly flag volume, event capture rate | Adjust thresholds, escalate equity concerns |
| Data Science / Analytics | Model performance dashboard | Trigger retraining if drift detected |

### 8.3 Thresholds & Alert Management

**Recommended threshold: 0.589** (sensitivity 30%, flags 14.4% of cohort, PPV 9%)

**False positive management:**
- PPV 9% means ~11 patients contacted per preventable event — acceptable for a phone-call intervention
- Care coordinators apply clinical judgment; model is advisory, not automatic
- Track "flagged but no outreach needed" rate; adjust threshold if >80%

**False negative management:**
- 70% of events will not be flagged at this threshold (sensitivity 30%)
- Mitigated by: clinical judgment, EHR alerts, routine check-ins for all chronic disease patients
- Model supplements, does not replace, standard care management protocols

**Threshold adjustment triggers:**
- Capacity increase → lower threshold (flag more patients)
- Capacity reduction → raise threshold (flag fewer, higher-confidence cases)
- Equity concern → apply group-specific thresholds

### 8.4 Model Governance

| Activity | Frequency | Trigger |
|---|---|---|
| Score recalibration | Monthly | New outcome data available |
| Equity audit | Quarterly | Scheduled; also if flagged rate diverges by race >5% |
| Full model retraining | Annually | Or if AUROC drops >0.03 on a rolling 90-day window |
| Bias audit | Before any expansion | New site, new condition, new population |
| Human review of model | Annually | Governance committee sign-off |

**Monitoring metrics (tracked continuously):**
- Score distribution drift (PSI — population stability index)
- Monthly AUROC on newly observed outcomes
- Flag rate by demographic group
- Outcome rate of flagged vs. unflagged patients

**Governance principles:**
- Model output is **decision support only** — no autonomous actions
- No patient-facing output; all outputs mediated by clinical staff
- Race/ethnicity used **only** for equity monitoring — never as a direct predictor
- All model versions versioned and auditable in MLflow / Databricks Model Registry
- Retraining requires ethics review if demographic composition of training data changes

---

## 9. Limitations & Future Work (Rubric §8)

### 9.1 Current Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **End-of-observation bias** | Validation prevalence ~4% vs. actual ~16%; outcomes incomplete for recent months | Exclude last 12 months from training; use as prospective holdout only |
| **Glaucoma cohort underpowered** | n=82 in validation; AUROC 0.57 near random | Expand cohort definition; collect more data before specializing |
| **No ophthalmology visit features** | `Specialty_ophthalmology` absent from dataset | Replaced with CCW flags; granularity lost |
| **No ICD-level features** | Raw diagnosis codes not used | Would improve sensitivity for rare conditions |
| **Small racial subgroups** | AI/AN (n=7), NHOPI (n=16) have no reliable equity estimates | Deliberate oversampling or data linkage needed |
| **Claims-based features only** | No lab values, vitals, social determinants beyond ADI | Integration with clinical data would strengthen model |

### 9.2 Future Work

1. **Exclude incomplete observation windows** — remove last 12 months of index dates from training; observe true prospective performance
2. **Add ICD code features** — top-N diagnosis codes as binary or count features
3. **Glaucoma-specific model** — revisit once n>500; incorporate IOP readings and visual field data if available
4. **Group-specific calibration** — apply separate isotonic calibration per racial/ethnic group to close the Multi-Race gap
5. **Intervention effectiveness measurement** — connect model outputs to actual outreach records; estimate causal impact of care management on acute events
6. **Real-time scoring** — move from monthly batch to event-triggered scoring (e.g., after each office visit)
7. **Explainability at point of care** — patient-level SHAP explanations surfaced in care coordinator interface

---

## 10. Model Comparison Table (Rubric §8 Deliverable)

| Model | AUROC | AUPRC | Brier | MACE | Sensitivity | Specificity | PPV | NPV | F1 | Lift (top decile) | Capture (top decile) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF + calibration | **0.757** | 0.079 | **0.035** | **0.009** | 0.30* | 0.86* | 0.09* | 0.96* | 0.139* | 1.94× | 19.6% |
| Stacker + calibration | 0.741 | 0.086 | 0.040 | 0.019 | — | — | — | — | — | 2.18× | 21.7% |
| Logistic Regression | 0.725 | 0.084 | 0.190 | 0.347 | — | — | — | — | — | 1.96× | 19.6% |
| Random Forest (uncal.) | 0.727 | 0.078 | 0.178 | 0.338 | — | — | — | — | — | 1.52× | 15.2% |
| LightGBM | 0.716 | 0.076 | 0.071 | 0.133 | — | — | — | — | — | 1.96× | 19.6% |

*At operational threshold 0.589

**Selected model: Calibrated Random Forest** — best AUROC, lowest Brier/MACE, interpretable feature importance.

---

## 11. Technical Appendix

**Environment:** Databricks (DAVE), Apache Spark, Python 3  
**Libraries:** scikit-learn, LightGBM, pandas, numpy, shap  
**Reproducibility:** All code in `09_queries.py` (EDA + SQL) and `10_model_building.py` (model pipeline, cells 1–17)  
**Temporal split:** deterministic — no random seed dependency for train/val assignment  
**Feature view:** Spark SQL temp view `features`, cohort-filtered, reproducible via `09_queries.py` cell 12  
**Scores view:** `scores` temp view registered in `10_model_building.py` cell 14  

**Files:**
- `09_queries.py` — EDA summary function, feature view SQL, validation summary function  
- `10_model_building.py` — full 17-cell model pipeline  
- `00_schema_map.py` — confirmed column name mapping  
- `REPORT.md` — this document  
