# Predicting Acute Care Utilization in Glaucoma & Diabetic Patients
### UC Davis Health Hackathon — May 22, 2026 | Leonardo Zavala-Jimenez

---

## Executive Summary

This project built a risk prediction model to identify which glaucoma and diabetic patients at UC Davis Health are most likely to end up in the emergency department or be hospitalized within the next 12 months. Using historical patient data from 2021 to 2027, we trained and compared multiple machine learning models, selected the best-performing one, and evaluated how fairly it works across different patient groups. The model correctly identifies higher-risk patients at roughly twice the rate of random guessing. Rather than replacing clinical judgment, it is designed as a decision-support tool — giving care coordinators a prioritized list of patients to contact, so limited outreach capacity is used where it matters most. Key concerns around fairness were identified and concrete mitigation steps are proposed before any deployment.

---

## 1. Problem Statement

**What are we trying to predict?**  
Given what we know about a patient today, will they have an emergency department (ED) visit or be admitted to the hospital within the next 12 months?

**Who are we predicting for?**  
Patients who have been diagnosed with glaucoma, diabetes, or both within the past 3 years.

**Why does this matter?**  
Care management teams can only reach out to so many patients. Without a model, they either contact patients randomly or rely on general intuition. A risk model lets them prioritize — spending outreach time on the patients most likely to need it, and potentially preventing costly, disruptive hospitalizations.

**Data used:**  
UC Davis Health population health dataset — 44,250 patient records spanning January 2021 to February 2027. About 1 in 6 patients (16.35%) in the full dataset had an acute care event within 12 months.

---

## 2. What the Data Showed Before Any Modeling

Before building any model, we explored the data to understand which patients are most at risk and why.

### 2.1 Patients with Multiple Conditions Are at Much Higher Risk

| Patient Group | Event Rate |
|---|---|
| Glaucoma + Diabetic (both conditions) | **25.6%** — 1 in 4 patients |
| Heart Failure patients | elevated |
| Chronic Kidney Disease patients | elevated |
| Diabetic only | moderate |
| Glaucoma only | lower |

Patients with both glaucoma and diabetes have a **56% higher chance** of an acute care event compared to the average. They are the highest-priority group for intervention.

### 2.2 Neighborhood Poverty Strongly Predicts Risk

The Area Deprivation Index (ADI) measures how economically disadvantaged a patient's neighborhood is. Higher ADI = more deprived.

| Deprivation Level | Event Rate |
|---|---|
| Least deprived (Q1) | 13.5% |
| Q2 | 14.8% |
| Q3 | 16.0% |
| Q4 | 17.4% |
| Most deprived (Q5) | 21.0% |

Patients in the most deprived neighborhoods are **1.6× more likely** to have an acute event than those in the least deprived. This is a real disparity in the population — not something the model creates.

### 2.3 Racial Disparities Exist in the Data

| Patient Group | Event Rate |
|---|---|
| Black / African American | 26.2% |
| Multi-racial | ~22% |
| White / Caucasian | ~16% |
| Latinx / Hispanic | ~14% |

These differences exist **before any model is applied** — they reflect real-world disparities in health outcomes. The model's job is to detect risk accurately across all groups, not amplify these gaps.

### 2.4 Missing Information

Some patient records were missing neighborhood deprivation scores (6.9%) and area health index scores (10.4%). Missing data often affects the most socially vulnerable patients — this is noted in the fairness analysis.

---

## 3. How We Built the Model

### 3.1 Training vs. Testing — Keeping It Honest

We trained the model on older data (January 2021 through early 2026) and tested it on more recent data (early 2026 through February 2027). This mimics how the model would actually be used — learning from the past and predicting the future. We did **not** randomly shuffle the data before splitting, because that would let the model "peek" at future information during training.

### 3.2 What Information the Model Uses

The model takes in structured data already collected in the patient's health record:

| Type of Information | Examples |
|---|---|
| Condition flags | Does the patient have glaucoma, diabetes, kidney disease, heart failure, depression? (yes/no for each) |
| Neighborhood factors | How deprived is their area? How far do they live from a clinic? |
| Healthcare use | How many office visits in the past 6 or 12 months? How many different specialists did they see? |
| Derived signals | Are visits declining? Does the patient have many different chronic conditions? |
| Demographics | Race/ethnicity, sex, marital status — used only to monitor fairness, not to make predictions |

### 3.3 Models We Compared

We trained four different types of predictive models and compared them:

1. **Logistic Regression** — a simple, transparent model that learns which factors push risk up or down
2. **Random Forest** — builds hundreds of decision trees and averages their predictions, more powerful than logistic regression
3. **LightGBM** — a fast, accurate model that learns from its own mistakes in rounds
4. **Stacking model** — a second-level model that combines the predictions from all three above

After selecting the best model, we applied a **probability adjustment step** (isotonic calibration) to make sure the risk scores it outputs are accurate — not just that it ranks patients correctly, but that a score of 10% actually means roughly a 1-in-10 chance of an event.

---

## 4. How Well the Model Works on Each Patient Group

We evaluated the model separately for each patient condition group to see where it performs well and where it struggles.

### 4.1 Performance by Condition Group

We evaluated the model on both the rubric-specified cohorts (CKD, HF, Cancer, multi-condition) and our primary focus cohorts (Glaucoma, Diabetic).

**Rubric cohorts — CKD, HF, Cancer, Multi-condition**

| Patient Group | # Patients (test) | % with Event | Model Accuracy* | Notes |
|---|---|---|---|---|
| Cancer only | 34 | 2.9% | **Excellent (0.849)** | Best-performing subgroup; small n |
| HF only | 140 | 4.3% | **Good (0.766)** | Reliable; 3.3× top-decile lift |
| Multi-condition (2+ conditions) | 338 | 5.9% | Good (0.718) | Highest prevalence; specialized model helps |
| CKD only | 423 | 2.8% | Moderate (0.706) | Largest group; general model sufficient |

**Primary focus cohorts — Glaucoma / Diabetic**

| Patient Group | # Patients (test) | % with Event | Model Accuracy* | Notes |
|---|---|---|---|---|
| Diabetic only | 937 | 4.5% | **Good (0.74)** | Main driver of model performance |
| Glaucoma + Diabetic | 62 | 1.6% | Good (0.74) | Small group — results may vary |
| Glaucoma only | 82 | 3.7% | Weak (0.57) | Near random — too few patients |

*Model accuracy here means how well it ranks high-risk patients above low-risk ones. A score of 0.50 = random guessing, 1.00 = perfect. Above 0.70 is considered useful for clinical screening.

### 4.2 Should We Build Separate Models per Group?

We trained a dedicated model for each condition group and compared it against the one general model.

**Rubric cohorts**

| Patient Group | General Model | Dedicated Model | Difference | Decision |
|---|---|---|---|---|
| Multi-condition | 0.718 | **0.743** | **+0.025 better** | **Specialize** — meaningful gain, n=338 |
| HF only | 0.766 | 0.761 | −0.005 | General sufficient |
| CKD only | 0.706 | 0.670 | −0.036 | General better |
| Cancer only | 0.849 | 0.758 | −0.091 | General better — dedicated model overfit (n=34 too small) |

**Primary focus cohorts**

| Patient Group | General Model | Dedicated Model | Difference | Decision |
|---|---|---|---|---|
| Glaucoma only | 0.57 | 0.63 | **+0.06 better** | Would help, but only 82 test patients — not enough to trust |
| Diabetic only | 0.74 | 0.72 | −0.02 worse | General model is fine |
| Glaucoma + Diabetic | 0.74 | 0.72 | −0.02 worse | General model is fine |

**Decision:** Deploy one general model for most subgroups. The one exception is **multi-condition patients** (those with 2 or more of CKD, HF, Cancer) — a dedicated model improves accuracy by +0.025 AUROC and the group is large enough (n=338) to trust the result. Cancer-only patients had the highest general model accuracy (0.849) but too few patients (n=34) to build a reliable dedicated model.

---

## 5. Patient Subtypes — Clustering Analysis

Beyond condition groups, we asked: are there naturally occurring "types" of patients in this population that the model should be aware of?

Using an unsupervised grouping technique (KMeans clustering), we discovered 4 distinct patient profiles:

| Group | Size | Hospitalization Risk | Profile |
|---|---|---|---|
| 0 — Mixed/atypical | 540 (4%) | Low | Small, heterogeneous group |
| 1 — Well-resourced | 5,276 (40%) | Lower | Least deprived neighborhoods, good healthcare access, moderate chronic conditions |
| 2 — High-acuity complex | 3,815 (29%) | **Highest (32%)** | Nearly 15 clinic visits per year, 27 chronic conditions on average, 30% have depression — already in the system, hard to miss |
| 3 — High-deprivation, low-access | 3,443 (26%) | **High (30%)** | Most deprived neighborhoods, many chronic conditions, but only 2.2 visits/year — at risk due to access barriers |

**The key insight from clustering:** Groups 2 and 3 have similar overall risk levels but for completely different reasons. Group 2 patients are heavy healthcare users who need better coordination. Group 3 patients are nearly invisible to the health system — they have high need but low contact. These two groups call for different interventions.

Adding these group labels as model features improved the accuracy of the probability estimates by 60%, though it did not improve the ranking performance. Ultimately, the general model is recommended for deployment because the complexity of managing group-specific models is not justified by the performance gain.

---

## 6. Model Performance — Full Evaluation

### 6.1 How Well Does It Rank Patients? (Discrimination)

| Model | Ranking Accuracy (AUROC)* | Precision Score (AUPRC)** |
|---|---|---|
| Random Forest + calibration | **0.757** | 0.079 |
| Stacking model + calibration | 0.741 | 0.086 |
| Logistic Regression | 0.725 | 0.084 |
| Random Forest (no calibration) | 0.727 | 0.078 |
| LightGBM | 0.716 | 0.076 |

*AUROC (Area Under the ROC Curve): measures how well the model separates high-risk from low-risk patients. 0.50 = coin flip, 1.00 = perfect. Our best model at 0.757 means that when you randomly pick one patient who had an event and one who didn't, the model correctly ranks the event patient as higher risk 75.7% of the time.

**AUPRC (Area Under the Precision-Recall Curve): relevant when events are rare. Our event rate is ~4%, so a random model scores 0.04. Our model scores 0.08 — twice as good as random.

### 6.2 Are the Risk Scores Trustworthy? (Calibration)

A model can rank patients correctly but still give misleading probability scores. We checked whether a score of "10% risk" actually corresponds to about 1 in 10 patients having an event.

- **Brier Score: 0.035** — measures how far predicted probabilities are from actual outcomes (0 = perfect, 0.25 = uninformative). Our score is excellent.
- **Mean Calibration Error: 0.009** — on average, the model's probability estimates are less than 1 percentage point off from actual rates. This is very good.

The calibration table below shows predicted vs. actual rates for 10 groups of patients (sorted from lowest to highest predicted risk):

| Risk Group | Model Predicted | Actual Outcome Rate | Gap |
|---|---|---|---|
| 1 — Lowest risk | 0.0% | 0.0% | 0.0% |
| 2 | 1.3% | 0.9% | small |
| 3 | 2.3% | 1.9% | small |
| 4 | 3.3% | 0.0% | ⚠ noise |
| 5 | 4.1% | 6.5% | slight under |
| 6 | 4.1% | 0.9% | ⚠ noise |
| 7 | 4.1% | 3.7% | small |
| 8 | 8.2% | 8.3% | nearly perfect |
| 9 | 8.6% | 10.2% | slight under |
| 10 — Highest risk | 10.7% | 10.2% | nearly perfect |

The model is most reliable at the extremes (lowest and highest risk groups). The noise in the middle groups is due to the small number of events in the test set (only 46 total).

### 6.3 What Happens When We Make a Decision? (Classification)

The model outputs a continuous risk score. To make a yes/no decision (flag this patient or not), we set a threshold. Because acute events are rare (~4% of patients), a standard 50% threshold flags nobody. We tuned the threshold to be clinically useful.

At our recommended threshold of **0.589**:
- **Sensitivity: 30%** — the model catches 30% of patients who will actually have an event
- **Specificity: 86%** — correctly identifies 86% of patients who will not have an event
- **Precision (PPV): 9%** — of those flagged, 9% will actually have an event
- **Negative Predictive Value: 96%** — of those not flagged, 96% truly will not have an event
- **F1-Score: 0.139** — combined measure of precision and recall

In plain terms: if you contact 11 flagged patients, you will reach 1 who would have had an avoidable acute event. Whether that's acceptable depends on the cost and nature of the outreach (a phone call is low-cost; an intensive care management program is not).

### 6.4 Operational Value — Does It Actually Help?

**Lift table:** If we rank all patients by risk score and divide them into 10 equal groups, how do the event rates compare to random?

| Risk Group (1 = highest risk) | # Patients | Events Found | Event Rate | vs. Random |
|---|---|---|---|---|
| Top 10% | 109 | 9 | 8.3% | **1.9× better** |
| Next 10% | 108 | 12 | 11.1% | **2.6× better** |
| Next 10% | 108 | 10 | 9.3% | **2.2× better** |
| Next 10% | 108 | 6 | 5.6% | 1.3× better |
| Bottom 60% | 648 | 9 | 1.4% | below average |

The model concentrates most of the risk in the top 30% of patients. Focusing outreach on that group is roughly 2× more efficient than random contact.

**Top 10% capture rate: 19.6%** — by contacting just the top-scored 10% of patients, you reach 19.6% of all patients who will have an acute event. Randomly contacting 10% of patients would capture only 10%.

**Outreach scenarios:**

| Approach | Threshold | Events Caught | Patients Contacted | Contacts per Event Found |
|---|---|---|---|---|
| Cast wide net | 0.554 | 50% of events | 21.4% of patients | ~10 contacts |
| **Recommended** | **0.589** | **30% of events** | **14.4% of patients** | **~11 contacts** |
| Resource-limited | 0.625 | 15% of events | 10.0% of patients | ~16 contacts |

---

## 7. Fairness Analysis

We checked whether the model performs consistently across demographic groups, or whether it is systematically better or worse for some patients than others.

### 7.1 Ranking Accuracy by Group

We measured how well the model ranks high-risk vs. low-risk patients within each demographic group. A large gap between groups means the model is less reliable for some.

| Dimension | Best-performing Group | Score | Lowest-performing Group | Score | Gap | Concern? |
|---|---|---|---|---|---|---|
| Race/Ethnicity | "Other" category | 0.819 | Latinx / Hispanic | 0.640 | **0.179** | ⚠ Yes |
| Neighborhood Deprivation | Moderate deprivation (Q4) | 0.838 | Least deprived (Q1) | 0.710 | **0.127** | ⚠ Yes |
| Sex | Male | 0.771 | Female | 0.701 | **0.070** | ⚠ Yes |
| Area Health Index | Lowest-health areas (Q1) | 0.768 | Healthiest areas (Q4) | 0.695 | 0.074 | ⚠ Yes |
| Marital Status | — | — | — | — | 0.049 | OK |

### 7.2 Are the Probability Scores Fair?

Beyond ranking, we checked whether the model's predicted risk matches actual outcomes equally across groups.

| Group | Actual Event Rate | Model's Predicted Risk | Difference | Problem |
|---|---|---|---|---|
| Multi-racial patients | 10.5% | 5.7% | **Model underestimates by 4.8%** | These patients would be under-prioritized — a missed care opportunity |
| Latinx / Hispanic | 3.7% | 5.4% | Model overestimates by 1.7% | These patients get flagged more than necessary |
| Moderate deprivation (Q3) | 2.8% | 6.0% | Model overestimates by 3.2% | Unnecessary outreach for this group |
| Black / African American | 5.6% | 6.5% | Overestimates slightly | Minor |
| White / Caucasian | 4.0% | 4.2% | Nearly exact | Good |

**Most critical finding:** Multi-racial patients are at 10.5% actual risk but the model only assigns them a 5.7% predicted score — meaning they would systematically receive less outreach than they need. This must be corrected before deployment.

### 7.3 What We Recommend to Address These Gaps

1. Apply a separate probability adjustment for multi-racial patients to correct the underestimation
2. Check false-negative rates (missed high-risk patients) by race/ethnicity every quarter
3. Race/ethnicity is used only to monitor fairness — it is **never used as a direct input to predict risk**
4. Set outreach targets so each demographic group receives outreach proportional to their actual risk level
5. Do not deploy the model for American Indian/Alaska Native or Native Hawaiian/Pacific Islander patients without additional validation — the test groups were too small (fewer than 20 patients each) to draw any reliable conclusions

---

## 8. Deployment Plan

### 8.1 How It Fits into the Workflow

The model would run automatically once a month, score every eligible patient, and produce a prioritized worklist for care coordinators.

```
Every month:
  → Model scores all glaucoma/diabetic patients
  → Top 15% flagged for review
  → Care coordinators receive ranked worklist
  → They make outreach calls, schedule visits
  → Outcomes recorded in EHR
  → Data fed back to retrain model
```

The worklist would show each patient's risk score, the top reasons driving their score (e.g., "missed 3 appointments," "newly diagnosed with kidney disease"), and recent utilization history.

### 8.2 Who Uses It and What They Do

| Person | What They See | What They Do |
|---|---|---|
| Care Coordinator | Ranked patient list with risk scores and top risk factors | Make outreach calls, schedule wellness visits |
| Physician / Nurse Practitioner | Optional risk flag in the patient's EHR chart | Review high-risk patients proactively |
| Population Health Manager | Monthly summary dashboard | Adjust thresholds if capacity changes; escalate fairness concerns |
| Analytics Team | Model performance tracking | Trigger retraining if accuracy drops |

### 8.3 Managing Flags — False Positives and False Negatives

**About false positives** (patients flagged who won't actually have an event):  
At our recommended threshold, about 9 out of 10 flagged patients will not have an acute event. This sounds like many wasted contacts — but for a low-cost intervention like a phone call, contacting 11 patients to prevent 1 hospitalization is often considered worthwhile. Care coordinators also apply clinical judgment and can deprioritize patients where outreach is clearly not needed.

**About false negatives** (high-risk patients the model misses):  
The model will miss 70% of patients who will have an event. This is expected — the model supplements, not replaces, standard care. Patients not flagged still receive routine care management as usual; the model only helps direct extra attention more efficiently.

**Adjusting the threshold based on capacity:**
- More capacity to make calls → lower the threshold → flag more patients
- Less capacity → raise the threshold → focus only on highest-confidence cases
- If fairness gaps are detected → apply group-specific thresholds

### 8.4 Keeping the Model Reliable Over Time

Models can become less accurate as patient populations and care patterns shift. We recommend:

| Task | How Often | Why |
|---|---|---|
| Recalibrate probability scores | Monthly | Patient population changes over time |
| Fairness audit | Quarterly | Detect if any group is being systematically missed |
| Full model retraining | Annually | Or sooner if accuracy drops noticeably |
| Independent bias review | Before any expansion to new sites or conditions | Ensure fairness in new contexts |
| Clinical governance sign-off | Annually | Ensure continued clinical appropriateness |

**Core principles:**
- The model is a tool for clinicians — it never makes decisions on its own
- No patient ever sees their own risk score directly
- Race/ethnicity informs fairness monitoring only — it does not drive the risk prediction
- Every model version is logged and can be audited

---

## 9. Limitations & Future Improvements

### 9.1 Known Limitations

| Limitation | What It Means | How to Address It |
|---|---|---|
| Recent data has incomplete outcomes | Patients from the last 12 months may not yet have had their full 12-month follow-up, making the test set appear lower-risk than it truly is | Exclude the most recent year from training; use it as a true future test set |
| Glaucoma-only group is too small | Only 82 glaucoma-only patients in the test set — results for this group are unreliable | Collect more data; revisit once group exceeds 500 patients |
| No eye specialist visit data | Eye clinic visit records were unavailable in the dataset | Would significantly strengthen predictions for the glaucoma group |
| No lab results or vital signs | The model only uses administrative data, not clinical measurements | Integrating lab values (e.g., HbA1c for diabetics) would improve accuracy |
| Some racial groups are too small to evaluate fairly | Fewer than 20 patients in several groups | Targeted data collection or linkage with external datasets |

### 9.2 What We Would Do Next

1. Fix the observation window problem by excluding the most recent 12 months from training
2. Add diagnosis code features — specific ICD codes as additional signals
3. Build a dedicated glaucoma model once enough patients are available, ideally incorporating eye pressure readings and visual field data
4. Fix the Multi-Race calibration gap by applying a group-specific probability adjustment
5. Measure whether outreach actually prevents events — connect model flags to care coordination records to estimate real-world impact
6. Explore real-time scoring triggered after each clinic visit, rather than monthly batch runs
7. Surface patient-level explanations (e.g., "this patient is flagged because they missed 4 visits and have new kidney disease") at the point of care

---

## 10. Model Comparison Table

| Model | Ranking Accuracy (AUROC) | Calibration Error (Brier) | Sensitivity* | Specificity* | Precision* | F1* | Top 10% Lift | Top 10% Capture |
|---|---|---|---|---|---|---|---|---|
| Random Forest + calibration | **0.757** | **0.035** | 30% | 86% | 9% | 0.139 | 1.94× | 19.6% |
| Stacking model + calibration | 0.741 | 0.040 | — | — | — | — | 2.18× | 21.7% |
| Logistic Regression | 0.725 | 0.190 | — | — | — | — | 1.96× | 19.6% |
| Random Forest (no calibration) | 0.727 | 0.178 | — | — | — | — | 1.52× | 15.2% |
| LightGBM | 0.716 | 0.071 | — | — | — | — | 1.96× | 19.6% |

*At recommended decision threshold 0.589

**Selected model: Calibrated Random Forest** — best overall ranking accuracy, most accurate probability estimates, and most interpretable for clinical use.

---

## 11. Technical Appendix

**Computing environment:** Databricks (DAVE platform), Apache Spark, Python 3.10  
**Libraries used:** scikit-learn, LightGBM, pandas, numpy, shap  
**Source data:** `hackathon.data.pophealth_pdf_train_patientlevel` — 44,250 rows, 417 columns, CMS Chronic Condition Warehouse condition flags  
**Train/test split:** by time — older months for training, most recent months for testing; fully deterministic, no randomness  
**Patient grouping (clustering):** dimension reduction (PCA, 20 components) followed by KMeans grouping (4 groups) on scaled training features  
**Frontend:** model predictions exported to `retina-risk/src/data/predictions.json` for the VisionWatch visualization

**Code files:**
- `00_queries.py` — data exploration, feature table creation, validation queries (all output as printed text for easy copying from Databricks)
- `01_model_building.py` — full 18-step model pipeline: data loading → feature engineering → train/test split → model training → calibration → subgroup evaluation → ensemble → fairness analysis → threshold tuning → clustering → export
- `REPORT.md` — this document
