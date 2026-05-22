# VisionWatch — 4-Minute Demo Talking Points
### UC Davis Health Hackathon | May 22, 2026

---

## 0:00–0:30 — Hook

"We talked to an eye surgeon who said she has no systematic way to know which of her patients will end up in the ED between visits. She's making that call on intuition.

We built a tool to change that."

---

## 0:30–1:30 — Live Demo (have site open)

**Click any cohort node.**

> "Each node is a patient cohort — glaucoma, diabetic, heart failure, kidney disease, cancer. Position reflects disease burden. Color reflects predicted risk. The closer to center, the higher the burden."

**Point to the side panel.**

> "This cohort has a 39% predicted 12-month risk of an ED visit or inpatient admission. The model's accuracy on this group is 0.743 AUROC — well above random. Below that you see the equity context — what we know about disparities specific to this group."

**Click the trend chart.**

> "The trend line shows how risk evolves across the observation window. The top features tell you what's driving it — diagnosis count, distance to care."

---

## 1:30–2:00 — The Number That Matters

> "We tested on a held-out dataset we had never seen. 3,572 patients, 20.9% event rate.

> AUROC 0.718. That means if you randomly pick one patient who had an acute event and one who didn't, the model correctly identifies the higher-risk one 72% of the time.

> More practically: by contacting just the top-flagged 14% of patients, you capture 30% of all acute events. That's the efficiency gain for a care coordination team."

---

## 2:00–2:45 — The Cluster Finding (strongest insight)

**Switch to Clusters toggle.**

> "Beyond condition groups, we ran unsupervised clustering and found four patient archetypes."

**Point to Clusters 2 and 3 — the two closest nodes.**

> "These two groups have nearly identical risk. 32% and 30% hospitalization rate. But look at why.

> This one — High-Acuity Complex — has 15 clinic visits a year and 27 chronic conditions on average. They're already in the system. The intervention is better care coordination.

> This one — High-Deprivation, Low-Access — has just 2.2 visits a year despite similar disease burden. They're nearly invisible to the health system. Their risk comes from access barriers, not lack of disease.

> Same risk level. Completely different intervention. Without clustering you'd treat them the same."

---

## 2:45–3:30 — Fairness & Honesty

> "We ran a full equity audit. Patients in the most deprived neighborhoods have a 21% event rate versus 13.5% in the least deprived — a 1.6× gap that exists before the model touches anything. The model doesn't create that disparity, but we documented it and built it into the interface.

> We also documented where the model is weakest. Glaucoma-only patients: AUROC 0.676. Only 327 patients in the holdout. We flag that rather than hiding it."

---

## 3:30–4:00 — Close

**Click Report or Glossary button.**

> "The full analysis, subgroup breakdowns, calibration tables, and a plain-language glossary are all in the app — not a separate deck.

> The model is built. The visualization is live. The next step is recalibrating on complete outcome windows and wiring this into the care coordinator workflow."

---

## Key Numbers to Have Ready

| Metric | Value |
|---|---|
| Holdout AUROC | 0.718 |
| Holdout n | 3,572 |
| Holdout event rate | 20.9% |
| Top 14% flagged → captures | 30% of events |
| ADI Q5 vs Q1 event rate | 21% vs 13.5% (1.6×) |
| Cluster 2 risk / visits/yr | 32% / 14.8 |
| Cluster 3 risk / visits/yr | 30% / 2.2 |
| Multi-condition AUROC | 0.743 |
| Cancer AUROC | 0.849 |

---

## If Asked About the Model

"We compared logistic regression, random forest, LightGBM, and a stacking ensemble. Calibrated random forest won at AUROC 0.757 on internal validation and 0.718 on the holdout. We use isotonic calibration to make the probability scores meaningful, not just the rankings."

## If Asked About Fairness

"We audited by race/ethnicity, neighborhood deprivation, sex, and area health index. The disparities in the data are real — Black patients have a 26% event rate versus 16% for white patients. The model's job is to detect risk accurately across all groups, and we monitor that separately from overall AUROC."

## If Asked About Deployment

"The recommended threshold is 0.589 — flags 14% of patients, catches 30% of events, about 11 contacts per event found. Whether that's operationally acceptable depends on the cost of outreach. A phone call is different from an intensive care management program."
