# Retina Risk — Devpost Submission

## Inspiration
We spoke with a surgeon specializing in eye surgery who described the challenge of knowing which patients were most likely to end up in the ED between visits — and having no systematic way to prioritize outreach. That conversation pointed us toward using population health data to surface exactly that risk, with glaucoma and diabetic patients as our focus because of their high prevalence and the real consequences of delayed intervention.

## What it does
Retina Risk is a risk stratification tool for care coordinators. It uses a machine learning model trained on UC Davis Health data to predict which glaucoma and diabetic patients are most likely to have an ED visit or inpatient admission in the next 12 months. The predictions are surfaced through VisionWatch — a 3D retina visualization where each node represents a patient cohort (Glaucoma, Diabetic, Heart Failure, CKD, Cancer, Multi-condition), positioned and colored by disease burden and predicted risk. Clicking a cohort shows the risk score, model accuracy, top predictive features, and equity context for that group.

## How we built it
The ML pipeline runs entirely in Databricks. We compared four models — Logistic Regression, Random Forest, LightGBM, and a stacking ensemble — using a temporal train/test split to prevent data leakage. The best model (calibrated Random Forest, AUROC 0.757) was evaluated across seven patient subgroups and audited for fairness by race/ethnicity, neighborhood deprivation, and sex. We also ran KMeans clustering to identify four distinct patient archetypes — including a "high-deprivation, low-contact" group that needs fundamentally different outreach than high-utilization patients. The frontend is built with React 19, Vite, and Three.js.

## Challenges we ran into
When we first opened the dataset we were overwhelmed by its breadth — 44,000+ patients, hundreds of features, and no single obvious angle. Narrowing to eye-related disease gave us focus. Technically, the biggest challenge was the temporal nature of the data: outcome windows at the end of the dataset are incomplete (a patient enrolled in late 2026 hasn't had 12 months to have an event yet), which artificially deflated our validation prevalence from ~16% to ~4% and forced us to rethink how we set the decision threshold. Standard 50% thresholds flagged no one; we tuned it down to 0.589 to be clinically useful.

## Accomplishments that we're proud of
The model achieves 2× lift over random guessing in its top risk decile, and our calibration error is under 1 percentage point — meaning a "9% risk" score actually corresponds to about 1 in 11 patients having an event. We're proud that we didn't just ship a number: we ran a full equity audit (finding that patients in the most deprived neighborhoods face 1.6× the event rate of the least deprived), documented where the model is weakest (Glaucoma-only, AUROC 0.57), and built the visualization to communicate uncertainty honestly rather than hiding it.

## What we learned
End-of-observation bias is a real and underappreciated problem in healthcare ML — your validation set can look dramatically different from your training distribution not because the model is wrong, but because the outcome window hasn't closed yet. We also learned that subgroup analysis is where the most actionable insight lives: the Cancer cohort had our best AUROC (0.849) with only 34 patients, while CKD — the largest cohort — was the hardest to predict, because kidney disease risk is driven by full clinical context rather than the condition alone.

## What's next for Retina Risk
With the full validation set, we'd retrain on the complete temporal split and measure how much the end-of-observation bias was inflating our difficulty estimates. The multi-condition cohort (patients with 2+ of CKD, HF, Cancer) is the one group where a dedicated model meaningfully outperforms the general one — that's the next model to productionize. Longer term, integrating real-time EHR scores and adding a care coordinator workflow (not just a visualization) would be the path from demo to deployment.
