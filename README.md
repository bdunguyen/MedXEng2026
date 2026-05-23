# Retina Risk — VisionWatch
**UC Davis Health Hackathon | May 22, 2026**
**Team: Leonardo Zavala-Jimenez & Britney Nguyen**

Predicting 12-month emergency department and inpatient admission risk for glaucoma and diabetic patients, visualized as an interactive 3D retina.

---

## What It Does

Retina Risk uses a machine learning model trained on UC Davis Health population data to identify which glaucoma and diabetic patients are most likely to have an acute care event in the next 12 months. Results are surfaced through **VisionWatch** — a 3D visualization where each node represents a patient cohort, positioned by disease burden and colored by predicted risk.

**Holdout performance:** AUROC 0.718 on 3,572 held-out patients (20.9% event rate).

---

## Repo Structure

```
MedXEng2026/
├── hackathon/
│   ├── 00_queries.py          # EDA and SQL queries (Databricks)
│   ├── 01_model_building.py   # Full ML pipeline (Databricks)
│   └── REPORT.md              # Full project report
│
└── retina-risk/               # VisionWatch frontend
    ├── src/
    │   ├── components/
    │   │   ├── RetinaScene.jsx    # Three.js 3D scene
    │   │   ├── SidePanel.jsx      # Cohort details panel
    │   │   ├── TrendChart.jsx     # Risk trend overlay
    │   │   ├── ReportOverlay.jsx  # In-app report viewer
    │   │   ├── GlossaryOverlay.jsx
    │   │   └── Node.jsx           # 3D node geometry and animation
    │   └── data/
    │       ├── actual_predictions.json   # 7 condition cohorts (real data)
    │       ├── cluster_predictions.json  # 4 KMeans patient archetypes
    │       └── demo_predictions.json
    ├── public/
    │   └── report.md          # Static report served in-app
    ├── server/
    │   └── geminiInsights.mjs # Cohort insight generation
    └── api/
        └── insights.js        # Vercel serverless function
```

---

## ML Pipeline (Databricks)

**Data:** `hackathon.data.pophealth_pdf_train_patientlevel` — 44,250 patient records, Jan 2021–Feb 2027. Cohort filter: glaucoma or diabetic diagnosis within 3 years.

**Models trained:** Logistic Regression, Random Forest, LightGBM, Stacking Ensemble

**Best model:** Calibrated Random Forest (isotonic regression)

**Evaluation:**
- Internal validation (temporal split): AUROC 0.757
- True holdout (`pophealth_pdf_holdout_patientlevel`): AUROC 0.718, n=3,572

**Key findings:**
- Multi-condition patients (CKD + HF + Cancer) have the highest risk (5.9% validation event rate) and are the top priority for care coordination
- Cancer-only cohort has the best model accuracy (AUROC 0.849) but too few patients (n=34) for a dedicated model
- Clustering reveals two equally high-risk groups needing different interventions: high-acuity complex patients (15 visits/yr) vs high-deprivation low-access patients (2.2 visits/yr)
- Patients in the most deprived neighborhoods (ADI Q5) have 1.6× the event rate of the least deprived

---

## Frontend — VisionWatch

Built with **React 19**, **Vite**, and **Three.js**.

### Run locally

```bash
cd retina-risk
npm install
cp .env.example .env   # add GEMINI_API_KEY if available
npm run dev
```

### Dataset toggle

| View | Description |
|---|---|
| Actual | 7 condition cohorts with real holdout metrics |
| Clusters | 4 KMeans patient archetypes |
| Demo | Synthetic demo data |

### Deployment

Deployed via Vercel from the `Lumiho/retinarisk` repo (git subtree of `retina-risk/`).

To push updates:
```bash
git push deploy $(git subtree split --prefix retina-risk HEAD):main --force
```

---

## Report

The full project report (methodology, subgroup analysis, equity audit, model comparison) is available:
- In the app: click **Report** in the side panel
- In this repo: [`hackathon/REPORT.md`](hackathon/REPORT.md)
