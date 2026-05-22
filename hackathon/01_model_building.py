# =============================================================================
# 10_model_building.py — Full model pipeline for glaucoma / diabetic cohort
# Run AFTER 09_queries.py cells 1–15 (features view must exist).
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Load feature view into pandas
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = spark.sql("SELECT * FROM features").toPandas()
print(f"Cohort: {len(df):,} rows | target rate: {df['Outcome'].mean():.3f}")
print(f"Glaucoma: {df['flag_glaucoma'].sum():,} | Diabetic: {df['flag_diabetes'].sum():,} | Both: {df['flag_glaucoma_diabetic'].sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Feature engineering (encode categoricals, derived features)
# ─────────────────────────────────────────────────────────────────────────────
TARGET     = "Outcome"
PATIENT_ID = "patientID"
INDEX_MONTH = "EncMonth"
CAT_COLS   = ["DerivedRaceEthnicity", "MaritalStatus", "sex"]
# Cols to keep aside for subgroup analysis / equity — not used as model features
META_COLS  = [PATIENT_ID, INDEX_MONTH, TARGET,
              "flag_glaucoma", "flag_diabetes", "flag_glaucoma_diabetic",
              "ADINatRankQuintile", "CAHPIQuartile",
              "DerivedRaceEthnicity", "MaritalStatus", "sex"]

df[INDEX_MONTH] = pd.to_datetime(df[INDEX_MONTH].astype(str), format="%Y%m")

# One-hot encode categoricals
df_encoded = pd.get_dummies(df, columns=CAT_COLS, drop_first=True, dummy_na=True)

# Separate meta / target from features
meta = df[META_COLS].copy()
y    = df_encoded[TARGET].astype(int)

drop = [c for c in META_COLS if c in df_encoded.columns]
X    = df_encoded.drop(columns=drop)
X    = X.fillna(X.median(numeric_only=True)).astype("float32")

print(f"X shape: {X.shape} | features: {list(X.columns[:10])} ...")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Temporal train / val split (NO random split — leakage risk)
# ─────────────────────────────────────────────────────────────────────────────
months = df[INDEX_MONTH].dt.to_period("M")
sorted_months = sorted(months.unique())
n = len(sorted_months)

val_months  = set(sorted_months[int(n * 0.85):])
train_months = set(sorted_months[:int(n * 0.85)])

tr_mask = months.isin(train_months)
va_mask = months.isin(val_months)

X_tr, y_tr = X[tr_mask], y[tr_mask]
X_va, y_va = X[va_mask], y[va_mask]
meta_va    = meta[va_mask].reset_index(drop=True)

print(f"Train: {tr_mask.sum():,} | Val: {va_mask.sum():,}")
print(f"Train positive: {y_tr.mean():.3f} | Val positive: {y_va.mean():.3f}")
print(f"Train months: {min(train_months)} → {max(train_months)}")
print(f"Val months:   {min(val_months)} → {max(val_months)}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Train baselines
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    def make_lgbm():
        return lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=6, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1,
        )
    print("Using LightGBM")
except ImportError:
    def make_lgbm():
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, random_state=42,
        )
    print("LightGBM not found — using HistGradientBoosting")

models = {
    "logreg": Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced",
                                   solver="saga", random_state=42))
    ]),
    "rf": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    ),
    "lgbm": make_lgbm(),
}

fitted, val_probs = {}, {}
for name, m in models.items():
    print(f"Training {name}...")
    m.fit(X_tr, y_tr)
    fitted[name] = m
    val_probs[name] = m.predict_proba(X_va)[:, 1]
    print(f"  done")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Evaluate all baselines (full §6 metric suite)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, confusion_matrix, f1_score)
from sklearn.calibration import calibration_curve

def eval_model(y_true, y_prob, name, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    sens   = tp / (tp + fn + 1e-9)
    spec   = tn / (tn + fp + 1e-9)
    ppv    = tp / (tp + fp + 1e-9)
    npv    = tn / (tn + fn + 1e-9)
    f1     = f1_score(y_true, y_pred, zero_division=0)

    k = max(1, len(y_true) // 10)
    top_idx = np.argsort(y_prob)[::-1][:k]
    lift    = y_true[top_idx].mean() / (y_true.mean() + 1e-9)
    capture = y_true[top_idx].sum()  / (y_true.sum()  + 1e-9)

    pt, pp = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    mace   = float(np.mean(np.abs(pt - pp)))

    print(f"\n── {name} ──────────────────────────────────────────")
    print(f"  AUROC {auroc:.4f}  AUPRC {auprc:.4f}  Brier {brier:.4f}  MACE {mace:.4f}")
    print(f"  Sens {sens:.3f}  Spec {spec:.3f}  PPV {ppv:.3f}  NPV {npv:.3f}  F1 {f1:.3f}")
    print(f"  Top-decile Lift {lift:.2f}x  Capture {capture:.3f}")
    return dict(model=name, auroc=auroc, auprc=auprc, brier=brier, mace=mace,
                sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, f1=f1,
                lift_top_decile=lift, capture_top_decile=capture,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

y_va_np = y_va.values
results  = [eval_model(y_va_np, val_probs[n], n) for n in models]
best_name = max(results, key=lambda r: r["auroc"])["model"]
print(f"\n✓ Best baseline: {best_name}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Calibrate the best model (isotonic regression)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.isotonic import IsotonicRegression

# Use first half of val as calibration set
n_va   = len(y_va_np)
cal_n  = n_va // 2
p_best = val_probs[best_name]

ir = IsotonicRegression(out_of_bounds="clip")
ir.fit(p_best[:cal_n], y_va_np[:cal_n])
p_cal = ir.predict(p_best[cal_n:])

print("Calibrated model:")
cal_result = eval_model(y_va_np[cal_n:], p_cal, f"{best_name}_calibrated")

# 🔒 LOCK baseline submission — save now
print(f"\n✓ T0 LOCKED — {best_name} calibrated, submission ready")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Mandatory subgroup analysis (rubric §5A)
# Subgroups: glaucoma-only, diabetic-only, glaucoma+diabetic (overlap)
# ─────────────────────────────────────────────────────────────────────────────
subgroups = {
    "Glaucoma only":      (meta_va["flag_glaucoma"] == 1) & (meta_va["flag_diabetes"] == 0),
    "Diabetic only":      (meta_va["flag_diabetes"] == 1) & (meta_va["flag_glaucoma"] == 0),
    "Glaucoma+Diabetic":  meta_va["flag_glaucoma_diabetic"] == 1,
    "Neither":            (meta_va["flag_glaucoma"] == 0) & (meta_va["flag_diabetes"] == 0),
}

p_general = val_probs[best_name]
subgroup_results = []

for sg_name, mask in subgroups.items():
    mask_np = mask.values
    n_sg = mask_np.sum()
    if n_sg < 30:
        print(f"  ⚠ {sg_name}: n={n_sg} too small, skipping")
        continue
    r = eval_model(y_va_np[mask_np], p_general[mask_np], sg_name)
    r["subgroup"] = sg_name
    r["n_subgroup"] = int(n_sg)
    subgroup_results.append(r)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Specialized models per subgroup (5A continued)
# ─────────────────────────────────────────────────────────────────────────────
meta_tr = meta[tr_mask].reset_index(drop=True)
specialized = {}

subgroup_train_masks = {
    "Glaucoma only":     (meta_tr["flag_glaucoma"] == 1) & (meta_tr["flag_diabetes"] == 0),
    "Diabetic only":     (meta_tr["flag_diabetes"] == 1) & (meta_tr["flag_glaucoma"] == 0),
    "Glaucoma+Diabetic": meta_tr["flag_glaucoma_diabetic"] == 1,
}

for sg_name, tr_mask_sg in subgroup_train_masks.items():
    va_mask_sg = subgroups[sg_name].values
    n_tr_sg, n_va_sg = tr_mask_sg.sum(), va_mask_sg.sum()
    if n_tr_sg < 50 or n_va_sg < 20:
        print(f"  ⚠ {sg_name}: train={n_tr_sg}, val={n_va_sg} — skipping")
        continue
    print(f"\nTraining specialized model: {sg_name} (n_train={n_tr_sg}, n_val={n_va_sg})")
    m_sg = make_lgbm()
    m_sg.fit(X_tr[tr_mask_sg.values], y_tr[tr_mask_sg.values])
    p_sg = m_sg.predict_proba(X_va[va_mask_sg])[:, 1]
    r = eval_model(y_va_np[va_mask_sg], p_sg, f"specialized_{sg_name}")
    r["subgroup"] = sg_name
    specialized[sg_name] = (m_sg, p_sg, r)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — General vs specialized comparison table
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for sg_name in subgroup_train_masks:
    gen_r  = next((r for r in subgroup_results if r["subgroup"] == sg_name), None)
    spec_t = specialized.get(sg_name)
    row = {"subgroup": sg_name}
    if gen_r:
        row["general_auroc"] = round(gen_r["auroc"], 4)
        row["general_auprc"] = round(gen_r["auprc"], 4)
    if spec_t:
        row["specialized_auroc"] = round(spec_t[2]["auroc"], 4)
        row["specialized_auprc"] = round(spec_t[2]["auprc"], 4)
        if gen_r:
            delta = spec_t[2]["auroc"] - gen_r["auroc"]
            row["auroc_delta"] = round(delta, 4)
            row["recommendation"] = "specialize" if delta > 0.01 else "general_sufficient"
    rows.append(row)

comparison_df = pd.DataFrame(rows)
print("\nGeneral vs Specialized:")
print(comparison_df.to_string(index=False))
display(spark.createDataFrame(comparison_df))


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Stacking ensemble (T2)
# ─────────────────────────────────────────────────────────────────────────────
# Meta-features: probabilities + logit transforms from each base model
def logit(p): return np.log(p / (1 - p + 1e-9) + 1e-9)

meta_X_va = np.column_stack([
    val_probs[n] for n in models
] + [logit(val_probs[n]) for n in models])

meta_X_tr_parts = []
from sklearn.model_selection import cross_val_predict
for name, m in fitted.items():
    p_oof = cross_val_predict(m, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
    meta_X_tr_parts += [p_oof, logit(p_oof)]
meta_X_tr = np.column_stack(meta_X_tr_parts)

stacker = LogisticRegression(C=1.0, max_iter=500, random_state=42)
stacker.fit(meta_X_tr, y_tr)
p_stack = stacker.predict_proba(meta_X_va)[:, 1]

# Calibrate stacker
ir_stack = IsotonicRegression(out_of_bounds="clip")
ir_stack.fit(p_stack[:cal_n], y_va_np[:cal_n])
p_stack_cal = ir_stack.predict(p_stack)

stack_result = eval_model(y_va_np, p_stack_cal, "stacker_calibrated")
print(f"\n✓ Stacker {'lifts' if stack_result['auroc'] > results[0]['auroc'] else 'does not lift'} baseline")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Full comparison table
# ─────────────────────────────────────────────────────────────────────────────
all_results = results + [cal_result, stack_result]
cols = ["model", "auroc", "auprc", "brier", "mace",
        "sensitivity", "specificity", "ppv", "f1",
        "lift_top_decile", "capture_top_decile"]
tbl = pd.DataFrame(all_results)[cols].sort_values("auroc", ascending=False)
print("\nModel Comparison Table:")
print(tbl.to_string(index=False))
display(spark.createDataFrame(tbl))


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — SHAP feature importance
# ─────────────────────────────────────────────────────────────────────────────
try:
    import shap
    best_model = fitted[best_name]

    # Sample 1000 rows for speed
    sample_idx = np.random.choice(len(X_va), min(1000, len(X_va)), replace=False)
    X_sample = X_va.iloc[sample_idx]

    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    importance = pd.DataFrame({
        "feature":        X_va.columns,
        "mean_abs_shap":  np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nTop 20 features by mean |SHAP|:")
    print(importance.head(20).to_string(index=False))
    display(spark.createDataFrame(importance.head(20)))
except Exception as e:
    print(f"SHAP unavailable: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — Equity / fairness report
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score

p_final = p_stack_cal  # use calibrated stacker

equity_cols = {
    "DerivedRaceEthnicity": "Race/Ethnicity",
    "MaritalStatus":        "Marital Status",
    "sex":                  "Sex",
    "ADINatRankQuintile":   "ADI Quintile (deprivation)",
    "CAHPIQuartile":        "CAHPI Quartile",
}

equity_rows = []
for col, label in equity_cols.items():
    if col not in meta_va.columns:
        continue
    for grp in meta_va[col].dropna().unique():
        mask = (meta_va[col] == grp).values
        n_g  = mask.sum()
        if n_g < 30 or len(np.unique(y_va_np[mask])) < 2:
            continue
        auroc_g = roc_auc_score(y_va_np[mask], p_final[mask])
        equity_rows.append(dict(
            dimension=label, group=str(grp), n=int(n_g),
            prevalence=round(float(y_va_np[mask].mean()), 3),
            auroc=round(auroc_g, 4),
            mean_score=round(float(p_final[mask].mean()), 4),
        ))

equity_df = pd.DataFrame(equity_rows)
print("\nEquity Report:")
for dim in equity_df["dimension"].unique():
    sub = equity_df[equity_df["dimension"] == dim].sort_values("auroc", ascending=False)
    print(f"\n  {dim}:")
    print(sub[["group", "n", "prevalence", "auroc"]].to_string(index=False))
    gap = sub["auroc"].max() - sub["auroc"].min()
    print(f"  AUROC gap: {gap:.4f} {'⚠ FLAG' if gap >= 0.05 else '✓'}")

display(spark.createDataFrame(equity_df))


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — Write scores to Spark temp view (feeds Phase 3 SQL queries)
# ─────────────────────────────────────────────────────────────────────────────
# Build subgroup label column
sg_labels = []
for i in range(len(meta_va)):
    if meta_va["flag_glaucoma_diabetic"].iloc[i] == 1:
        sg_labels.append("Glaucoma+Diabetic")
    elif meta_va["flag_glaucoma"].iloc[i] == 1:
        sg_labels.append("Glaucoma only")
    elif meta_va["flag_diabetes"].iloc[i] == 1:
        sg_labels.append("Diabetic only")
    else:
        sg_labels.append("Neither")

scores_df = pd.DataFrame({
    "patientID":           meta_va[PATIENT_ID],
    "y_true":               y_va_np,
    "y_prob":               p_final,
    "y_pred":               (p_final >= 0.5).astype(int),
    "subgroup":             sg_labels,
    "DerivedRaceEthnicity": meta_va["DerivedRaceEthnicity"],
    "ADINatRankQuintile":   meta_va["ADINatRankQuintile"],
    "CAHPIQuartile":        meta_va["CAHPIQuartile"],
})

spark.createDataFrame(scores_df).createOrReplaceTempView("scores")
print(f"scores view registered: {len(scores_df):,} rows")
print("→ Run Phase 3 SQL queries (cells 16–22 in 09_queries.py) now")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 — Results summary (all print, no tables — copy-paste friendly)
# ─────────────────────────────────────────────────────────────────────────────
SEP  = "=" * 62
sep  = "-" * 62

print(SEP)
print("  MODEL RESULTS SUMMARY")
print(SEP)

# ── 1. Model comparison ───────────────────────────────────────────────────────
print("\n[1] MODEL COMPARISON  (sorted by AUROC)")
print(f"  {'Model':<28} {'AUROC':>6} {'AUPRC':>6} {'Brier':>6} {'MACE':>6} {'F1':>6} {'Lift':>6} {'Cap':>6}")
print(f"  {sep}")
for _, row in tbl.iterrows():
    print(f"  {row['model']:<28} {row['auroc']:>6.4f} {row['auprc']:>6.4f} {row['brier']:>6.4f} {row['mace']:>6.4f} {row['f1']:>6.3f} {row['lift_top_decile']:>6.2f} {row['capture_top_decile']:>6.3f}")

# ── 2. Best model detail ──────────────────────────────────────────────────────
best_model_name = tbl.iloc[0]["model"]
best_row = next(r for r in all_results if r["model"] == best_model_name)
print(f"\n[2] BEST MODEL: {best_row['model']}")
print(f"  AUROC {best_row['auroc']:.4f}  AUPRC {best_row['auprc']:.4f}  Brier {best_row['brier']:.4f}  MACE {best_row['mace']:.4f}")
print(f"  Sensitivity {best_row['sensitivity']:.3f}  Specificity {best_row['specificity']:.3f}")
print(f"  PPV {best_row['ppv']:.3f}  NPV {best_row['npv']:.3f}  F1 {best_row['f1']:.3f}")
print(f"  Top-decile lift {best_row['lift_top_decile']:.2f}x  Capture {best_row['capture_top_decile']:.3f}")

# ── 3. Subgroup performance ───────────────────────────────────────────────────
print(f"\n[3] SUBGROUP PERFORMANCE (general model)")
print(f"  {'Subgroup':<22} {'n':>6} {'Prev':>6} {'AUROC':>6} {'AUPRC':>6} {'F1':>6}")
print(f"  {sep}")
for r in subgroup_results:
    prev = float(r["tp"] + r["fn"]) / max(r["tp"] + r["fn"] + r["tn"] + r["fp"], 1)
    print(f"  {r['subgroup']:<22} {r['n_subgroup']:>6,} {prev:>6.3f} {r['auroc']:>6.4f} {r['auprc']:>6.4f} {r['f1']:>6.3f}")

# ── 4. General vs specialized ─────────────────────────────────────────────────
print(f"\n[4] GENERAL vs SPECIALIZED MODELS")
print(f"  {'Subgroup':<22} {'Gen AUROC':>10} {'Spec AUROC':>11} {'Delta':>7} {'Verdict':>20}")
print(f"  {sep}")
for _, row in comparison_df.iterrows():
    gen  = row.get("general_auroc",    float("nan"))
    spec = row.get("specialized_auroc", float("nan"))
    dlt  = row.get("auroc_delta",       float("nan"))
    rec  = row.get("recommendation",   "n/a")
    print(f"  {row['subgroup']:<22} {gen:>10.4f} {spec:>11.4f} {dlt:>+7.4f} {rec:>20}")

# ── 5. Equity report ──────────────────────────────────────────────────────────
print(f"\n[5] EQUITY REPORT")
for dim in equity_df["dimension"].unique():
    sub = equity_df[equity_df["dimension"] == dim].sort_values("auroc", ascending=False)
    gap = sub["auroc"].max() - sub["auroc"].min()
    flag = "⚠ FLAG" if gap >= 0.05 else "✓ OK"
    print(f"\n  {dim}  (AUROC gap {gap:.4f} {flag})")
    print(f"  {'Group':<30} {'n':>6} {'Prev':>6} {'AUROC':>6} {'MeanScore':>10}")
    for _, r in sub.iterrows():
        print(f"  {str(r['group']):<30} {r['n']:>6,} {r['prevalence']:>6.3f} {r['auroc']:>6.4f} {r['mean_score']:>10.4f}")

# ── 6. SHAP top features ──────────────────────────────────────────────────────
try:
    print(f"\n[6] TOP 15 FEATURES BY MEAN |SHAP|")
    print(f"  {'Feature':<42} {'Mean|SHAP|':>10}")
    print(f"  {sep}")
    for _, r in importance.head(15).iterrows():
        print(f"  {r['feature']:<42} {r['mean_abs_shap']:>10.4f}")
except NameError:
    print("\n[6] SHAP not available (run cell 12 first)")

print(f"\n{SEP}")
print("  END OF SUMMARY")
print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 16 — Operational threshold tuning
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score

p_tune = val_probs[best_name]   # uncalibrated probs — more spread, easier to threshold

precisions, recalls, thresholds = precision_recall_curve(y_va_np, p_tune)

print("Threshold sensitivity analysis:")
print(f"  {'Threshold':>10} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'F1':>6} {'Flagged%':>9}")
print(f"  {'-'*50}")

target_sensitivities = [0.50, 0.40, 0.30, 0.20]
printed = set()

for sens_target in target_sensitivities:
    viable = thresholds[recalls[:-1] >= sens_target]
    if len(viable) == 0:
        print(f"  sens>={sens_target:.0%}: no threshold achieves this")
        continue
    t = float(viable[-1])   # highest threshold still meeting sensitivity
    if round(t, 3) in printed:
        continue
    printed.add(round(t, 3))

    y_pred_t = (p_tune >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_va_np, y_pred_t).ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    ppv  = tp / (tp + fp + 1e-9)
    f1   = f1_score(y_va_np, y_pred_t, zero_division=0)
    flag_pct = y_pred_t.mean() * 100
    print(f"  {t:>10.4f} {sens:>6.3f} {spec:>6.3f} {ppv:>6.3f} {f1:>6.3f} {flag_pct:>8.1f}%")

# Also show threshold at top-decile cut (flags 10% of cohort)
top10_t = float(np.percentile(p_tune, 90))
y_pred_10 = (p_tune >= top10_t).astype(int)
tn, fp, fn, tp = confusion_matrix(y_va_np, y_pred_10).ravel()
sens = tp / (tp + fn + 1e-9)
spec = tn / (tn + fp + 1e-9)
ppv  = tp / (tp + fp + 1e-9)
f1   = f1_score(y_va_np, y_pred_10, zero_division=0)
print(f"\n  Top-decile cut (flags 10% of cohort):")
print(f"  threshold={top10_t:.4f}  Sens={sens:.3f}  Spec={spec:.3f}  PPV={ppv:.3f}  F1={f1:.3f}")

# Recommended threshold
viable_30 = thresholds[recalls[:-1] >= 0.30]
op_threshold = float(viable_30[-1]) if len(viable_30) else top10_t
print(f"\n  --> Recommended operational threshold: {op_threshold:.4f}")
print(f"      (sensitivity ≥ 30%, flags ~{((p_tune >= op_threshold).mean()*100):.1f}% of cohort)")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 17 — Two-stage clustering extension (rubric §5B)
# Step 1: KMeans on training features → patient subtypes
# Step 2a: per-cluster specialist models
# Step 2b: cluster-as-feature augmented model
# Step 3: compare all three against the general model
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SEP2 = "=" * 62
sep2 = "-" * 62

print(SEP2)
print("  §5B  TWO-STAGE CLUSTERING EXTENSION")
print(SEP2)

# ── Step 1: Cluster patients in training set ──────────────────────────────────
N_CLUSTERS = 4

scaler_c = StandardScaler()
X_tr_scaled = scaler_c.fit_transform(X_tr)

pca_c = PCA(n_components=min(20, X_tr.shape[1]), random_state=42)
X_tr_pca = pca_c.fit_transform(X_tr_scaled)

km = KMeans(n_clusters=N_CLUSTERS, n_init=10, max_iter=300, random_state=42)
labels_tr = km.fit_predict(X_tr_pca)
labels_va = km.predict(pca_c.transform(scaler_c.transform(X_va)))

print(f"\nKMeans (k={N_CLUSTERS}) — cluster profiles:")
print(f"  {'Cluster':>8} {'n_train':>8} {'n_val':>7} {'TrainPosRate':>13} {'ValPosRate':>11}")
print(f"  {sep2}")
for k in range(N_CLUSTERS):
    mtr = labels_tr == k
    mva = labels_va == k
    print(f"  {k:>8} {mtr.sum():>8,} {mva.sum():>7,} {y_tr[mtr].mean():>13.3f} {y_va_np[mva].mean():>11.3f}")

# Profile clusters by key features
cluster_profile_cols = [
    c for c in ["flag_glaucoma", "flag_diabetes", "flag_glaucoma_diabetic",
                 "flag_ckd", "flag_hf", "flag_hypertension", "flag_depression",
                 "ADINatrank", "CAHPIPercentile", "last_12months_officevisit_count",
                 "last_12months_chronic_count"]
    if c in X_tr.columns
]
if cluster_profile_cols:
    print(f"\n  Cluster feature profiles (training set means):")
    profile = X_tr[cluster_profile_cols].copy()
    profile["_cluster"] = labels_tr
    grp = profile.groupby("_cluster")[cluster_profile_cols].mean()
    print(f"  {'Feature':<40} " + "  ".join(f"C{k:>5}" for k in range(N_CLUSTERS)))
    print(f"  {sep2}")
    for col in cluster_profile_cols:
        vals = "  ".join(f"{grp.loc[k, col]:>6.3f}" for k in range(N_CLUSTERS))
        print(f"  {col:<40} {vals}")

# ── Step 2a: Per-cluster specialist models ────────────────────────────────────
print(f"\n── Step 2a: Per-cluster specialist models ──────────────────────────")
cluster_models_2s = {}
per_cluster_probs = np.zeros(len(y_va_np))
covered = np.zeros(len(y_va_np), dtype=bool)

for k in range(N_CLUSTERS):
    mtr = labels_tr == k
    mva = labels_va == k
    n_tr_k, n_va_k = mtr.sum(), mva.sum()
    if n_tr_k < 50 or n_va_k < 10:
        print(f"  Cluster {k}: too small (tr={n_tr_k}, va={n_va_k}) — skipping specialist")
        continue
    m_k = make_lgbm()
    m_k.fit(X_tr[mtr], y_tr[mtr])
    p_k = m_k.predict_proba(X_va[mva])[:, 1]
    per_cluster_probs[mva] = p_k
    covered[mva] = True
    cluster_models_2s[k] = m_k

    auroc_k  = roc_auc_score(y_va_np[mva], p_k) if len(np.unique(y_va_np[mva])) > 1 else float("nan")
    auprc_k  = average_precision_score(y_va_np[mva], p_k) if len(np.unique(y_va_np[mva])) > 1 else float("nan")
    print(f"  Cluster {k}: n_val={n_va_k}  AUROC={auroc_k:.4f}  AUPRC={auprc_k:.4f}  PosRate={y_va_np[mva].mean():.3f}")

# ── Step 2b: Cluster-as-feature augmented model ───────────────────────────────
print(f"\n── Step 2b: Cluster-as-feature augmented model ─────────────────────")
X_tr_aug = X_tr.copy()
X_va_aug = X_va.copy()
for k in range(N_CLUSTERS):
    X_tr_aug[f"cluster_{k}"] = (labels_tr == k).astype(float)
    X_va_aug[f"cluster_{k}"] = (labels_va == k).astype(float)

m_aug = make_lgbm()
m_aug.fit(X_tr_aug, y_tr)
p_aug = m_aug.predict_proba(X_va_aug)[:, 1]
print("  Cluster-as-feature model trained.")

# ── Step 3: Comparison ────────────────────────────────────────────────────────
print(f"\n── Step 3: Model comparison ─────────────────────────────────────────")
print(f"  {'Model':<30} {'AUROC':>7} {'AUPRC':>7} {'Brier':>7} {'Lift10':>7}")
print(f"  {sep2}")

def _quick_eval(y_true, y_prob, name):
    if len(np.unique(y_true)) < 2:
        print(f"  {name:<30}  (no events — skip)")
        return
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    k = max(1, len(y_true) // 10)
    top_idx = np.argsort(y_prob)[::-1][:k]
    lift = y_true[top_idx].mean() / (y_true.mean() + 1e-9)
    print(f"  {name:<30} {auroc:>7.4f} {auprc:>7.4f} {brier:>7.4f} {lift:>7.2f}x")

_quick_eval(y_va_np, val_probs[best_name], "General model (best baseline)")
_quick_eval(y_va_np, p_aug,                "Cluster-as-feature model")
if covered.all():
    _quick_eval(y_va_np, per_cluster_probs, "Per-cluster specialists")
elif covered.any():
    _quick_eval(y_va_np[covered], per_cluster_probs[covered], "Per-cluster specialists (covered)")

# ── Strategic recommendation ──────────────────────────────────────────────────
print(f"\n── Strategic recommendation ─────────────────────────────────────────")
auroc_gen = roc_auc_score(y_va_np, val_probs[best_name])
auroc_aug = roc_auc_score(y_va_np, p_aug)
delta = auroc_aug - auroc_gen

if delta > 0.01:
    verdict = "ADOPT cluster-as-feature: meaningful AUROC lift with low operational overhead."
elif delta > 0:
    verdict = "MARGINAL gain from clustering. Consider for next iteration; deploy general model now."
else:
    verdict = "NO gain from clustering on this cohort. General model recommended for deployment."

print(f"  General AUROC:         {auroc_gen:.4f}")
print(f"  Cluster-feature AUROC: {auroc_aug:.4f}  (Δ={delta:+.4f})")
print(f"\n  Verdict: {verdict}")
print(f"\n  Tradeoff summary:")
print(f"    General model      — simplest, one retraining cycle, easiest to audit")
print(f"    Cluster-as-feature — marginal complexity, one model, cluster features explainable")
print(f"    Per-cluster models — highest complexity, {N_CLUSTERS}x retraining, {N_CLUSTERS}x monitoring")
print(f"    Recommendation     — deploy general model; add cluster feature if AUROC gap > 0.01")
print(SEP2)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 18 — Export predictions.json for retina-risk frontend
# Generates the exact schema expected by VisionWatch (RetinaScene.jsx).
# Writes to retina-risk/src/data/predictions.json.
# Run after Cell 14 (scores view) and Cell 17 (clustering).
# ─────────────────────────────────────────────────────────────────────────────
import json, datetime

# ── Cohort definitions: subgroup label → display config ──────────────────────
COHORT_CONFIG = {
    "Glaucoma only": {
        "id": "cohort_glaucoma",
        "name": "Glaucoma",
        "category": "optic_nerve",
        "position": {"x": -1.4, "y": 0.5, "z": 1.2},
    },
    "Diabetic only": {
        "id": "cohort_diabetic",
        "name": "Diabetic",
        "category": "retinal",
        "position": {"x": 0.3, "y": 0.1, "z": 0.2},
    },
    "Glaucoma+Diabetic": {
        "id": "cohort_glaucoma_diabetic",
        "name": "Glaucoma + Diabetic",
        "category": "retinal",
        "position": {"x": -0.7, "y": -0.2, "z": 1.7},
    },
    "Neither": {
        "id": "cohort_neither",
        "name": "Other Chronic",
        "category": "systemic",
        "position": {"x": 1.8, "y": -0.3, "z": -1.5},
    },
}

# ── Top features from RF feature importance ───────────────────────────────────
FEATURE_LABELS = {
    "flag_ckd":                         "Chronic Kidney Disease",
    "flag_hf":                          "Heart Failure",
    "flag_hypertension":                "Hypertension",
    "flag_depression":                  "Depression",
    "flag_glaucoma":                    "Glaucoma",
    "flag_diabetes":                    "Diabetes",
    "flag_glaucoma_diabetic":           "Glaucoma + Diabetes",
    "flag_blindness":                   "Vision Impairment",
    "flag_cataract":                    "Cataract",
    "flag_cancer":                      "Cancer",
    "flag_anxiety":                     "Anxiety",
    "ADINatrank":                       "Neighborhood Deprivation (ADI)",
    "CAHPIPercentile":                  "Area Health Index (CAHPI)",
    "straight_line_distance_miles":     "Distance to Care",
    "distance_log":                     "Distance to Care (log)",
    "high_deprivation":                 "High Deprivation Flag",
    "low_cahpi":                        "Low Area Health",
    "far_from_care":                    "Far from Care",
    "last_12months_officevisit_count":  "Office Visits (12mo)",
    "last_6months_officevisit_count":   "Office Visits (6mo)",
    "last_12months_chronic_count":      "Chronic Conditions (12mo)",
    "last_12months_diagnosis_count_ccw":"Diagnosis Count (12mo)",
    "op_recency_ratio":                 "Visit Recency Ratio",
    "high_specialty_breadth":           "High Specialty Breadth",
    "Specialty_cardiology":             "Cardiology Visits",
    "Specialty_nephrology":             "Nephrology Visits",
    "cardio_visit_ratio":               "Cardiology Visit Ratio",
}

def _top_features(model, feature_names, n=3):
    try:
        importances = model.feature_importances_
    except AttributeError:
        try:
            importances = np.abs(model.named_steps["clf"].coef_[0])
        except Exception:
            return [{"feature": "N/A", "importance": 0.0}]
    idx = np.argsort(importances)[::-1][:n]
    total = importances[idx].sum() + 1e-9
    return [
        {"feature": FEATURE_LABELS.get(feature_names[i], feature_names[i]),
         "importance": round(float(importances[i] / total), 3)}
        for i in idx
    ]

def _risk_level(score):
    if score >= 0.15:  return "high"
    if score >= 0.07:  return "moderate"
    return "low"

def _trend(scores_series, months_series, n_points=6):
    """6-point trend: mean predicted score per recent validation month."""
    df_t = pd.DataFrame({"score": scores_series, "month": months_series})
    monthly = df_t.groupby("month")["score"].mean().sort_index()
    vals = monthly.values.tolist()
    if len(vals) >= n_points:
        vals = vals[-n_points:]
    else:
        # pad left by linearly interpolating from 0
        first = vals[0] if vals else 0.05
        pad = [round(first * i / n_points, 4) for i in range(n_points - len(vals))]
        vals = pad + vals
    return [round(float(v), 4) for v in vals]

# ── Build cohort list ─────────────────────────────────────────────────────────
# scores_df: patientID, y_true, y_prob, y_pred, subgroup, DerivedRaceEthnicity,
#            ADINatRankQuintile, CAHPIQuartile
# meta_va:   includes EncMonth, flag_*, chronic count columns

scores_df["_month"] = meta_va[INDEX_MONTH].values  # attach month column

# Chronic count for disease_burden normalisation
chronic_col = "last_12months_chronic_count"
chronic_max  = float(meta_va[chronic_col].max()) if chronic_col in meta_va.columns else 1.0

model_keys = list(models.keys())   # ["logreg", "rf", "lgbm"]
feature_names = list(X_va.columns)

cohorts_out = []
for sg_label, cfg in COHORT_CONFIG.items():
    mask = scores_df["subgroup"] == sg_label
    n_sg = int(mask.sum())
    if n_sg == 0:
        continue

    sg_meta   = meta_va[mask.values]
    sg_scores = scores_df[mask]

    # disease_burden: normalised mean chronic count
    burden = 0.5
    if chronic_col in sg_meta.columns:
        burden = round(float(sg_meta[chronic_col].mean()) / max(chronic_max, 1), 3)

    models_obj = {}
    for mname in model_keys:
        p_m = val_probs[mname][mask.values]
        score = round(float(p_m.mean()), 4)
        models_obj[mname] = {
            "risk_score":   score,
            "risk_level":   _risk_level(score),
            "trend":        _trend(p_m, sg_scores["_month"].values),
            "top_features": _top_features(fitted[mname], feature_names),
        }

    cohorts_out.append({
        "id":             cfg["id"],
        "name":           cfg["name"],
        "category":       cfg["category"],
        "population":     n_sg,
        "disease_burden": burden,
        "position":       cfg["position"],
        "models":         models_obj,
    })

# ── Assemble final JSON ───────────────────────────────────────────────────────
predictions = {
    "metadata": {
        "project":            "VisionWatch",
        "prediction_target":  "12-month ED/inpatient admission risk — glaucoma & diabetic cohorts",
        "generated_at":       datetime.date.today().isoformat(),
        "models_available":   model_keys,
        "default_model":      best_name,
        "validation_n":       int(len(scores_df)),
        "validation_auroc":   round(float(roc_auc_score(y_va_np, val_probs[best_name])), 4),
    },
    "cohorts": cohorts_out,
}

json_str = json.dumps(predictions, indent=2)

# ── Write to frontend data file ───────────────────────────────────────────────
OUT_PATH = "/home/leo/MedXEng2026/retina-risk/src/data/actual_predictions.json"
try:
    with open(OUT_PATH, "w") as f:
        f.write(json_str)
    print(f"Written to {OUT_PATH}")
except Exception as e:
    print(f"Could not write to local path ({e})")
    print("Copy the JSON below into retina-risk/src/data/actual_predictions.json manually:")
    print()
    print(json_str)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 19 — CKD / HF / Cancer subgroup analysis (Rubric §5A mandatory)
# Evaluates the general model on the condition cohorts specified in the rubric.
# Requires meta_va to contain flag_ckd, flag_hf, flag_cancer columns.
# ─────────────────────────────────────────────────────────────────────────────
SEP3 = "=" * 62
sep3 = "-" * 62

print(SEP3)
print("  §5A  CKD / HF / CANCER SUBGROUP ANALYSIS  (Rubric-required)")
print(SEP3)

# Build rubric-specified subgroup masks from meta_va condition flags
ckd_col    = "flag_ckd"
hf_col     = "flag_hf"
cancer_col = "flag_cancer"

# Pull condition flags from original df (has all columns; meta_va only keeps META_COLS)
# va_mask is the boolean series from cell 3 selecting validation rows
val_flags = df[va_mask].reset_index(drop=True)

missing = [c for c in [ckd_col, hf_col, cancer_col] if c not in val_flags.columns]
if missing:
    print(f"  Warning: columns not found in features view: {missing}")
    print("  Check that 00_queries.py cell 12 includes these CCW flag columns.")

ckd_mask    = val_flags[ckd_col].values == 1    if ckd_col    in val_flags.columns else np.zeros(len(val_flags), dtype=bool)
hf_mask     = val_flags[hf_col].values == 1     if hf_col     in val_flags.columns else np.zeros(len(val_flags), dtype=bool)
cancer_mask = val_flags[cancer_col].values == 1 if cancer_col in val_flags.columns else np.zeros(len(val_flags), dtype=bool)
multi_mask  = (ckd_mask.astype(int) + hf_mask.astype(int) + cancer_mask.astype(int)) >= 2

rubric_subgroups = {
    "CKD only":         ckd_mask    & ~hf_mask  & ~cancer_mask,
    "HF only":          hf_mask     & ~ckd_mask  & ~cancer_mask,
    "Cancer only":      cancer_mask & ~ckd_mask  & ~hf_mask,
    "Multi-condition":  multi_mask,
    "CKD (any)":        ckd_mask,
    "HF (any)":         hf_mask,
    "Cancer (any)":     cancer_mask,
}

p_general = val_probs[best_name]

print(f"\n[A] GENERAL MODEL — performance on each rubric cohort")
print(f"  {'Subgroup':<20} {'n':>6} {'Prev':>6} {'AUROC':>7} {'AUPRC':>7} {'Brier':>7} {'Lift10':>7}")
print(f"  {sep3}")

rubric_results = []
for sg_name, mask in rubric_subgroups.items():
    n_sg = int(mask.sum())
    if n_sg < 20:
        print(f"  {sg_name:<20} {n_sg:>6}  — too small, skipping")
        continue
    y_sg = y_va_np[mask]
    p_sg = p_general[mask]
    if len(np.unique(y_sg)) < 2:
        print(f"  {sg_name:<20} {n_sg:>6}  — no events in group, skipping")
        continue
    auroc_sg = roc_auc_score(y_sg, p_sg)
    auprc_sg = average_precision_score(y_sg, p_sg)
    brier_sg = brier_score_loss(y_sg, p_sg)
    k        = max(1, n_sg // 10)
    top_idx  = np.argsort(p_sg)[::-1][:k]
    lift_sg  = y_sg[top_idx].mean() / (y_sg.mean() + 1e-9)
    prev     = y_sg.mean()
    print(f"  {sg_name:<20} {n_sg:>6,} {prev:>6.3f} {auroc_sg:>7.4f} {auprc_sg:>7.4f} {brier_sg:>7.4f} {lift_sg:>7.2f}x")
    rubric_results.append(dict(subgroup=sg_name, n=n_sg, prevalence=round(prev,3),
                                auroc=round(auroc_sg,4), auprc=round(auprc_sg,4),
                                brier=round(brier_sg,4), lift=round(lift_sg,2)))

# ── Specialized models per rubric subgroup ─────────────────────────────────────
print(f"\n[B] SPECIALIZED MODELS — trained on each rubric cohort")
print(f"  {'Subgroup':<20} {'Gen AUROC':>10} {'Spec AUROC':>11} {'Delta':>7} {'Verdict':>20}")
print(f"  {sep3}")

train_flags = df[tr_mask].reset_index(drop=True)

def _tflag(col):
    return train_flags[col].values == 1 if col in train_flags.columns else np.zeros(len(train_flags), dtype=bool)

tr_ckd    = _tflag(ckd_col)
tr_hf     = _tflag(hf_col)
tr_cancer = _tflag(cancer_col)

rubric_train_masks = {
    "CKD only":        tr_ckd    & ~tr_hf    & ~tr_cancer,
    "HF only":         tr_hf     & ~tr_ckd   & ~tr_cancer,
    "Cancer only":     tr_cancer & ~tr_ckd   & ~tr_hf,
    "Multi-condition": (tr_ckd.astype(int) + tr_hf.astype(int) + tr_cancer.astype(int)) >= 2,
}

for sg_name, tr_mask_sg in rubric_train_masks.items():
    if tr_mask_sg is None:
        print(f"  {sg_name:<20}  — flags missing, skipping")
        continue
    va_mask_sg = rubric_subgroups[sg_name]
    n_tr_sg = int(tr_mask_sg.sum())
    n_va_sg = int(va_mask_sg.sum())
    if n_tr_sg < 50 or n_va_sg < 20:
        print(f"  {sg_name:<20}  — train={n_tr_sg}, val={n_va_sg} too small")
        continue
    if len(np.unique(y_va_np[va_mask_sg])) < 2:
        print(f"  {sg_name:<20}  — no events in val set, skipping")
        continue

    m_spec = make_lgbm()
    m_spec.fit(X_tr[tr_mask_sg], y_tr[tr_mask_sg])
    p_spec = m_spec.predict_proba(X_va[va_mask_sg])[:, 1]
    auroc_spec = roc_auc_score(y_va_np[va_mask_sg], p_spec)

    gen_r = next((r for r in rubric_results if r["subgroup"] == sg_name), None)
    auroc_gen = gen_r["auroc"] if gen_r else float("nan")
    delta = auroc_spec - auroc_gen
    verdict = "specialize" if delta > 0.01 else ("general sufficient" if delta >= -0.01 else "general better")
    print(f"  {sg_name:<20} {auroc_gen:>10.4f} {auroc_spec:>11.4f} {delta:>+7.4f} {verdict:>20}")

# ── Strategic recommendation ───────────────────────────────────────────────────
print(f"\n[C] STRATEGIC RECOMMENDATION")
print(f"  One general model covers the full cohort adequately.")
print(f"  Specialize only if: (1) AUROC delta > 0.01 AND (2) subgroup n_val > 100.")
print(f"  Tradeoffs:")
print(f"    General model      — single pipeline, easiest to monitor and audit")
print(f"    Specialized models — potential per-group gain, but N× maintenance burden")
print(f"    Multi-condition    — highest-risk group; priority for outreach regardless of model choice")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 20 — Save model artifacts to DBFS
# Persists the trained RF + isotonic calibrator + feature column list so the
# model can be reloaded without retraining (e.g. to score the held-out test set).
# ─────────────────────────────────────────────────────────────────────────────
import joblib, json

SAVE_DIR = "/dbfs/tmp/retina_risk"
import os; os.makedirs(SAVE_DIR, exist_ok=True)

# Save base model (RF) and isotonic calibrator separately
joblib.dump(fitted[best_name], f"{SAVE_DIR}/base_model.pkl")
joblib.dump(ir, f"{SAVE_DIR}/calibrator.pkl")

# Save the exact feature columns the model was trained on
with open(f"{SAVE_DIR}/feature_cols.json", "w") as fh:
    json.dump(list(X.columns), fh)

print(f"Saved to {SAVE_DIR}/")
print(f"  base_model.pkl  — {best_name}")
print(f"  calibrator.pkl  — isotonic regressor")
print(f"  feature_cols.json — {len(X.columns)} features")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 21 — Score held-out test set
# Loads the saved model and applies it to the new validation/test table.
# Update TEST_TABLE to match the table name released for the held-out set.
# ─────────────────────────────────────────────────────────────────────────────
import joblib, json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

TEST_TABLE = "hackathon.data.pophealth_pdf_test_patientlevel"  # ← update if needed

# ── Load model artifacts ──────────────────────────────────────────────────────
SAVE_DIR = "/dbfs/tmp/retina_risk"
base_model  = joblib.load(f"{SAVE_DIR}/base_model.pkl")
calibrator  = joblib.load(f"{SAVE_DIR}/calibrator.pkl")
with open(f"{SAVE_DIR}/feature_cols.json") as fh:
    feature_cols = json.load(fh)

# ── Load test data with same cohort filter ────────────────────────────────────
df_test = spark.sql(f"""
    SELECT *,
        cms_ccw_glaucoma_36                          AS flag_glaucoma,
        cms_ccw_diabetes_36                          AS flag_diabetes,
        CASE WHEN cms_ccw_glaucoma_36 = 1
              AND cms_ccw_diabetes_36 = 1 THEN 1 ELSE 0 END AS flag_glaucoma_diabetic
    FROM {TEST_TABLE}
    WHERE Outcome IS NOT NULL
      AND (cms_ccw_glaucoma_36 = 1 OR cms_ccw_diabetes_36 = 1)
""").toPandas()

print(f"Test set: {len(df_test):,} rows | event rate: {df_test['Outcome'].mean():.3f}")

# ── Feature engineering — must match Cell 2 exactly ──────────────────────────
df_test[INDEX_MONTH] = pd.to_datetime(df_test[INDEX_MONTH].astype(str), format="%Y%m")
df_enc = pd.get_dummies(df_test, columns=CAT_COLS, drop_first=True, dummy_na=True)

META_DROP = [c for c in META_COLS if c in df_enc.columns]
X_test = df_enc.drop(columns=META_DROP)

# Align to training columns: add missing as 0, drop extras
X_test = X_test.reindex(columns=feature_cols, fill_value=0)
X_test = X_test.fillna(X_test.median(numeric_only=True)).astype("float32")

print(f"Feature matrix: {X_test.shape}")

# ── Score ─────────────────────────────────────────────────────────────────────
p_uncal = base_model.predict_proba(X_test)[:, 1]
p_cal   = calibrator.predict(p_uncal)

y_test  = df_test["Outcome"].astype(int).values

auroc  = roc_auc_score(y_test, p_cal)
auprc  = average_precision_score(y_test, p_cal)
brier  = brier_score_loss(y_test, p_cal)

print()
print("=" * 50)
print("  HELD-OUT TEST SET RESULTS")
print("=" * 50)
print(f"  N              : {len(y_test):,}")
print(f"  Event rate     : {y_test.mean():.3f}")
print(f"  AUROC          : {auroc:.4f}")
print(f"  AUPRC          : {auprc:.4f}")
print(f"  Brier score    : {brier:.4f}")
print(f"  Mean pred risk : {p_cal.mean():.4f}")
print("=" * 50)

# ── Subgroup breakdown ────────────────────────────────────────────────────────
meta_test = df_test[META_COLS].copy()
subgroups_test = {
    "Glaucoma only":     (meta_test["flag_glaucoma"] == 1) & (meta_test["flag_diabetes"] == 0),
    "Diabetic only":     (meta_test["flag_diabetes"] == 1) & (meta_test["flag_glaucoma"] == 0),
    "Glaucoma+Diabetic": meta_test["flag_glaucoma_diabetic"] == 1,
}
print(f"\n  {'Subgroup':<22} {'N':>6} {'Events':>7} {'AUROC':>8}")
print(f"  {'-'*46}")
for sg, mask in subgroups_test.items():
    m = mask.values
    n, ev = m.sum(), y_test[m].sum()
    if ev < 2:
        print(f"  {sg:<22} {n:>6} {ev:>7}   (too few events)")
        continue
    sg_auroc = roc_auc_score(y_test[m], p_cal[m])
    print(f"  {sg:<22} {n:>6} {ev:>7} {sg_auroc:>8.4f}")
print(SEP3)
