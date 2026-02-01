# ============================================================
# RANDOM FOREST + SMOTE + COST
# FULL PAPER-ALIGNED PIPELINE (SINGLE CELL)
# ============================================================

# ---------- Install deps (run once) ----------
#!pip -q install gdown imbalanced-learn scikit-learn shap umap-learn matplotlib seaborn

# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import gdown

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# 2. LOAD DATASET (OFFICIAL PAPER DATA)
# ============================================================

FILE_ID = "1hXqJhaB3uKYzHoz1S31YKnUNW6Tjgfem"
CSV_PATH = "embarazo_saludable.csv"

gdown.download(
    url=f"https://drive.google.com/uc?id={FILE_ID}",
    output=CSV_PATH,
    quiet=False
)

df_raw = pd.read_csv(CSV_PATH, encoding="latin1", low_memory=False)

# ============================================================
# 3. NORMALIZE COLUMN NAMES
# ============================================================

df_raw.columns = (
    df_raw.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("á", "a")
    .str.replace("é", "e")
    .str.replace("í", "i")
    .str.replace("ó", "o")
    .str.replace("ú", "u")
    .str.replace("ñ", "n")
    .str.replace(".", "", regex=False)
)

# ============================================================
# 4. FEATURE SELECTION (26 VARIABLES – PAPER)
# ============================================================

FEATURE_MAP = {
    "edad": "age",
    "hijos_nacidos_vivos": "lb",
    "hijos_nacidos_muertos": "sb",
    "pas": "sbp",
    "pad": "dbp",
    "imc": "bmi",
    "crl": "crl",
    "dbp": "bpd",
    "cc": "hc",
    "ca": "ac",
    "lf": "fl",
    "ila": "ala",
    "hb": "hb",
    "glucosa": "gc",
    "latido_fetal": "fh",
    "posicion_del_feto": "fp",
    "intraut": "iu",
    "1ert": "fst",
    "2do_t": "sndt",
    "3er_t": "trdt",
    "placenta": "pl",
    "anemia": "an"
}

df = df_raw[list(FEATURE_MAP.keys())].rename(columns=FEATURE_MAP)

# ============================================================
# 5. DERIVED VARIABLES (PAPER LOGIC)
# ============================================================

# Pregnancy weeks (approximation from trimester flags)
df["wks"] = np.select(
    [df["fst"] == 1, df["sndt"] == 1, df["trdt"] == 1],
    [10, 20, 32],
    default=np.nan
)

# Sexually transmitted infections (merged)
df["sxt"] = (
    (df_raw.get("vih", 0) == 1) |
    (df_raw.get("sifilis", 0) == 1) |
    (df_raw.get("vhb", 0) == 1)
).astype(int)

# Urine test aggregation
def derive_urine(row):
    if row.get("proteinas", 0) == 1:
        return "protein"
    if row.get("nitritos", 0) == 1:
        return "nitrite"
    if row.get("leucocitos", 0) == 1:
        return "leukocyte"
    if row.get("orina:negativo", 0) == 1 or row.get("normal", 0) == 1:
        return "normal"
    return "unknown"

df["ur"] = df_raw.apply(derive_urine, axis=1)

# ============================================================
# 6. REFERRAL LABEL (GROUND TRUTH)
# ============================================================

referral_cols = [
    "ref:_cs",
    "ref:_h_distrital",
    "ref:_hregional",
    "ref:_hnacional"
]

existing = [c for c in referral_cols if c in df_raw.columns]
df["referral"] = (df_raw[existing].notna().sum(axis=1) > 0).astype(int)

# ============================================================
# 7. TYPE FIXES
# ============================================================

numeric_cols = [
    "age","lb","sb","sbp","dbp","bmi","crl","bpd","hc",
    "ac","fl","ala","hb","gc","wks"
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ============================================================
# 8. ML MATRICES
# ============================================================

X = df.drop(columns=["referral"])
y = df["referral"]

# Drop broken feature (all NaN → SMOTE geometry explodes)
X = X.drop(columns=["wks"])

# One-hot encode categoricals (paper)
X = pd.get_dummies(X, drop_first=True)

# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================================
# 10. COST RATIO (INVERSE IMBALANCE)
# ============================================================

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
cost_ratio = n_neg / n_pos   # ≈ 11

class_weights = {0: 1, 1: cost_ratio}

print(f"\nCost ratio (FN:FP) ≈ {cost_ratio:.2f}:1")

# ============================================================
# 11. RF + SMOTE PIPELINE
# ============================================================

rf_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("smote", SMOTE(random_state=42, k_neighbors=3)),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight=class_weights,
        n_jobs=-1,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

# ============================================================
# 12. PROBABILITIES
# ============================================================

probs = rf_pipeline.predict_proba(X_test)[:, 1]

# ============================================================
# 13. THRESHOLD SELECTION (G-MEAN)
# ============================================================

thresholds = np.linspace(0.01, 0.99, 300)
best_gmean, best_threshold = 0, 0

for t in thresholds:
    y_tmp = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_tmp).ravel()

    if (tp + fn) == 0 or (tn + fp) == 0:
        continue

    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    gmean = np.sqrt(se * sp)

    if gmean > best_gmean:
        best_gmean = gmean
        best_threshold = t

print(f"\nBest threshold (G-Mean): {best_threshold:.3f}")
print(f"Best G-Mean: {best_gmean:.3f}")

# ============================================================
# 14. FINAL EVALUATION
# ============================================================

y_pred = (probs >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
f1 = f1_score(y_test, y_pred)
g_mean = np.sqrt(specificity * sensitivity)

print("\n================ RF RESULTS ================")
print(f"Specificity (Sp): {specificity:.3f}")
print(f"Sensitivity (Se): {sensitivity:.3f}")
print(f"F1-score        : {f1:.3f}")
print(f"G-Mean          : {g_mean:.3f}")

print("\nConfusion Matrix")
print(f"TN={tn}, FP={fp}")
print(f"FN={fn}, TP={tp}")

# ============================================================
# SHAP + UMAP INTERPRETABILITY — RANDOM FOREST (KERNEL SHAP)
# ============================================================

import shap
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline

# ============================================================
# 1. PREPROCESS DATA (SAME AS SVM LOGIC)
# ============================================================

# Only imputer, NO SMOTE, NO scaling
preprocess = Pipeline(steps=[
    ("imputer", rf_pipeline.named_steps["imputer"])
])

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc  = preprocess.transform(X_test)

feature_names = X_train.columns.tolist()

# ============================================================
# 2. KERNEL SHAP FOR RANDOM FOREST
# ============================================================

# Small background for speed (same as SVM)
background = shap.sample(X_train_proc, 50, random_state=42)
X_test_sample = X_test_proc[:50]

def rf_decision_fn(X):
    return rf_pipeline.named_steps["rf"].predict_proba(X)[:, 1]

explainer = shap.KernelExplainer(
    rf_decision_fn,
    background
)

shap_values = explainer.shap_values(
    X_test_sample,
    nsamples=100
)

# ==========================
# 2.1 SHAP SUMMARY (GLOBAL)
# ==========================

shap.summary_plot(
    shap_values,
    X_test_sample,
    feature_names=feature_names,
    show=False
)
plt.title("SHAP Summary – RF Referral Model")
plt.tight_layout()
plt.show()

# ==========================
# 2.2 SHAP FORCE (LOCAL)
# ==========================

idx_pos = np.where(y_test.values == 1)[0][0]
idx_neg = np.where(y_test.values == 0)[0][0]

shap.force_plot(
    explainer.expected_value,
    explainer.shap_values(X_test_proc[idx_pos:idx_pos+1]),
    feature_names=feature_names,
    matplotlib=True
)

shap.force_plot(
    explainer.expected_value,
    explainer.shap_values(X_test_proc[idx_neg:idx_neg+1]),
    feature_names=feature_names,
    matplotlib=True
)

# ============================================================
# 3. UMAP — SAME AS SVM
# ============================================================

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=25,
    min_dist=0.1,
    random_state=42
)

X_umap_train = umap_model.fit_transform(X_train_proc)
X_umap_test  = umap_model.transform(X_test_proc)

# ==========================
# 3.1 UMAP — GROUND TRUTH
# ==========================

plt.figure(figsize=(8, 6))
plt.scatter(
    X_umap_train[:, 0],
    X_umap_train[:, 1],
    c=y_train,
    cmap="coolwarm",
    alpha=0.6,
    s=10
)
plt.title("UMAP Projection (Train) – Referral Ground Truth (RF)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(label="Referral")
plt.show()

# ==========================
# 3.2 UMAP — RF PREDICTION
# ==========================

rf_probs_test = rf_pipeline.predict_proba(X_test)[:, 1]
y_pred_umap = (rf_probs_test >= best_threshold).astype(int)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_umap_test[:, 0],
    X_umap_test[:, 1],
    c=y_pred_umap,
    cmap="coolwarm",
    alpha=0.7,
    s=15
)
plt.title("UMAP Projection (Test) – RF Prediction")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(label="Predicted Referral")
plt.show()

# ==========================
# 3.3 HIGH-RISK DENSITY
# ==========================

plt.figure(figsize=(8, 6))
sns.kdeplot(
    x=X_umap_train[y_train == 1][:, 0],
    y=X_umap_train[y_train == 1][:, 1],
    fill=True,
    cmap="Reds",
    levels=30,
    alpha=0.7
)
plt.scatter(
    X_umap_train[:, 0],
    X_umap_train[:, 1],
    c="gray",
    s=5,
    alpha=0.2
)
plt.title("UMAP High-Risk Density – RF")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()
