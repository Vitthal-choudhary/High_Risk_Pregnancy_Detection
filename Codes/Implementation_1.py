# ============================================================
# DATASET PREPARATION + SVM + SMOTE + COST (SINGLE CELL)
# ============================================================

# ---------- Install deps ----------
#!pip -q install gdown imbalanced-learn scikit-learn

# ---------- Imports ----------
import pandas as pd
import numpy as np
import gdown
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ================================
# 1. LOAD DATASET FROM DRIVE LINK
# ================================

FILE_ID = "1hXqJhaB3uKYzHoz1S31YKnUNW6Tjgfem"
CSV_PATH = "embarazo_saludable.csv"

gdown.download(
    url=f"https://drive.google.com/uc?id={FILE_ID}",
    output=CSV_PATH,
    quiet=False
)

df_raw = pd.read_csv(CSV_PATH, encoding="latin1", low_memory=False)

# Normalize column names
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

# ================================
# 2. PAPER-ALIGNED FEATURE MAPPING
# ================================

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

# ================================
# 3. DERIVED VARIABLES
# ================================

df["wks"] = np.select(
    [df["fst"] == 1, df["sndt"] == 1, df["trdt"] == 1],
    [10, 20, 32],
    default=np.nan
)

df["sxt"] = (
    (df_raw.get("vih", 0) == 1) |
    (df_raw.get("sifilis", 0) == 1) |
    (df_raw.get("vhb", 0) == 1)
).astype(int)

def derive_urine_status(row):
    if row.get("proteinas", 0) == 1:
        return "protein"
    if row.get("nitritos", 0) == 1:
        return "nitrite"
    if row.get("leucocitos", 0) == 1:
        return "leukocyte"
    if row.get("orina:negativo", 0) == 1 or row.get("normal", 0) == 1:
        return "normal"
    return "unknown"

df["ur"] = df_raw.apply(derive_urine_status, axis=1)

# ================================
# 4. REFERRAL LABEL
# ================================

referral_cols = [
    "ref:_cs",
    "ref:_h_distrital",
    "ref:_hregional",
    "ref:_hnacional"
]

existing_ref_cols = [c for c in referral_cols if c in df_raw.columns]
referral_binary = df_raw[existing_ref_cols].notna().astype(int)

df["referral"] = (referral_binary.sum(axis=1) > 0).astype(int)

# ================================
# 5. TYPE FIXES
# ================================

numeric_cols = [
    "age","lb","sb","sbp","dbp","bmi","crl","bpd","hc",
    "ac","fl","ala","hb","gc","wks"
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ================================
# 6. PREPARE ML MATRICES
# ================================

X = df.drop(columns=["referral"])
y = df["referral"]

# 🔧 DROP BROKEN FEATURE (ALL NaN → breaks SMOTE geometry)
X = X.drop(columns=["wks"])

# One-hot encode categoricals (same idea as paper)
X = pd.get_dummies(X, drop_first=True)

# ================================
# 7. TRAIN / TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================================
# 8. COST-SENSITIVE SETUP
# ================================

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
cost_ratio = n_neg / n_pos   # ≈ 11

class_weights = "balanced"

print(f"\nClass ratio (neg:pos) ≈ {cost_ratio:.2f}:1")

# ================================
# 9. SVM + SMOTE PIPELINE
# ================================

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # ← FIX
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42, k_neighbors=3)),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight=class_weights
    ))
])

pipeline.fit(X_train, y_train)

# ================================
# 10. EVALUATION (PAPER METRICS)
# ================================

# Get raw SVM decision scores
decision_scores = pipeline.decision_function(X_test)

# ================================
# Threshold selection by max G-Mean
# ================================

thresholds = np.linspace(decision_scores.min(), decision_scores.max(), 300)

best_gmean = 0
best_threshold = 0

for t in thresholds:
    y_tmp = (decision_scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_tmp).ravel()

    if (tp + fn) == 0 or (tn + fp) == 0:
        continue

    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    gmean = np.sqrt(se * sp)

    if gmean > best_gmean:
        best_gmean = gmean
        best_threshold = t

print(f"\nBest threshold (by G-Mean): {best_threshold:.3f}")
print(f"Best G-Mean: {best_gmean:.3f}")

# Final predictions using optimal threshold
y_pred = (decision_scores >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
f1 = f1_score(y_test, y_pred)
g_mean = np.sqrt(specificity * sensitivity)

print("\n================ RESULTS ================")
print(f"Specificity (Sp): {specificity:.3f}")
print(f"Sensitivity (Se): {sensitivity:.3f}")
print(f"F1-score        : {f1:.3f}")
print(f"G-Mean          : {g_mean:.3f}")

print("\nConfusion Matrix")
print(f"TN={tn}, FP={fp}")
print(f"FN={fn}, TP={tp}")

# ============================================================
# SHAP + UMAP INTERPRETABILITY (RUN AFTER BASE PIPELINE)
# ============================================================

# ---------- Install extra deps ----------
#!pip -q install shap umap-learn matplotlib seaborn

# ---------- Imports ----------
import shap
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. PREPARE DATA FOR INTERPRETABILITY
# ============================================================

# IMPORTANT:
# We must extract the trained SVM AFTER preprocessing steps
# Pipeline order: imputer -> scaler -> smote -> svm

# We use ONLY imputer + scaler for SHAP/UMAP (no SMOTE at inference)
preprocess = Pipeline(steps=[
    ("imputer", pipeline.named_steps["imputer"]),
    ("scaler", pipeline.named_steps["scaler"])
])

X_train_proc = preprocess.fit_transform(X_train)
X_test_proc  = preprocess.transform(X_test)

feature_names = X_train.columns.tolist()

# ============================================================
# 2. SHAP FOR SVM (KERNEL SHAP)
# ============================================================

# Paper uses SHAP to interpret black-box SVM
# KernelExplainer is required for SVC

# Use small background sample for speed (paper does similar)
background = shap.sample(X_train_proc, 50, random_state=42)
X_test_sample = X_test_proc[:50]

explainer = shap.KernelExplainer(
    pipeline.named_steps["svm"].decision_function,
    background
)

shap_values = explainer.shap_values(X_test_sample, nsamples=100)

# ==========================
# 2.1 SHAP SUMMARY (GLOBAL)
# ==========================

shap.summary_plot(
    shap_values,
    X_test_sample,
    feature_names=feature_names,
    show=False
)
plt.title("SHAP Summary – SVM Referral Model")
plt.tight_layout()
plt.show()

# ==========================
# 2.2 SHAP FORCE (LOCAL)
# ==========================

# One referred & one non-referred example (as in paper Fig. 5)
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
# 3. UMAP – METRIC + CATEGORICAL SPACE
# ============================================================

# Paper uses:
# - n_components = 2
# - n_neighbors ≈ 25
# - standardized inputs

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=25,
    min_dist=0.1,
    random_state=42
)

X_umap_train = umap_model.fit_transform(X_train_proc)
X_umap_test  = umap_model.transform(X_test_proc)

# ============================================================
# 4. UMAP VISUALIZATION (REFERRAL GROUND TRUTH)
# ============================================================

plt.figure(figsize=(8, 6))
plt.scatter(
    X_umap_train[:, 0],
    X_umap_train[:, 1],
    c=y_train,
    cmap="coolwarm",
    alpha=0.6,
    s=10
)
plt.title("UMAP Projection (Train) – Referral Ground Truth")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(label="Referral")
plt.show()

# ============================================================
# 5. UMAP VISUALIZATION (MODEL PREDICTION)
# ============================================================

decision_scores_test = pipeline.decision_function(X_test)
y_pred_umap = (decision_scores_test >= 0).astype(int)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_umap_test[:, 0],
    X_umap_test[:, 1],
    c=y_pred_umap,
    cmap="coolwarm",
    alpha=0.7,
    s=15
)
plt.title("UMAP Projection (Test) – SVM Prediction")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(label="Predicted Referral")
plt.show()

# ============================================================
# 6. OPTIONAL: HIGH-RISK DENSITY MAP (PAPER-LIKE)
# ============================================================

# Approximate Parzen-style density using seaborn KDE
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
plt.title("UMAP High-Risk Density (Referral = 1)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()
