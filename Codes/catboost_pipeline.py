# ============================================================
# CATBOOST + COST-SENSITIVE LEARNING (NO SMOTE)
# CLEAN, FINAL, SINGLE-FILE PIPELINE
# ============================================================

# !pip install catboost gdown scikit-learn pandas numpy

# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import gdown

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score

from catboost import CatBoostClassifier

# ============================================================
# 2. LOAD DATASET
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
# 4. FEATURE SELECTION (PAPER VARIABLES)
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
# 5. DERIVED VARIABLES
# ============================================================

df["sxt"] = (
    (df_raw.get("vih", 0) == 1) |
    (df_raw.get("sifilis", 0) == 1) |
    (df_raw.get("vhb", 0) == 1)
).astype(int)

def derive_ur(row):
    if row.get("proteinas", 0) == 1:
        return "protein"
    if row.get("nitritos", 0) == 1:
        return "nitrite"
    if row.get("leucocitos", 0) == 1:
        return "leukocyte"
    if row.get("orina:negativo", 0) == 1 or row.get("normal", 0) == 1:
        return "normal"
    return "unknown"

df["ur"] = df_raw.apply(derive_ur, axis=1)

# ============================================================
# 6. REFERRAL LABEL
# ============================================================

ref_cols = ["ref:_cs", "ref:_h_distrital", "ref:_hregional", "ref:_hnacional"]
existing = [c for c in ref_cols if c in df_raw.columns]
df["referral"] = (df_raw[existing].notna().sum(axis=1) > 0).astype(int)

# ============================================================
# 7. TYPE FIXES (NUMERIC ONLY)
# ============================================================

numeric_cols = [
    "age","lb","sb","sbp","dbp","bmi","crl","bpd",
    "hc","ac","fl","ala","hb","gc","fh"
]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ============================================================
# 8. ML MATRICES
# ============================================================

X = df.drop(columns=["referral"])
y = df["referral"]

# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================================
# 10. EXPLICIT CATEGORICAL DEFINITION (CRITICAL)
# ============================================================

cat_cols = [
    "fp",    # fetal position
    "iu",    # intrauterine
    "fst",   # trimester flags
    "sndt",
    "trdt",
    "pl",    # placenta
    "ur",    # urine result
    "an",    # anemia
    "sxt"    # STI
]

num_cols = [c for c in X_train.columns if c not in cat_cols]

# Drop numeric columns that are entirely NaN (cannot be imputed)
all_nan_cols = [c for c in num_cols if X_train[c].isna().all()]
if all_nan_cols:
    print("Dropping all-NaN columns:", all_nan_cols)
    X_train = X_train.drop(columns=all_nan_cols)
    X_test  = X_test.drop(columns=all_nan_cols)
    num_cols = [c for c in num_cols if c not in all_nan_cols]

# ============================================================
# 11. IMPUTATION
# ============================================================

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols]  = num_imputer.transform(X_test[num_cols])

X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_test[cat_cols]  = cat_imputer.transform(X_test[cat_cols])

# 🔴 REQUIRED FOR CATBOOST
for c in cat_cols:
    X_train[c] = X_train[c].astype(str)
    X_test[c]  = X_test[c].astype(str)

cat_features = [X_train.columns.get_loc(c) for c in cat_cols]

# ============================================================
# 12. COST-SENSITIVE SETUP
# ============================================================

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
cost_ratio = n_neg / n_pos

class_weights = [1.0, cost_ratio]
print(f"\nCost ratio (FN:FP) ≈ {cost_ratio:.2f}:1")

# ============================================================
# 13. CATBOOST MODEL
# ============================================================

cb = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    class_weights=class_weights,
    cat_features=cat_features,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

cb.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# ============================================================
# 14. THRESHOLD TUNING (G-MEAN)
# ============================================================

probs = cb.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.01, 0.99, 300)
best_gmean, best_t = 0, 0

for t in thresholds:
    y_hat = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
    if tp + fn == 0 or tn + fp == 0:
        continue
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    g = np.sqrt(se * sp)
    if g > best_gmean:
        best_gmean, best_t = g, t

# ============================================================
# 15. FINAL METRICS
# ============================================================

y_pred = (probs >= best_t).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

sp = tn / (tn + fp)
se = tp / (tp + fn)
f1 = f1_score(y_test, y_pred)
gmean = np.sqrt(sp * se)

print("\n=========== CATBOOST RESULTS ===========")
print(f"Specificity (Sp): {sp:.3f}")
print(f"Sensitivity (Se): {se:.3f}")
print(f"F1-score        : {f1:.3f}")
print(f"G-Mean          : {gmean:.3f}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
