import pandas as pd
import numpy as np
import gdown

# ================================
# 1. LOAD DATASET FROM DRIVE LINK
# ================================

# Google Drive file ID
FILE_ID = "1hXqJhaB3uKYzHoz1S31YKnUNW6Tjgfem"
CSV_PATH = "embarazo_saludable.csv"

# Download file
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
    # Metric variables
    "edad": "age",
    "hijos_nacidos_vivos": "lb",
    "hijos_nacidos_muertos": "sb",
    "pas": "sbp",
    "pad": "dbp",
    "imc": "bmi",
    "crl": "crl",
    "dbp": "bpd",          # fetal DBP
    "cc": "hc",
    "ca": "ac",
    "lf": "fl",
    "ila": "ala",
    "hb": "hb",
    "glucosa": "gc",

    # Categorical / binary
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
# 3. DERIVED VARIABLES (paper-style)
# ================================

# Weeks of gestation (from trimesters)
df["wks"] = np.select(
    [
        df["fst"] == 1,
        df["sndt"] == 1,
        df["trdt"] == 1
    ],
    [
        10,   # representative first trimester
        20,
        32
    ],
    default=np.nan
)

# Sexually transmitted infections (merged)
df["sxt"] = (
    (df_raw.get("vih", 0) == 1) |
    (df_raw.get("sifilis", 0) == 1) |
    (df_raw.get("vhb", 0) == 1)
).astype(int)

# Urine test (simplified categorical)
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
# 4. REFERRAL LABEL (ground truth)
# ================================
referral_cols = [
    "ref:_cs",
    "ref:_h_distrital",
    "ref:_hregional",
    "ref:_hnacional"
]

# Ensure columns exist (defensive programming)
existing_ref_cols = [c for c in referral_cols if c in df_raw.columns]

if len(existing_ref_cols) == 0:
    raise ValueError("No referral columns found — check column normalization")

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
# 6. FINAL CHECK
# ================================
print("Final shape:", df.shape)
print("Final columns:", df.columns.tolist())
print(df.head())

print("\n✅ 26-variable dataset aligned with paper, ready for SVM + SMOTE + costs") 
print(df["referral"].value_counts(normalize=True))
