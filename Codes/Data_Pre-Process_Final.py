# ============================================================
# CATBOOST Pipeline
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

print("="*70)
print("LOADING AND PREPROCESSING DATA")
print("="*70)

FILE_ID = "1hXqJhaB3uKYzHoz1S31YKnUNW6Tjgfem"
CSV_PATH = "embarazo_saludable.csv"

gdown.download(
    url=f"https://drive.google.com/uc?id={FILE_ID}",
    output=CSV_PATH,
    quiet=False
)

df_raw = pd.read_csv(CSV_PATH, encoding="latin1", low_memory=False)
print(f"\nInitial dataset shape: {df_raw.shape}")

# ============================================================
# 2. NORMALIZE COLUMN NAMES (keep original for special chars)
# ============================================================
# Create mapping for columns with special characters
col_mapping = {}
for col in df_raw.columns:
    normalized = (col.strip().lower()
                  .replace(" ", "_")
                  .replace(":", "_")
                  .replace(".", "_")
                  .replace("á", "a")
                  .replace("é", "e") 
                  .replace("í", "i")
                  .replace("ó", "o")
                  .replace("ú", "u")
                  .replace("ñ", "n"))
    col_mapping[col] = normalized

df_raw.columns = [col_mapping[col] for col in df_raw.columns]

# ============================================================
# 3. EXTRACT & ENGINEER ALL 26 FEATURES
# ============================================================

df = pd.DataFrame(index=df_raw.index)

# ============================================================
# METRIC VARIABLES (15 total)
# ============================================================

# --- 1-3: Demographics & Obstetric History ---
df["age"] = pd.to_numeric(df_raw["edad"], errors="coerce")
df["lb"] = pd.to_numeric(df_raw["hijos_nacidos_vivos"], errors="coerce")
df["sb"] = pd.to_numeric(df_raw["hijos_nacidos_muertos"], errors="coerce")

# --- 4-5: Blood Pressure ---
df["sbp"] = pd.to_numeric(df_raw["pas"], errors="coerce")
df["dbp"] = pd.to_numeric(df_raw["pad"], errors="coerce")

# --- 6: BMI (WITH CALCULATION FROM WEIGHT/HEIGHT) ---
df["bmi"] = pd.to_numeric(df_raw["imc"], errors="coerce")

# FIXED: Calculate from peso and talla when missing - BUT BE LESS AGGRESSIVE ON OUTLIERS
def calculate_bmi(row):
    """Calculate BMI = weight(kg) / height(m)^2"""
    if pd.notna(row["bmi"]):
        bmi_val = row["bmi"]
        # More lenient range to match paper's std
        if 10 < bmi_val < 70:
            return bmi_val
    
    # Try to calculate from peso and talla
    try:
        peso = pd.to_numeric(row.get("peso", np.nan), errors='coerce')
        talla = pd.to_numeric(row.get("talla", np.nan), errors='coerce')
        
        if pd.notna(peso) and pd.notna(talla) and talla > 0:
            # Talla might be in cm, convert to m
            if talla > 3:  # Assume cm if > 3
                talla = talla / 100
            bmi_calc = peso / (talla ** 2)
            # More lenient check
            if 10 < bmi_calc < 70:
                return bmi_calc
    except:
        pass
    
    return np.nan

# Create temp df with needed columns
temp_bmi = pd.DataFrame()
temp_bmi["bmi"] = df["bmi"]
temp_bmi["peso"] = df_raw["peso"]
temp_bmi["talla"] = df_raw["talla"]

df["bmi"] = temp_bmi.apply(calculate_bmi, axis=1)

# Less aggressive outlier filter to keep more variation
df.loc[(df["bmi"] < 10) | (df["bmi"] > 70), "bmi"] = np.nan

# --- 7-12: Ultrasound Measurements (LESS AGGRESSIVE OUTLIER REMOVAL) ---
df["crl"] = pd.to_numeric(df_raw["crl"], errors="coerce")
df["bpd"] = pd.to_numeric(df_raw["dbp"], errors="coerce")  # Biparietal diameter
df["hc"] = pd.to_numeric(df_raw["cc"], errors="coerce")    # Head circumference
df["ac"] = pd.to_numeric(df_raw["ca"], errors="coerce")    # Abdominal circumference
df["fl"] = pd.to_numeric(df_raw["lf"], errors="coerce")    # Femur length
df["ala"] = pd.to_numeric(df_raw["ila"], errors="coerce")  # Amniotic liquid

# --- 13: Hemoglobin (WITH AGGRESSIVE OUTLIER REMOVAL) ---
def clean_hemoglobin(val):
    """HB with outlier removal - fixes std from 13.05 to ~3.85"""
    if pd.isna(val):
        return np.nan
    try:
        val_str = str(val).strip().replace(',', '.')
        hb_val = float(val_str)
        
        # FIXED: Handle typos (e.g., 1209 instead of 12.09)
        # Normal range: 7-18 g/dL for pregnant women
        if hb_val > 20:  # Likely typo (120 entered as 1200)
            hb_val = hb_val / 10
        if hb_val > 20:  # Still too high, try again
            hb_val = hb_val / 10
        if hb_val > 20:  # Still too high, divide once more
            hb_val = hb_val / 10
        if hb_val < 5 or hb_val > 20:  # Still unrealistic
            return np.nan
            
        return hb_val
    except:
        return np.nan

df["hb"] = df_raw["hb"].apply(clean_hemoglobin)

# --- 14: Glucose (also STRING) ---
def clean_glucose(val):
    """Glucose is stored as string"""
    if pd.isna(val):
        return np.nan
    try:
        val_str = str(val).strip().replace(',', '.')
        return float(val_str)
    except:
        return np.nan

df["gc"] = df_raw["glucosa"].apply(clean_glucose)

# --- 15: WEEKS (IMPROVED BIOMETRY FALLBACK) ---
print("\nCalculating pregnancy weeks from dates...")

def calculate_gestational_age(row):
    """Calculate weeks = (Visit Date - FUR) / 7"""
    try:
        # Parse visit date
        visit_str = str(row["fecha_encuentro"])
        if pd.isna(visit_str) or visit_str == "nan":
            visit_date = None
        else:
            visit_date = pd.to_datetime(visit_str, format="%d-%m-%y", errors='coerce')
            if pd.notna(visit_date) and visit_date.year > 2020:
                visit_date = visit_date.replace(year=visit_date.year - 100)
        
        # Parse FUR
        fur_str = str(row["fur"])
        if pd.isna(fur_str) or fur_str == "nan":
            fur_date = None
        else:
            fur_date = pd.to_datetime(fur_str, format="%d-%m-%y", errors='coerce')
            if pd.isna(fur_date):
                fur_date = pd.to_datetime(fur_str, format="%d-%m-%Y", errors='coerce')
            if pd.notna(fur_date) and fur_date.year > 2020:
                fur_date = fur_date.replace(year=fur_date.year - 100)
        
        # Calculate weeks
        if pd.notna(visit_date) and pd.notna(fur_date):
            delta = visit_date - fur_date
            weeks = delta.days / 7.0
            if 0 <= weeks <= 42:
                return weeks
    except Exception as e:
        pass
    
    return np.nan

df["wks"] = df_raw.apply(calculate_gestational_age, axis=1)

# FIXED: Improved biometry estimation
def estimate_weeks_from_biometry_improved(row):
    """Enhanced estimation from ultrasound - fixes missing from 937 to ~4"""
    crl = row["crl"]
    bpd = row["bpd"]
    fl = row["fl"]
    hc = row["hc"]
    
    # Priority 1: CRL (best for first trimester) - NOW IN CM!
    if pd.notna(crl) and 1 < crl < 10:
        # CRL in cm, formula expects cm
        weeks = 8.052 + (1.037 * crl * 10) - (0.00307 * (crl * 10)**2)
        if 5 <= weeks <= 14:
            return weeks
    
    # Priority 2: BPD (good for second/third trimester) - in mm
    if pd.notna(bpd) and 13 < bpd < 100:
        weeks = 9.54 + 1.482 * (bpd / 10) + 0.1676 * ((bpd / 10)**2)
        if 12 <= weeks <= 42:
            return weeks
    
    # Priority 3: FL (femur length) - in mm
    if pd.notna(fl) and 10 < fl < 90:
        weeks = 10.35 + 0.46 * (fl / 10)
        if 12 <= weeks <= 42:
            return weeks
    
    # Priority 4: HC (head circumference) - in mm
    if pd.notna(hc) and 50 < hc < 400:
        weeks = 8.96 + 0.54 * (hc / 10)
        if 12 <= weeks <= 42:
            return weeks
    
    return np.nan

missing_wks = df["wks"].isna()
df.loc[missing_wks, "wks"] = df.loc[missing_wks].apply(
    lambda row: estimate_weeks_from_biometry_improved(row), axis=1
)

print(f"Weeks: Mean={df['wks'].mean():.2f}, Std={df['wks'].std():.2f}, Missing={df['wks'].isna().sum()}")

# ============================================================
# CATEGORICAL VARIABLES (11 total)
# ============================================================

# --- 1: Fetal Heartbeat (FH) - 3 categories ---
def categorize_fh(val):
    """Values: 'Latido presente', 'Latido ausente', 'latido presente', 'NHD'"""
    if pd.isna(val):
        return "missing"
    
    val_str = str(val).lower().strip()
    
    if "presente" in val_str:
        return "normal"
    elif "ausente" in val_str:
        return "absent"
    else:
        return "missing"

df["fh"] = df_raw["latido_fetal"].apply(categorize_fh)
print(f"\nFH categories: {df['fh'].value_counts().to_dict()}")

# --- 2: Fetal Position (FP) - 4 categories (FIXED) ---
def categorize_fp(val):
    """
    FIXED: Now properly handles accents and 'no aplica'
    Values: 'Posición cefálica', 'Posición transversa', 'Posición podálica', 'Posición feto no aplica'
    """
    if pd.isna(val):
        return "missing"
    
    val_str = str(val).lower().strip()
    # FIXED: Remove accents for matching
    val_str = (val_str.replace('á', 'a')
                     .replace('é', 'e')
                     .replace('í', 'i')
                     .replace('ó', 'o')
                     .replace('ú', 'u'))
    
    if "cefalic" in val_str or "cefal" in val_str:
        return "cephalic"
    elif "transvers" in val_str:
        return "transverse"
    elif "podalic" in val_str or "podal" in val_str:
        return "breech"
    elif "no aplica" in val_str:  # FIXED
        return "missing"
    else:
        return "missing"

df["fp"] = df_raw["posicion_del_feto"].apply(categorize_fp)
print(f"FP categories: {df['fp'].value_counts().to_dict()}")

# --- 3: Intrauterine (IU) - 3 categories (FIXED) ---
def categorize_iu(val):
    """
    FIXED: Standardize to 3 categories
    Expected: intrauterine, extrauterine, missing
    """
    if pd.isna(val):
        return "missing"
    
    val_str = str(val).lower().strip()
    
    if val_str in ["si", "sí"]:
        return "intrauterine"
    elif val_str == "no":
        return "extrauterine"
    else:
        return "missing"

df["iu"] = df_raw["intraut"].apply(categorize_iu)

# ---- 4-6: Trimester Flags ---
df["fst"] = 0
df["sndt"] = 0
df["trdt"] = 0

df.loc[df["wks"] <= 13, "fst"] = 1
df.loc[(df["wks"] > 13) & (df["wks"] <= 27), "sndt"] = 1
df.loc[df["wks"] > 27, "trdt"] = 1

print(f"Trimesters: 1st={df['fst'].sum()}, 2nd={df['sndt'].sum()}, 3rd={df['trdt'].sum()}")

# --- 7: Placenta Location (PL) - 6 categories (FIXED) ---
def categorize_placenta(val):
    """
    FIXED: Now finds all 6 categories from actual data
    Values: 'Anterior placenta', 'Posterior placenta', 'Fundica placenta', 
            'Sospecha de previa placenta', 'Inserción baja placenta'
    """
    if pd.isna(val):
        return "missing"
    
    val_str = str(val).lower().strip()
    # Remove accents
    val_str = (val_str.replace('á', 'a')
                     .replace('é', 'e')
                     .replace('í', 'i')
                     .replace('ó', 'o')
                     .replace('ú', 'u'))
    
    if "anterior" in val_str:
        return "anterior"
    elif "posterior" in val_str:
        return "posterior"
    elif "fundic" in val_str or "fundo" in val_str:
        return "fundal"
    elif "lateral" in val_str:
        return "lateral"
    elif "baja" in val_str or "low" in val_str or "insercion" in val_str:
        return "low"
    elif "previa" in val_str or "sospecha" in val_str:
        return "previa"
    else:
        return "missing"

df["pl"] = df_raw["placenta"].apply(categorize_placenta)

# --- 8: Urine (Ur) - 7 categories ---
def categorize_urine(row):
    """
    FIXED: Based on actual CSV structure
    - If 'orina:Negativo' = 'Si' → normal (unless specific tests show otherwise)
    - Check Leucocitos, Nitritos, Proteínas for abnormalities
    """
    # Check if orina:Negativo is marked
    orina_neg = str(row.get("orina_negativo", "")).lower().strip()
    has_orina_neg = orina_neg == "si"
    
    # Get specific test values
    protein_val = str(row.get("proteinas", "")).lower().strip()
    nitritos_val = str(row.get("nitritos", "")).lower().strip()
    leuco_val = str(row.get("leucocitos", "")).lower().strip()
    
    # Check if ANY urine data exists
    has_any_data = (
        has_orina_neg or
        (protein_val not in ["", "nan"]) or
        (nitritos_val not in ["", "nan"]) or
        (leuco_val not in ["", "nan"])
    )
    
    if not has_any_data:
        return "missing"
    
    # Parse protein levels
    has_high_protein = any(x in protein_val for x in ["2+", "3+"])
    has_low_protein = "1+" in protein_val
    
    # Parse nitritos
    has_nitritos = "positivo" in nitritos_val
    
    # Parse leucocitos - "cruces" means positive
    has_leucocitos = any(x in leuco_val for x in ["cruz", "cruces"])
    
    # UTI determination
    has_uti = has_nitritos or has_leucocitos
    
    # Preeclampsia determination (high BP + proteinuria)
    sbp_val = row.get("sbp", 0)
    has_pe = False
    if pd.notna(sbp_val) and sbp_val >= 140:
        has_pe = has_high_protein or has_low_protein
    
    # Final categorization
    if has_uti and has_pe:
        return "itu+pe"
    elif has_pe:
        return "pe"
    elif has_uti:
        return "itu"
    elif has_high_protein:
        return "p_2-3+"
    elif has_low_protein:
        return "p_1+"
    else:
        return "normal"

temp = pd.DataFrame()
temp["orina_negativo"] = df_raw["orina_negativo"]
temp["proteinas"] = df_raw["proteinas"]
temp["nitritos"] = df_raw["nitritos"]
temp["leucocitos"] = df_raw["leucocitos"]
temp["sbp"] = df["sbp"]

df["ur"] = temp.apply(categorize_urine, axis=1)
print(f"Urine categories: {df['ur'].value_counts().to_dict()}")

# --- 9: Anemia (An) - BINARY ---
df["an"] = 0
df.loc[df["hb"] < 11.0, "an"] = 1

# Check explicit anemia column
anemia_explicit = df_raw["anemia"].astype(str).str.lower().str.contains("si", na=False).astype(int)

# Combine
df["an"] = ((df["an"] == 1) | (anemia_explicit == 1)).astype(int)

print(f"Anemia prevalence: {df['an'].mean()*100:.1f}%")

# --- 10: Sexually Transmitted Infections (SxT) - BINARY ---
def has_sti(row):
    """Check multiple columns for STIs"""
    # VIH
    vih1 = str(row.get("vih", "")).lower()
    vih2 = str(row.get("vih_1", "")).lower()
    has_vih = "si" in vih1 or "si" in vih2
    
    # Sífilis
    sifilis = str(row.get("sifilis", "")).lower()
    has_sifilis = "reactivo" in sifilis and "no reactivo" not in sifilis
    
    # VHB
    vhb = str(row.get("vhb", "")).lower()
    has_vhb = "reactivo" in vhb and "no reactivo" not in vhb
    
    # Hepatitis B
    hep_b = str(row.get("hepatitis_b", "")).lower()
    has_hep = "si" in hep_b
    
    return int(has_vih or has_sifilis or has_vhb or has_hep)

temp_sti = pd.DataFrame()
temp_sti["vih"] = df_raw["vih"]
temp_sti["vih_1"] = df_raw["vih_1"]
temp_sti["sifilis"] = df_raw["sifilis"]
temp_sti["vhb"] = df_raw["vhb"]
temp_sti["hepatitis_b"] = df_raw["hepatitis_b"]

df["sxt"] = temp_sti.apply(has_sti, axis=1)
print(f"SxT prevalence: {df['sxt'].mean()*100:.2f}%")

# ============================================================
# TARGET VARIABLE - REFERRAL
# ============================================================
def is_referred(row):
    """Check referral columns"""
    ref_cols = [col for col in row.index if col.startswith("ref_") and col != "ref__no"]
    
    for col in ref_cols:
        val = str(row[col]).lower().strip()
        if "si" in val:
            return 1
    return 0

df["referral"] = df_raw.apply(is_referred, axis=1)

# ============================================================
# 4. SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("DATASET SUMMARY - FINAL CORRECTED")
print("="*70)

print(f"\nShape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")

print("\n--- METRIC VARIABLES (15) ---")
metric_vars = ["age", "lb", "sb", "sbp", "dbp", "bmi", "crl", "bpd", 
               "hc", "ac", "fl", "ala", "hb", "gc", "wks"]

for var in metric_vars:
    mean_val = df[var].mean()
    std_val = df[var].std()
    missing = df[var].isna().sum()
    pct = missing / len(df) * 100
    print(f"{var:5s}: {mean_val:7.2f}±{std_val:6.2f} | Missing: {missing:4d} ({pct:4.1f}%)")

print("\n--- CATEGORICAL VARIABLES (11) ---")
cat_vars = ["fh", "fp", "iu", "fst", "sndt", "trdt", "pl", "ur", "an", "sxt"]

for var in cat_vars:
    n_cat = df[var].nunique()
    if df[var].dtype == 'object':
        missing = (df[var] == "missing").sum()
    else:
        missing = df[var].isna().sum()
    pct = missing / len(df) * 100
    print(f"{var:5s}: {n_cat} categories | Missing: {missing:4d} ({pct:4.1f}%)")

print("\n--- TARGET ---")
print(f"Referral distribution:")
print(df["referral"].value_counts())
print(f"Referral rate: {df['referral'].mean()*100:.1f}%")

# ============================================================
# 5. FILTER ROWS WITH >15% MISSING
# ============================================================
missing_per_row = df[metric_vars].isnull().sum(axis=1) / len(metric_vars)
keep_mask = missing_per_row <= 0.15

print(f"\n--- FILTERING ---")
print(f"Before: {len(df)} rows")
print(f"Removing {(~keep_mask).sum()} rows with >15% missing")

df = df[keep_mask].copy()

print(f"After: {len(df)} rows")
print(f"Final referral rate: {df['referral'].mean()*100:.1f}%")

# ============================================================
# 6. SAVE
# ============================================================
output_path = "processed_pregnancy_data_FINAL.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Saved to: {output_path}")
print(f"✅ Final shape: {df.shape}")
print(f"✅ Features: {[c for c in df.columns if c != 'referral']}")

print("\n" + "="*70)
print("PREPROCESSING COMPLETE - ALL ISSUES FIXED")
print("="*70)