import pandas as pd

# ================================
# KONFIGURASI
# ================================
INPUT_CSV = "test1.csv"          # CSV kamu (882 kolom)
OUTPUT_CSV = "test1_fixed.csv"   # CSV hasil (462 kolom)

N_CHANNELS = 14

STAT_FEATURES = [
    'min', 'max', 'ar1', 'ar2', 'ar3', 'ar4', 'md', 'var', 'sd',
    'am', 're', 'le', 'sh', 'te', 'lrssv', 'mte', 'me', 'mcl',
    'n2d', '2d', 'n1d', '1d', 'kurt', 'skew', 'hc', 'hm', 'ha',
    'bpd', 'bpt', 'bpa', 'bpb', 'bpg', 'rba'
]

# ================================
# GENERATE NAMA KOLOM YANG DIPAKAI MODEL
# ================================
EXPECTED_COLUMNS = [
    f"{stat}_{ch}"
    for ch in range(1, N_CHANNELS + 1)
    for stat in STAT_FEATURES
]

# ================================
# LOAD CSV ASLI
# ================================
df = pd.read_csv(INPUT_CSV)

print("CSV ASLI SHAPE :", df.shape)

# ================================
# CEK KOLOM YANG ADA
# ================================
missing = set(EXPECTED_COLUMNS) - set(df.columns)
if missing:
    raise ValueError(f"Kolom hilang di CSV: {list(missing)[:5]} ...")

# ================================
# AMBIL & URUTKAN KOLOM YANG BENAR
# ================================
df_fixed = df[EXPECTED_COLUMNS]

print("CSV FIXED SHAPE:", df_fixed.shape)

# ================================
# SIMPAN CSV BARU
# ================================
df_fixed.to_csv(OUTPUT_CSV, index=False)

print(f"✅ CSV siap inferensi: {OUTPUT_CSV}")
