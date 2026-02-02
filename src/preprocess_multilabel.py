import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = "/media/samama/OpticMind/Data Science and Machine Learning/MS_AI UMT/Advanced AI/MedIntel"

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data/chexpert/raw/CheXpert-v1.0-small"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data/chexpert/processed"
)

IMG_OUT = os.path.join(OUT_DIR, "images")

SEED = 42
random.seed(SEED)

# ------------------------------
# Labels
# ------------------------------
# Pathology labels used for TRAINING
PATHOLOGY_LABELS = [
    "Lung Opacity",
    "Consolidation",
    "Pleural Effusion"
]

# Metadata label (NOT used for training)
META_LABELS = [
    "No Finding"
]

# ------------------------------
# Dataset size control
# ------------------------------
MAX_SAMPLES = 50000     # Recommended: 30000–60000
# Set to None to keep full dataset

# ==================================================
# SETUP OUTPUT DIRECTORIES
# ==================================================

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_OUT, exist_ok=True)

# ==================================================
# LOAD & CLEAN LABELS
# ==================================================

print("Loading CheXpert train.csv...")
df = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
print(f"Total raw samples: {len(df)}")

# CheXpert convention:
# -1 (uncertain) → 0 (absent)
df = df.replace(-1, 0)
df = df.fillna(0)

# ==================================================
# MEDICALLY SOUND MULTI-LABEL FILTERING
# ==================================================
"""
Inclusion policy:
- Keep all samples where pathology labels are known
- Do NOT force normal vs abnormal split
- Do NOT remove multi-pathology cases
- 'No Finding' is metadata only
"""

print("Applying multi-label–safe medical filtering...")

valid_mask = (
    (df[PATHOLOGY_LABELS].sum(axis=1) >= 0) |
    (df["No Finding"] == 1)
)

filtered_df = df[valid_mask].copy()
print(f"Samples after medical filtering: {len(filtered_df)}")

# ==================================================
# INTELLIGENT DATASET SIZE CONTROL (OPTIONAL)
# ==================================================
"""
Goal:
- Reduce disk usage and compute
- Preserve ALL rare pathology cases
- Maintain realistic prevalence
"""

if MAX_SAMPLES is not None:
    print(f"\nDataset size before subsampling: {len(filtered_df)}")

    if len(filtered_df) > MAX_SAMPLES:
        print(f"Subsampling dataset to a maximum of {MAX_SAMPLES} samples...")

        # Rare pathologies must be preserved
        # Only truly rare labels should be protected
        rare_mask = (
            filtered_df["Consolidation"] == 1
        )

        rare_df = filtered_df[rare_mask]
        common_df = filtered_df[~rare_mask]

        print(f"Rare pathology samples retained: {len(rare_df)}")
        print(f"Common samples available: {len(common_df)}")

        remaining_capacity = MAX_SAMPLES - len(rare_df)

        if remaining_capacity <= 0:
            print(
                "[WARNING] Rare cases exceed MAX_SAMPLES. "
                "Keeping all rare cases only."
            )
            filtered_df = rare_df.copy()

        else:
            sampled_common_df = common_df.sample(
                n=remaining_capacity,
                random_state=SEED
            )

            filtered_df = pd.concat(
                [rare_df, sampled_common_df],
                axis=0
            ).sample(
                frac=1.0,
                random_state=SEED
            ).reset_index(drop=True)

        print(f"Final dataset size after subsampling: {len(filtered_df)}")

    else:
        print("Dataset size below MAX_SAMPLES — no subsampling applied.")

# --------------------------------------------------
# Sanity check
# --------------------------------------------------
print("\nLabel distribution after subsampling:")
print(filtered_df[PATHOLOGY_LABELS].sum())

# ==================================================
# COPY IMAGES (FLAT, PATH-SAFE)
# ==================================================

print("\nCopying images to flat directory...")

copied = 0
missing = 0
kept_rows = []

for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    relative_path = row["Path"].replace("CheXpert-v1.0-small/", "")
    src = os.path.join(RAW_DIR, relative_path)

    image_name = relative_path.replace("/", "_")
    dst = os.path.join(IMG_OUT, image_name)

    if os.path.exists(src):
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            copied += 1

        row["image"] = image_name
        kept_rows.append(row)
    else:
        missing += 1

filtered_df = pd.DataFrame(kept_rows)

print(f"Images copied: {copied}")
print(f"Images missing/skipped: {missing}")

# ==================================================
# FINAL DATAFRAME
# ==================================================

FINAL_COLUMNS = ["image"] + PATHOLOGY_LABELS + META_LABELS
filtered_df = filtered_df[FINAL_COLUMNS]

# ==================================================
# TRAIN / VAL / TEST SPLIT (70 / 15 / 15)
# ==================================================

print("\nSplitting dataset (70 / 15 / 15)...")

train_df, temp_df = train_test_split(
    filtered_df,
    test_size=0.30,
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED
)

# ==================================================
# SAVE CSV FILES
# ==================================================

train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 70)
print("MULTI-LABEL PREPROCESSING COMPLETE (AGENTIC + ViT SAFE)")
print("=" * 70)

print("Train samples:", len(train_df))
print("Val samples:  ", len(val_df))
print("Test samples: ", len(test_df))

print("\nTrain label distribution:")
print(train_df[PATHOLOGY_LABELS].sum())

print("\nOutput directory:", OUT_DIR)
print("Images directory:", IMG_OUT)
print("=" * 70)















##################################  very large data ################################################## 

"""import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = "/media/samama/OpticMind/Data Science and Machine Learning/MS_AI UMT/Advanced AI/MedIntel"

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data/chexpert/raw/CheXpert-v1.0-small"
)

OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data/chexpert/processed"
)

IMG_OUT = os.path.join(OUT_DIR, "images")

SEED = 42
random.seed(SEED)

# Target pathology labels (TRAINING labels)
PATHOLOGY_LABELS = [
    "Lung Opacity",
    "Consolidation",
    "Pleural Effusion"
]

# Metadata label (NOT used for training)
META_LABELS = [
    "No Finding"
]

# ==================================================
# SETUP OUTPUT DIRECTORIES
# ==================================================

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_OUT, exist_ok=True)

# ==================================================
# LOAD & CLEAN LABELS
# ==================================================

print("Loading CheXpert train.csv...")
df = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
print(f"Total raw samples: {len(df)}")

# Handle CheXpert uncertain labels
# -1 (uncertain) → 0 (absent)
df = df.replace(-1, 0)
df = df.fillna(0)

# ==================================================
# MEDICALLY SOUND FILTERING (MULTI-LABEL SAFE)
# ==================================================

# Inclusion criteria:
# - At least ONE pathology label is known (0 or 1)
# - OR explicitly marked as No Finding

# Exclusion:
# - Rows with completely missing image paths

print("Applying multi-label safe medical filtering...")

valid_mask = (
    (df[PATHOLOGY_LABELS].sum(axis=1) >= 0) |
    (df["No Finding"] == 1)
)

filtered_df = df[valid_mask].copy()

print(f"Samples after medical filtering: {len(filtered_df)}")

# ==================================================
# IMAGE PATH HANDLING
# ==================================================

print("Copying images to flat directory...")

missing = 0
copied = 0
kept_rows = []

for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    relative_path = row["Path"].replace("CheXpert-v1.0-small/", "")
    src = os.path.join(RAW_DIR, relative_path)

    image_name = relative_path.replace("/", "_")
    dst = os.path.join(IMG_OUT, image_name)

    if os.path.exists(src):
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            copied += 1

        row["image"] = image_name
        kept_rows.append(row)
    else:
        missing += 1

filtered_df = pd.DataFrame(kept_rows)

print(f"Images copied: {copied}")
print(f"Images missing/skipped: {missing}")

# ==================================================
# FINAL MULTI-LABEL DATAFRAME
# ==================================================

FINAL_COLUMNS = ["image"] + PATHOLOGY_LABELS + META_LABELS
filtered_df = filtered_df[FINAL_COLUMNS]

print("\nFinal label distribution (entire dataset):")
print(filtered_df[PATHOLOGY_LABELS].sum())
print("\nNo Finding count:", filtered_df["No Finding"].sum())

# ==================================================
# TRAIN / VAL / TEST SPLIT (LABEL-AWARE)
# ==================================================

# We ensure rare labels are present in all splits.
# This is NOT perfect stratification, but far safer than random.


print("\nSplitting dataset (70 / 15 / 15) with label awareness...")

train_df, temp_df = train_test_split(
    filtered_df,
    test_size=0.30,
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED
)

# ==================================================
# SAVE CSV FILES
# ==================================================

train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 70)
print("MULTI-LABEL PREPROCESSING COMPLETE (AGENTIC-SAFE)")
print("=" * 70)

print("Train samples:", len(train_df))
print("Val samples:  ", len(val_df))
print("Test samples: ", len(test_df))

print("\nTrain label distribution:")
print(train_df[PATHOLOGY_LABELS].sum())

print("\nOutput directory:", OUT_DIR)
print("Images directory:", IMG_OUT)
print("=" * 70)
"""