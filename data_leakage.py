# ==========================================
# DATA LEAKAGE CHECK SCRIPT
# ==========================================

import os
from torchvision import datasets

# ==========================================
# PATHS (EDIT IF NEEDED)
# ==========================================
train_dir = r"D:/SET2_EXPANDED_PROJECT/data_augmented/train"
val_dir   = r"D:/SET2_EXPANDED_PROJECT/data_augmented/valid"
test_dir  = r"D:/SET2_EXPANDED_PROJECT/data_augmented/test"

# ==========================================
# LOAD DATASETS (NO TRANSFORMS NEEDED)
# ==========================================
train_dataset = datasets.ImageFolder(train_dir)
val_dataset   = datasets.ImageFolder(val_dir)
test_dataset  = datasets.ImageFolder(test_dir)

# ==========================================
# BASIC INFO
# ==========================================
print("\n===== DATASET INFO =====")
print("Train size:", len(train_dataset))
print("Val size  :", len(val_dataset))
print("Test size :", len(test_dataset))

print("\nClasses:", train_dataset.classes)

# ==========================================
# GET FILE PATHS
# ==========================================
train_files = set([os.path.abspath(x[0]) for x in train_dataset.samples])
val_files   = set([os.path.abspath(x[0]) for x in val_dataset.samples])
test_files  = set([os.path.abspath(x[0]) for x in test_dataset.samples])

# ==========================================
# CHECK OVERLAPS
# ==========================================
train_val_overlap = train_files.intersection(val_files)
train_test_overlap = train_files.intersection(test_files)
val_test_overlap = val_files.intersection(test_files)

print("\n===== DATA LEAKAGE CHECK =====")
print("Train-Val overlap :", len(train_val_overlap))
print("Train-Test overlap:", len(train_test_overlap))
print("Val-Test overlap  :", len(val_test_overlap))

# ==========================================
# OPTIONAL: PRINT SAMPLE DUPLICATES
# ==========================================
def show_examples(overlap_set, name):
    if len(overlap_set) > 0:
        print(f"\nSample duplicates in {name}:")
        for i, file in enumerate(list(overlap_set)[:5]):
            print(file)

show_examples(train_val_overlap, "Train-Val")
show_examples(train_test_overlap, "Train-Test")
show_examples(val_test_overlap, "Val-Test")

# ==========================================
# FINAL VERDICT
# ==========================================
if (len(train_val_overlap) == 0 and
    len(train_test_overlap) == 0 and
    len(val_test_overlap) == 0):
    
    print("\n✅ No data leakage detected — your splits are clean.")
else:
    print("\n❌ Data leakage detected — fix dataset splits immediately!")