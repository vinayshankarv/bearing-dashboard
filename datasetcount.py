import os

# ================================
# CHANGE THIS PATH
# ================================
base_dir = r"D:/SET2_EXPANDED_PROJECT/data_augmented"

splits = ["train", "valid", "test"]
classes = ["Bad", "Good"]

total_images = 0

print("\n===== DATASET DISTRIBUTION =====\n")

for split in splits:
    split_path = os.path.join(base_dir, split)

    print(f"--- {split.upper()} ---")

    split_total = 0

    for cls in classes:
        class_path = os.path.join(split_path, cls)

        if not os.path.exists(class_path):
            print(f"{cls}: Folder not found")
            continue

        count = len(os.listdir(class_path))
        print(f"{cls}: {count}")

        split_total += count

    print(f"Total ({split}): {split_total}\n")

    total_images += split_total

print("===== OVERALL DATASET SIZE =====")
print(f"Total Images: {total_images}")