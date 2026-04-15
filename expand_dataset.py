import os
import cv2
import random
from tqdm import tqdm
import albumentations as A

input_dir = r"D:/SET2_EXPANDED_PROJECT/data_original/train"
output_dir = r"D:/SET2_EXPANDED_PROJECT/data_augmented/train"

classes = ["Bad", "Good"]

target_per_class = 7500   # → 15k total (change to 10000 for 20k)

# Create folders
for c in classes:
    os.makedirs(os.path.join(output_dir, c), exist_ok=True)

# Multi-level augmentation
mild = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5)
])

medium = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2)
])

strong = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Perspective(scale=(0.02, 0.05), p=0.3)
])

for c in classes:

    class_input = os.path.join(input_dir, c)
    class_output = os.path.join(output_dir, c)

    images = os.listdir(class_input)

    print(f"\nProcessing {c}")

    count = 0
    pbar = tqdm(total=target_per_class)

    while count < target_per_class:

        img_name = random.choice(images)
        img_path = os.path.join(class_input, img_name)

        image = cv2.imread(img_path)

        # randomly choose augmentation strength
        choice = random.random()

        if choice < 0.4:
            augmented = mild(image=image)["image"]
        elif choice < 0.8:
            augmented = medium(image=image)["image"]
        else:
            augmented = strong(image=image)["image"]

        save_path = os.path.join(
            class_output,
            f"{c}_aug_{count}.jpg"
        )

        cv2.imwrite(save_path, augmented)

        count += 1
        pbar.update(1)

    pbar.close()

print("Dataset expanded successfully!")