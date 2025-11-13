import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms

H5_FILE = "Galaxy10_DECals.h5"
OUTPUT_DIR = "balanced_data"
VAL_RATIO = 0.2

# ---------- load data ----------
with h5py.File(H5_FILE, "r") as f:
    images = np.array(f["images"])
    labels = np.array(f["ans"])

print(f"Total images: {len(images)}")
print(f"Classes: {np.unique(labels)}")

train_idx, val_idx = train_test_split(np.arange(len(images)), test_size=VAL_RATIO, stratify=labels, random_state=42)

splits = {
    "train": train_idx,
    "val": val_idx
}

train_labels = labels[train_idx]
unique_classes, train_counts = np.unique(train_labels, return_counts=True)

print("\n📊 Train set class distribution (before balancing):")
for cls, count in zip(unique_classes, train_counts):
    print(f"   Class {cls}: {count} samples")

max_count = train_counts.max()
print(f"\n🎯 Target count per class: {max_count}")

# ---------- make directories ----------
for split in ["train", "val"]:
    for label in unique_classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, str(label)), exist_ok=True)

# ---------- augmentation for balancing ----------
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# ---------- save images with balancing ----------
print("\n💾 Saving images with class balancing...")

# Save validation set (no augmentation needed)
print("\nSaving validation set...")
for i in tqdm(val_idx, desc="Validation set"):
    img = Image.fromarray(images[i])
    label = str(labels[i])
    img.save(os.path.join(OUTPUT_DIR, "val", label, f"img_{i:05d}.png"))

# Save training set with balancing
print("\nSaving training set with balancing...")
saved_count = {cls: 0 for cls in unique_classes}

for cls, current_count in zip(unique_classes, train_counts):
    class_indices = train_idx[train_labels == cls]
    needed_count = max_count - current_count

    print(f"\n📌 Class {cls}:")
    print(f"   Original: {current_count} samples")
    print(f"   Need to generate: {needed_count} samples")

    # Save original images
    for i in tqdm(class_indices, desc=f"  Saving originals (Class {cls})"):
        img = Image.fromarray(images[i])
        label = str(labels[i])
        img.save(os.path.join(OUTPUT_DIR, "train", label, f"img_{i:05d}.png"))
        saved_count[cls] += 1

    # Generate augmented images if needed
    if needed_count > 0:
        aug_counter = 0
        pbar = tqdm(total=needed_count, desc=f"  Generating augmented (Class {cls})")

        while aug_counter < needed_count:
            # Randomly select an image from this class
            random_idx = np.random.choice(class_indices)
            img = Image.fromarray(images[random_idx])

            # Apply augmentation
            augmented_img = augmentation(img)

            # Save augmented image
            label = str(labels[random_idx])
            aug_filename = f"aug_{cls}_{aug_counter:05d}.png"
            augmented_img.save(os.path.join(OUTPUT_DIR, "train", label, aug_filename))

            saved_count[cls] += 1
            aug_counter += 1
            pbar.update(1)

        pbar.close()

# ---------- final statistics ----------
print("\n" + "=" * 60)
print("✅ BALANCING COMPLETE!")
print("=" * 60)

print("\n📊 Final train set distribution:")
for cls in unique_classes:
    print(f"   Class {cls}: {saved_count[cls]} samples")

print(f"\n📁 Data saved in: {OUTPUT_DIR}/")
print(f"   - train/: {sum(saved_count.values())} images (balanced)")
print(f"   - val/: {len(val_idx)} images (original)")
print("\nDone! Dataset is now balanced! 🎉")
