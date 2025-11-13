import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

H5_FILE = "Galaxy10_DECals.h5"
OUTPUT_DIR = "data"
VAL_RATIO = 0.2

# ---------- load data ----------
with h5py.File(H5_FILE, "r") as f:
    images = np.array(f["images"])
    labels = np.array(f["ans"])

# ---------- split data ----------
train_idx, val_idx = train_test_split(np.arange(len(images)), test_size=VAL_RATIO, stratify=labels)

splits = {
    "train": train_idx,
    "val": val_idx
}

# ---------- make directories ----------
for split in ["train", "val"]:
    for label in np.unique(labels):
        os.makedirs(os.path.join(OUTPUT_DIR, split, str(label)), exist_ok=True)

# ---------- save images ----------
print("Saving images by class folders...")
for split, idxs in splits.items():
    for i in tqdm(idxs, desc=f"{split} set"):
        img = Image.fromarray(images[i])
        label = str(labels[i])
        img.save(os.path.join(OUTPUT_DIR, split, label, f"img_{i:05d}.png"))

print("Done! Images saved in class-based folders.")
