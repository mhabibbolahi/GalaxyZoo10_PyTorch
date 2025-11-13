# GalaxyZoo10 PyTorch Classification

A deep learning project for classifying galaxy morphologies using the Galaxy10 DECals dataset. This is a simple implementation completed in 2-3 days, featuring a custom CNN architecture with data balancing and augmentation techniques.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Network Architecture](#network-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)

## 🌌 Overview

This project implements a custom Convolutional Neural Network (CNN) to classify galaxies into 10 morphological categories from the Galaxy10 DECals dataset. The implementation includes:
- Custom deep CNN architecture (5 convolutional blocks)
- Data preprocessing and balancing pipeline
- FastAPI-based web interface for inference
- Training with early stopping and learning rate scheduling

**Note:** This is a simple educational project developed over 2-3 days to demonstrate galaxy classification using PyTorch.

## 🔭 Dataset

**Galaxy10 DECals Dataset**

This project uses the Galaxy10 DECaLS dataset, an improved version of the original Galaxy10 that combines Galaxy Zoo Data Release 2 classifications with high-quality images from the DESI Legacy Imaging Surveys (DECaLS).

### Dataset Overview
- **Total Images:** 17,736 colored galaxy images
- **Image Resolution:** 256×256 pixels
- **Color Bands:** g, r, and z bands (3 channels)
- **Format:** HDF5 file (`Galaxy10_DECals.h5`)
- **Source:** DESI Legacy Imaging Surveys (DECaLS, BASS, MzLS)
- **Labels:** Galaxy Zoo volunteer classifications

### Class Distribution (Imbalanced)
The dataset contains 10 morphological classes with varying sample counts:

| Class | Category | Image Count |
|-------|----------|-------------|
| 0 | Disturbed Galaxies | 1,081 |
| 1 | Merging Galaxies | 1,853 |
| 2 | Round Smooth Galaxies | 2,645 |
| 3 | In-between Round Smooth Galaxies | 2,027 |
| 4 | Cigar Shaped Smooth Galaxies | 334 |
| 5 | Barred Spiral Galaxies | 2,043 |
| 6 | Unbarred Tight Spiral Galaxies | 1,829 |
| 7 | Unbarred Loose Spiral Galaxies | 2,628 |
| 8 | Edge-on Galaxies without Bulge | 1,423 |
| 9 | Edge-on Galaxies with Bulge | 1,873 |

**Note:** The dataset is highly imbalanced, with Class 4 (Cigar Shaped) having only 334 images while Class 7 (Loose Spiral) has 2,628 images. This project addresses this imbalance through data augmentation.

### Dataset Citation
If you use this dataset, please cite:
- **Leung, H. W., & Bovy, J.** (2019). Deep learning of multi-element abundances from high-resolution spectroscopic data. *Monthly Notices of the Royal Astronomical Society*, 483(3), 3255-3277.
- **Dataset DOI:** [10.5281/zenodo.10845026](https://doi.org/10.5281/zenodo.10845026)
- **Original Paper:** [10.1093/mnras/sty3217](https://doi.org/10.1093/mnras/sty3217)

## 🧠 Network Architecture

### DeepCNN-5Block Architecture

The model consists of 5 convolutional blocks followed by a multi-layer classifier. Here's the detailed architecture:

#### Feature Extraction Layers

**Block 1: Input(256x256) → 128×128**
- 3× Conv2D(3→32, kernel=3×3, padding=1) + BatchNorm + ReLU
- MaxPool2D(2×2)
- Output: 32 channels, 128×128 spatial resolution

**Block 2: 128×128 → 64×64**
- 3× Conv2D(32→64, kernel=3×3, padding=1) + BatchNorm + ReLU
- MaxPool2D(2×2)
- Output: 64 channels, 64×64 spatial resolution

**Block 3: 64×64 → 32×32**
- 3× Conv2D(64→128, kernel=3×3, padding=1) + BatchNorm + ReLU
- MaxPool2D(2×2)
- Output: 128 channels, 32×32 spatial resolution

**Block 4: 32×32 → 16×16**
- 3× Conv2D(128→256, kernel=3×3, padding=1) + BatchNorm + ReLU
- MaxPool2D(2×2)
- Output: 256 channels, 16×16 spatial resolution

**Block 5: 16×16 → 8×8**
- 3× Conv2D(256→512, kernel=3×3, padding=1) + BatchNorm + ReLU
- MaxPool2D(2×2)
- Output: 512 channels, 8×8 spatial resolution

#### Classifier Layers

After flattening (512×8×8 = 32,768 features):

1. **FC Layer 1:** 32,768 → 2,048 + ReLU + Dropout(0.4)
2. **FC Layer 2:** 2,048 → 512 + ReLU + Dropout(0.3)
3. **FC Layer 3:** 512 → 128 + ReLU + Dropout(0.2)
4. **FC Layer 4:** 128 → 32 + ReLU + Dropout(0.1)
5. **Output Layer:** 32 → 10 (number of classes)

#### Key Architecture Features

- **Total Convolutional Layers:** 15 (3 per block × 5 blocks)
- **Batch Normalization:** Applied after each convolution
- **Activation Function:** ReLU (in-place for memory efficiency)
- **Regularization:** Progressive dropout (0.4 → 0.3 → 0.2 → 0.1)
- **Weight Initialization:** 
  - Kaiming Normal for Conv2D layers
  - Normal distribution for Linear layers
- **Input Size:** 256×256×3 (RGB images)
- **Output:** 10-class probability distribution

#### Architecture Design Principles

1. **Progressive Feature Extraction:** Channel depth increases (32→64→128→256→512) while spatial dimensions decrease
2. **Multiple Convolutions per Block:** 3 consecutive convolutions in each block for better feature learning
3. **BatchNorm After Each Conv:** Improves training stability and convergence
4. **Decreasing Dropout Rates:** Higher dropout in early FC layers, lower in later layers
5. **Deep Classifier:** 5 fully connected layers for complex decision boundaries

## ✨ Features

### Data Processing
- **Automatic Data Splitting:** 80% training, 20% validation (stratified)
- **Class Balancing:** Augmentation-based balancing to handle imbalanced classes
- **Data Augmentation (Training):**
  - Random horizontal flip (p=0.5)
  - Random rotation (±15°)
  - Color jitter (brightness & contrast ±20%)
  - Random affine transformation
  - Random perspective distortion

### Training Features
- **Early Stopping:** Patience of 10 epochs with minimum delta of 0.0001
- **Learning Rate Scheduling:** ReduceLROnPlateau (factor=0.5, patience=2)
- **Label Smoothing:** CrossEntropyLoss with smoothing=0.1
- **Gradient Clipping:** Max norm of 1.0 to prevent exploding gradients
- **Mixed Precision Training Ready:** CUDA optimization with pinned memory
- **Checkpoint Saving:** Complete model state, optimizer state, and training history

### Inference
- **FastAPI Web Interface:** Upload images for real-time classification
- **Simple Inference Script:** Command-line inference capability
- **Automatic Device Selection:** GPU if available, CPU fallback

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/mhabibbolahi/GalaxyZoo10_PyTorch.git
cd GalaxyZoo10_PyTorch
```

2. **Install dependencies:**
```bash
pip install torch torchvision
pip install h5py numpy pillow tqdm scikit-learn
pip install torchsummary
pip install fastapi uvicorn jinja2 python-multipart
```

3. **Download the Galaxy10 DECals dataset:**
   - Place `Galaxy10_DECals.h5` in the project root directory

## 📖 Usage

### 1. Data Preparation

**Extract and split the dataset:**
```bash
python load_data.py
```
This creates a `data/` directory with train/val splits.

**Balance the dataset (recommended):**
```bash
python preprocess_and_balancing.py
```
This creates a `balanced_data/` directory with balanced class distributions.

### 2. Training

**Train the model:**
```bash
python cnn.py
```

**Training Configuration** (in `Config` class):
- Image Size: 256×256
- Batch Size: 32
- Learning Rate: 0.001
- Max Epochs: 100
- Early Stopping Patience: 10

The trained model will be saved as `based_on_balanced_data_model.pth`.

### 3. Inference

**Option A: Web Interface**
```bash
python main.py
```
Then open `http://127.0.0.1:8000` in your browser and upload galaxy images.

**Option B: Programmatic Inference**
```python
from cnn_use_model import main

result = main('path/to/galaxy_image.jpg', 'based_on_balanced_data_model.pth')
print(f"Predicted class: {result}")
```

## 📁 Project Structure

```
GalaxyZoo10_PyTorch/
│
├── cnn.py                           # Main training script with CNN model
├── cnn_use_model.py                 # Inference script
├── load_data.py                     # Basic data extraction and splitting
├── preprocess_and_balancing.py      # Advanced preprocessing with balancing
├── main.py                          # FastAPI web application
│
├── Galaxy10_DECals.h5               # Dataset (not included, must download)
├── based_on_balanced_data_model.pth # Trained model checkpoint
│
├── data/                            # Unbalanced dataset (created by load_data.py)
│   ├── train/
│   │   ├── 0/
│   │   ├── 1/
│   │   └── ...
│   └── val/
│       ├── 0/
│       └── ...
│
├── balanced_data/                   # Balanced dataset (created by preprocess_and_balancing.py)
│   ├── train/
│   └── val/
│
├── templates/                       # HTML templates for web interface
│   ├── index.html
│   └── image_result.html
│
└── static/                          # Static files and uploaded images
```

## ⚙️ Configuration

All hyperparameters are centralized in the `Config` class in `cnn.py`:

```python
class Config:
    # Paths
    TRAIN_DIR = 'balanced_data/train'
    VAL_DIR = 'balanced_data/val'
    MODEL_PATH = 'based_on_balanced_data_model.pth'
    
    # Hyperparameters
    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    NUM_CLASSES = 10
    
    # Early Stopping
    PATIENCE = 10
    MIN_DELTA = 0.0001
```

## 📊 Results

The model training includes:
- Real-time training and validation metrics
- Automatic model checkpointing
- Learning rate reduction on plateau
- Early stopping to prevent overfitting

Training history includes:
- Train/validation loss per epoch
- Train/validation accuracy per epoch
- Learning rate schedule
- Final model performance metrics

## 🛠️ Technical Details

### Model Summary
- **Total Parameters:** ~42M (approximately)
- **Input Shape:** (batch_size, 3, 256, 256)
- **Output Shape:** (batch_size, 10)
- **Loss Function:** CrossEntropyLoss with label smoothing
- **Optimizer:** Adam (lr=0.001)
- **Scheduler:** ReduceLROnPlateau

### GPU Support
The code automatically detects and uses CUDA if available:
- Mixed precision training support
- Pinned memory for faster data transfer
- Multi-worker data loading (num_workers=2)

## 📝 Notes

- This is a **simple educational project** developed in 2-3 days
- The architecture is custom-designed but not optimized for production
- Training time depends on GPU availability (significantly faster with CUDA)
- The web interface is basic and intended for demonstration purposes
- For production use, consider using transfer learning with pre-trained models (ResNet, EfficientNet, etc.)

## 🤝 Contributing

This is a simple educational project, but feel free to:
- Fork the repository
- Experiment with different architectures
- Try different hyperparameters
- Implement additional features

## 📄 License

This project is open source and available for educational purposes.

## 🔗 Dataset Citation

If you use the Galaxy10 DECals dataset, please cite the original work appropriately.

---

**Developed by:** mhabibbolahi 
**Email** m.habibbolahi@gmail.com 
**GitHub:** [https://github.com/mhabibbolahi/GalaxyZoo10_PyTorch](https://github.com/mhabibbolahi/GalaxyZoo10_PyTorch)  
**Project Duration:** 2-3 days