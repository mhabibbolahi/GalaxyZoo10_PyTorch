import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from os import path
from collections import defaultdict

class Config:
    """central settings of project"""
    # paths
    TRAIN_DIR = path.join('balanced_data', 'train')
    VAL_DIR = path.join('balanced_data', 'val')
    MODEL_PATH = 'based_on_balanced_data_model.pth'

    # hiper parameters
    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    NUM_CLASSES = 10

    # Early Stopping
    PATIENCE = 10
    MIN_DELTA = 0.0001  # minimum acceptable improvement


def setup_device():
    """config devise"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU name : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU didn't find CPU returned")
    return device


def load_data(train_dir, val_dir, batch_size, image_size, device):
    """load & preprocess data"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == 'cuda'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == 'cuda'
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}\n")

    return train_loader, val_loader


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Feature Extraction Blocks
        self.features = nn.Sequential(
            # Block 1: 256x256 -> 128x128
            self._make_conv_block(in_channels=3, out_channels=32, num_convs=3, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),

            # Block 2: 128x128 -> 64x64
            self._make_conv_block(in_channels=32, out_channels=64, num_convs=3, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),

            # Block 3: 64x64 -> 32x32
            self._make_conv_block(in_channels=64, out_channels=128, num_convs=3, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),

            # Block 4: 32x32 -> 16x16
            self._make_conv_block(in_channels=128, out_channels=256, num_convs=3, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),

            # Block 5: 16x16 -> 8x8
            self._make_conv_block(in_channels=256, out_channels=512, num_convs=3, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 10),
        )
        # Fully Connected Layers
        self._initialize_weights()

    def _make_conv_block(self, in_channels, out_channels, num_convs=3, kernel_size=3, padding=1):
        """make a Conv block with BatchNorm"""
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """train model for each epoch"""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # Show progress
        if (batch_idx + 1) % 20 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = correct / total
            print(f"   Batch [{batch_idx + 1}/{len(train_loader)}] | "
                  f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """validation model on validation set"""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(model, num_epochs, train_loader, test_loader, optimizer, criterion, device, scheduler, min_delta,
                patience, model_path):
    last_val_loss = float('inf')
    patience_counter = 0
    train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    print(f"\n{'=' * 50}")
    print(f"{num_epochs} Epoch")
    print(f"{'=' * 50}\n")
    last_epoch, test_loss, test_acc = 0, 0, 0
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print("-" * 70)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step(test_loss)

        # save history
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(test_loss)
        train_history['val_acc'].append(test_acc)

        current_lr = optimizer.param_groups[0]['lr']

        # show progress of each epoch
        print(f"\n Epoch {epoch + 1} Summary:")
        print(f"   last batch Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test Loss: {test_loss:.4f}\n")
        print(f"   Learning Rate: {current_lr:.6f}")
        print(f"\n{'=' * 75}")
        if test_loss < last_val_loss - min_delta:
            last_val_loss = test_loss
            patience_counter = 0
            print(f"   Loss improved")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{patience})")

        # Early Stopping
        last_epoch = epoch + 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {last_epoch} epochs\n")
            break

    torch.save(model, model_path)
    torch.save({
        'epoch': last_epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_history,
        'final_val_loss': test_loss,
        'final_val_acc': test_acc,
    }, model_path)
    print(f"Final model saved at: {model_path}\n")

    return train_history


def main():
    # setup device
    device = setup_device()

    # Loading data
    print("\nLoading data...")
    train_loader, val_loader = load_data(
        Config.TRAIN_DIR,
        Config.VAL_DIR,
        Config.BATCH_SIZE,
        Config.IMAGE_SIZE,
        device
    )

    # making model
    print("Building model...")
    model = CNNModel().to(device)

    # show model summery
    summary(model, input_size=(3, 256, 256))

    # Training settings
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True,
        min_lr=1e-7
    )

    # train model
    history = train_model(model, Config.NUM_EPOCHS, train_loader, val_loader, optimizer, criterion, device, scheduler,
                          Config.MIN_DELTA, Config.PATIENCE, Config.MODEL_PATH)
    print(history)

    # loading final model for evaluation
    print("\n Loading final model for evaluation...")
    checkpoint = torch.load(Config.MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    # final evaluation
    final_loss, final_acc = evaluate(model, val_loader, criterion, device)

    print(f"\n{'=' * 70}")
    print(f"Final Validation Results:")
    print(f"Loss: {final_loss:.4f}")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"{'=' * 70}\n")
    print(f"Model saved at: {Config.MODEL_PATH}\n")


if __name__ == '__main__':
    main()
