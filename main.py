import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from sklearn.metrics import classification_report, confusion_matrix

print("Current working directory:", os.getcwd())
print("Files here:", os.listdir())

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = os.path.join("Oasis", "Data")

print("DATA_DIR:", DATA_DIR)
print("DATA_DIR absolute:", os.path.abspath(DATA_DIR))
print("Exists?", os.path.exists(DATA_DIR))
print("Contents:", os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else "N/A")


BATCH_SIZE = 32
NUM_EPOCHS_FROZEN = 8
NUM_EPOCHS_FINETUNE = 6

LR_FROZEN = 1e-3
LR_FINETUNE = 1e-4

USE_WEIGHTED_SAMPLER = True
SAVE_PATH = "resnet18_oasis_best.pt"
SEED = 42
VAL_RATIO = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# TRANSFORMS
# -------------------------
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------------
# DATASET + SPLIT
# -------------------------
# Start with train transforms
full_ds = datasets.ImageFolder(DATA_DIR, transform=train_tfms)

num_classes = len(full_ds.classes)
print("Classes:", full_ds.classes)
print("Total images:", len(full_ds))

val_size = int(len(full_ds) * VAL_RATIO)
train_size = len(full_ds) - val_size

train_ds, val_ds = random_split(
    full_ds,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# IMPORTANT:
# random_split Subset objects share the SAME underlying dataset object.
# So if we change full_ds.transform, it affects both subsets.
# Fix: create a second dataset for validation, and re-point val_ds to it.
val_base = datasets.ImageFolder(DATA_DIR, transform=val_tfms)
val_ds.dataset = val_base

print("Train size:", len(train_ds))
print("Val size:", len(val_ds))

# -------------------------
# DATALOADERS (with optional class imbalance handling)
# -------------------------
if USE_WEIGHTED_SAMPLER:
    # Get class targets for ONLY the training subset
    train_indices = train_ds.indices
    train_targets = [full_ds.targets[i] for i in train_indices]

    class_counts = np.bincount(train_targets, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-9)

    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
else:
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Sanity check batch
x, y = next(iter(train_loader))
print("Batch shape:", x.shape)  # [B, 3, 224, 224]
print("Label sample:", y[:10])

# -------------------------
# MODEL: ResNet18 Transfer Learning
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers
for p in model.parameters():
    p.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# -------------------------
# TRAIN / EVAL FUNCTIONS
# -------------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    acc = correct / total
    return acc, np.array(all_labels), np.array(all_preds)

# -------------------------
# PHASE 1: Train head (frozen backbone)
# -------------------------
optimizer = optim.Adam(model.fc.parameters(), lr=LR_FROZEN)
best_acc = 0.0

print("\n=== Phase 1: Training classifier head (backbone frozen) ===")
for epoch in range(NUM_EPOCHS_FROZEN):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer)
    val_acc, y_true, y_pred = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS_FROZEN} | "
          f"Train Loss {tr_loss:.4f} | Train Acc {tr_acc:.4f} | Val Acc {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)

# -------------------------
# PHASE 2: Fine-tune layer4 + head
# -------------------------
print("\n=== Phase 2: Fine-tuning layer4 + head ===")
for p in model.layer4.parameters():
    p.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINETUNE)

for epoch in range(NUM_EPOCHS_FINETUNE):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer)
    val_acc, y_true, y_pred = evaluate(model, val_loader)

    print(f"FT Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE} | "
          f"Train Loss {tr_loss:.4f} | Train Acc {tr_acc:.4f} | Val Acc {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)

print(f"\nBest Val Acc: {best_acc:.4f}")
print(f"Saved best model to: {SAVE_PATH}")

# -------------------------
# FINAL REPORT (load best weights)
# -------------------------
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
val_acc, y_true, y_pred = evaluate(model, val_loader)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=full_ds.classes))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))
