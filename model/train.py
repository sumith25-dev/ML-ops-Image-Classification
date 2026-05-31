"""
train.py — Train EfficientNet-B0 on a custom image dataset.

Usage:
    python model/train.py --data_dir ./data --epochs 10 --run_name baseline_v1

The script:
  1. Loads images from <data_dir>/<class_name>/*.jpg (ImageFolder layout)
  2. Fine-tunes EfficientNet-B0 (ImageNet pretrained, last layer replaced)
  3. Tracks all params, metrics, and the final model with MLflow
  4. Registers the model in the MLflow Model Registry under "image_classifier"
"""

import argparse
import logging
import os
import time

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
from torch.optim.lr_scheduler import OneCycleLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "image_classifier"

CLASSES = [
    "abstract", "landscape", "portrait", "still_life",
    "architecture", "street", "nature", "digital_art",
]


def get_transforms():
    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes: int, freeze_backbone: bool = True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def main(args):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_tf, val_tf = get_transforms()

    # ── Dataset ───────────────────────────────────────────────────────────────
    if os.path.isdir(args.data_dir):
        full_ds = datasets.ImageFolder(args.data_dir, transform=train_tf)
        n_val = max(1, int(0.2 * len(full_ds)))
        train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
        val_ds.dataset.transform = val_tf
        num_classes = len(full_ds.classes)
        logger.info(f"Dataset: {len(train_ds)} train / {len(val_ds)} val, {num_classes} classes")
    else:
        logger.warning("data_dir not found — using synthetic FakeData for smoke-test.")
        from torchvision.datasets import FakeData
        num_classes = len(CLASSES)
        train_ds = FakeData(size=200, image_size=(3, 224, 224), num_classes=num_classes, transform=train_tf)
        val_ds   = FakeData(size=40,  image_size=(3, 224, 224), num_classes=num_classes, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model / optimizer / scheduler ─────────────────────────────────────────
    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

    # ── MLflow run ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "model": "efficientnet_b0",
            "num_classes": num_classes,
            "freeze_backbone": True,
        })

        best_val_acc, best_state = 0.0, None

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_acc     = eval_epoch(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            }, step=epoch)

            logger.info(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train {train_loss:.4f}/{train_acc:.2%} | "
                f"val {val_loss:.4f}/{val_acc:.2%} | {elapsed:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                import copy
                best_state = copy.deepcopy(model.state_dict())

        # ── Save best model ───────────────────────────────────────────────────
        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_acc", best_val_acc)

        model_info = mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="image_classifier",
        )
        logger.info(f"Model registered. Run ID: {run.info.run_id}")
        logger.info(f"Best val accuracy: {best_val_acc:.2%}")

        print(f"\n✅  Training complete!")
        print(f"   Run ID     : {run.info.run_id}")
        print(f"   Best val acc: {best_val_acc:.2%}")
        print(f"   Model URI  : {model_info.model_uri}")
        print(f"\nNext steps:")
        print(f"  1. Open MLflow UI → http://localhost:5000")
        print(f"  2. Go to Models → image_classifier → assign alias 'stable'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./data/images")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--run_name",   default="training_run")
    main(parser.parse_args())
