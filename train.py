#!/usr/bin/env python3
"""
Art Style Classification Training Script
Supports 8 art styles with EfficientNet B7 and optional MSA-Net
"""

import os
import argparse
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import timm
import numpy as np
from sklearn.model_selection import train_test_split


class MSABlock(nn.Module):
    """Multimodal Style Aggregation Block"""
    def __init__(self, in_channels, reduction=16):
        super(MSABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Average and max pooling
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)

        # Concatenate and pass through FC layers
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.fc(concat).view(b, c, 1, 1)

        return x * attention.expand_as(x)


class EfficientNetWithMSA(nn.Module):
    """EfficientNet B7 with optional MSA-Net blocks"""
    def __init__(self, num_classes=8, use_msa=False, pretrained=True):
        super(EfficientNetWithMSA, self).__init__()

        # Load EfficientNet B7 with pretrained weights
        if pretrained:
            # Try different model variants with pretrained weights
            model_variants = [
                'tf_efficientnet_b7.ns_jft_in1k',  # ImageNet-21k pretrained
                'efficientnet_b7.ra_in1k',          # ImageNet-1k with RandAugment
                'tf_efficientnet_b7.aa_in1k',       # ImageNet-1k with AutoAugment
            ]

            backbone_loaded = False
            for variant in model_variants:
                try:
                    print(f"Attempting to load {variant}...")
                    self.backbone = timm.create_model(variant, pretrained=True)
                    print(f"Successfully loaded {variant}")
                    backbone_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load {variant}: {e}")
                    continue

            if not backbone_loaded:
                print("Warning: Could not load pretrained weights. Using random initialization.")
                self.backbone = timm.create_model('efficientnet_b7', pretrained=False)
        else:
            self.backbone = timm.create_model('efficientnet_b7', pretrained=False)

        # Get the number of features before the classifier
        in_features = self.backbone.classifier.in_features

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Add MSA blocks if requested
        self.use_msa = use_msa
        if use_msa:
            self.msa_block = MSABlock(in_features)

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

        # For style space visualization
        self.feature_extractor = nn.Linear(in_features, 2)

    def forward(self, x, return_features=False):
        # Extract features
        features = self.backbone(x)

        # Apply MSA if enabled
        if self.use_msa and features.dim() == 4:
            features = self.msa_block(features)
            features = features.view(features.size(0), -1)

        # Classification
        logits = self.classifier(features)

        if return_features:
            # For visualization
            coords = self.feature_extractor(features)
            return logits, coords

        return logits


def check_dataset_structure(data_dir):
    """
    Check dataset structure and determine if it's already split.

    Returns:
        str: 'split' if train/val dirs exist, 'unsplit' if class dirs exist directly
    """
    data_path = Path(data_dir)
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'

    # Check if already split
    if train_dir.exists() and val_dir.exists():
        return 'split'

    # Check for class directories
    expected_classes = ['anime', 'brush', 'thick', 'watercolor', 'photo', '3dcg', 'comic', 'pixelart']
    for class_name in expected_classes:
        class_path = data_path / class_name
        if class_path.exists() and class_path.is_dir():
            return 'unsplit'

    raise ValueError(
        f"Invalid dataset structure in {data_dir}. "
        f"Expected either train/val directories or class directories "
        f"({', '.join(expected_classes)})"
    )


def calculate_class_weights(dataset_path, class_names):
    """Calculate class weights for imbalanced datasets"""
    class_counts = {}

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
            class_counts[class_name] = count
        else:
            class_counts[class_name] = 0

    # Calculate weights (inverse frequency)
    total = sum(class_counts.values())
    if total == 0:
        raise ValueError(f"No images found in {dataset_path}")

    weights = {}
    for class_name, count in class_counts.items():
        if count > 0:
            weights[class_name] = total / (len(class_names) * count)
        else:
            weights[class_name] = 0.0

    print("\nDataset Statistics:")
    print("-" * 50)
    for class_name in class_names:
        print(f"{class_name:15s}: {class_counts[class_name]:6d} images (weight: {weights[class_name]:.4f})")
    print("-" * 50)
    print(f"Total: {total} images\n")

    return weights, class_counts


def get_data_loaders(data_dir, batch_size, val_split=0.15, random_seed=42, num_workers=4):
    """
    Create data loaders with virtual train/val split.

    Supports two dataset structures:
    1. Pre-split: data_dir/train/ and data_dir/val/
    2. Unsplit: data_dir/ with class folders directly (will be virtually split)
    """

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Check dataset structure
    structure = check_dataset_structure(data_dir)

    if structure == 'split':
        # Dataset already split into train/val
        print("Dataset already split into train/val directories")

        train_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'),
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, 'val'),
            transform=val_transform
        )

        class_names = train_dataset.classes

    else:
        # Unsplit dataset - perform virtual split
        print(f"Performing virtual train/val split ({1-val_split:.0%}/{val_split:.0%})...")

        # Load full dataset without augmentation first
        full_dataset = datasets.ImageFolder(root=data_dir)
        class_names = full_dataset.classes

        # Get labels for stratified split
        labels = [label for _, label in full_dataset.samples]

        # Perform stratified split
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_split,
            random_state=random_seed,
            stratify=labels
        )

        # Print split statistics
        print("\nVirtual Split Statistics:")
        print("-" * 50)
        train_labels = [labels[i] for i in train_indices]
        val_labels = [labels[i] for i in val_indices]

        for idx, class_name in enumerate(class_names):
            train_count = train_labels.count(idx)
            val_count = val_labels.count(idx)
            total = train_count + val_count
            print(f"{class_name:15s}: {train_count:4d} train, {val_count:4d} val (total: {total})")

        print("-" * 50)
        print(f"Total: {len(train_indices)} train, {len(val_indices)} val")
        print(f"Split ratio: {len(train_indices)/(len(train_indices)+len(val_indices)):.1%} / "
              f"{len(val_indices)/(len(train_indices)+len(val_indices)):.1%}\n")

        # Create subsets with appropriate transforms
        # Note: We need to create the full dataset with each transform
        train_dataset_full = datasets.ImageFolder(root=data_dir, transform=train_transform)
        val_dataset_full = datasets.ImageFolder(root=data_dir, transform=val_transform)

        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)

    # Calculate class weights from training data
    if structure == 'split':
        weights_dict, class_counts = calculate_class_weights(
            os.path.join(data_dir, 'train'),
            class_names
        )
        train_labels = [label for _, label in train_dataset.samples]
    else:
        # For virtual split, count from indices
        class_counts = {}
        for idx, class_name in enumerate(class_names):
            count = train_labels.count(idx)
            class_counts[class_name] = count

        total = sum(class_counts.values())
        weights_dict = {}
        for class_name, count in class_counts.items():
            if count > 0:
                weights_dict[class_name] = total / (len(class_names) * count)
            else:
                weights_dict[class_name] = 0.0

        print("\nDataset Statistics:")
        print("-" * 50)
        for class_name in class_names:
            print(f"{class_name:15s}: {class_counts[class_name]:6d} images (weight: {weights_dict[class_name]:.4f})")
        print("-" * 50)
        print(f"Total: {total} images\n")

    # Create sample weights for weighted sampling
    sample_weights = [weights_dict[class_names[label]] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create data loaders
    # Disable persistent_workers to avoid semaphore leak
    # Use 'forkserver' context for better GPU compatibility
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Convert weights to tensor for loss function
    class_weights = torch.tensor([weights_dict[name] for name in class_names],
                                  dtype=torch.float32)

    return train_loader, val_loader, class_weights, class_names


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, pbar):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Transfer data to GPU (critical for GPU training)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Verify GPU usage on first batch
        if batch_idx == 0 and epoch == 0:
            print(f"\n[GPU Check] Input device: {inputs.device}")
            print(f"[GPU Check] Target device: {targets.device}")
            print(f"[GPU Check] Model device: {next(model.parameters()).device}")
            if torch.cuda.is_available():
                print(f"[GPU Check] GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar (without creating new line)
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
        pbar.update(1)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    # Log to tensorboard
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy', epoch_acc, epoch)

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch, writer):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            # Transfer data to GPU (critical for GPU validation)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    # Log to tensorboard
    writer.add_scalar('Val/Loss', epoch_loss, epoch)
    writer.add_scalar('Val/Accuracy', epoch_acc, epoch)

    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Art Style Classification Training')
    parser.add_argument('--data_dir', type=str, default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use_msa', action='store_true',
                       help='Use MSA-Net blocks (default: False)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio if dataset is not already split (default: 0.15)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for dataset split (default: 42)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        print(f"  PyTorch CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")

        # Get GPU compute capability
        capability = torch.cuda.get_device_capability(0)
        print(f"  GPU Compute Capability: {capability[0]}.{capability[1]}")

        # Check CUDA compatibility
        try:
            # Test CUDA operation
            test_tensor = torch.zeros(1).cuda()
            print(f"  CUDA Test: Passed")
            del test_tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"\n{'='*80}")
            print("ERROR: CUDA compatibility issue detected!")
            print(f"{'='*80}")
            print(f"Error: {e}")
            print("\nPossible solutions:")
            print("1. Your GPU may not be supported by this PyTorch build")
            print("2. Reinstall PyTorch with the correct CUDA version:")
            print("   pip uninstall torch torchvision")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print(f"3. Your GPU Compute Capability ({capability[0]}.{capability[1]}) may not be supported")
            print("4. Try updating your NVIDIA drivers")
            print(f"{'='*80}\n")
            raise
    else:
        print("\nWarning: CUDA not available. Training will be very slow on CPU.")

    # Load data (with virtual split if necessary)
    print("\nLoading dataset...")
    train_loader, val_loader, class_weights, class_names = get_data_loaders(
        args.data_dir,
        args.batch_size,
        val_split=args.val_split,
        random_seed=args.random_seed,
        num_workers=args.num_workers
    )

    # Create model
    print(f"\nCreating model (MSA-Net: {'Enabled' if args.use_msa else 'Disabled'})...")
    model = EfficientNetWithMSA(
        num_classes=len(class_names),
        use_msa=args.use_msa,
        pretrained=True
    )

    # CRITICAL: Move model to GPU
    model = model.to(device)

    # Verify model is on GPU
    print(f"\n[GPU Status] Model device: {next(model.parameters()).device}")
    if torch.cuda.is_available():
        print(f"[GPU Status] Model is on GPU: {next(model.parameters()).is_cuda}")
        print(f"[GPU Status] Initial GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Loss function with class weights (move weights to GPU)
    class_weights_gpu = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_gpu)
    print(f"[GPU Status] Loss criterion weight device: {criterion.weight.device}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Save class names
    class_info = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'use_msa': args.use_msa
    }
    with open(os.path.join(args.output_dir, 'class_info.json'), 'w') as f:
        json.dump(class_info, f, indent=2)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)

        # Create progress bar for this epoch
        total_batches = len(train_loader)
        pbar = tqdm(total=total_batches, desc='Training',
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                   leave=True)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, pbar
        )
        pbar.close()

        # Validate
        print("Validating...", end=' ')
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'class_names': class_names,
            'use_msa': args.use_msa
        }

        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest_checkpoint.pth'))

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint['best_val_acc'] = best_val_acc
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"★ New best model saved! (Val Acc: {val_acc:.2f}%)")

        print(f"LR: {current_lr:.6f} | Best Val Acc: {best_val_acc:.2f}%")

    print("\n" + "=" * 80)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}")

    writer.close()


if __name__ == '__main__':
    main()
