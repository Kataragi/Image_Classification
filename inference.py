#!/usr/bin/env python3
"""
Art Style Classification Inference Script
Supports single image, batch inference, and style space visualization
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import tqdm

# Import model and utilities from train.py
from train import EfficientNetWithMSA


class StyleSpaceVisualizer:
    """Visualize images in 2D style space"""

    def __init__(self):
        self.class_centers = {}
        self.class_coords = {}
        self.fitted = False

    def fit(self, coords, labels, class_names):
        """Fit the visualizer to training data"""
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

        # Calculate centers for each class
        for idx, class_name in enumerate(class_names):
            mask = labels_np == idx
            if mask.any():
                class_coords = coords_np[mask]
                self.class_coords[class_name] = class_coords
                self.class_centers[class_name] = np.mean(class_coords, axis=0)

        self.fitted = True

    def save_reference(self, filepath):
        """Save reference coordinates"""
        data = {
            'class_centers': {k: v.tolist() for k, v in self.class_centers.items()},
            'class_coords': {k: v.tolist() for k, v in self.class_coords.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_reference(self, filepath):
        """Load reference coordinates"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.class_centers = {k: np.array(v) for k, v in data['class_centers'].items()}
        self.class_coords = {k: np.array(v) for k, v in data['class_coords'].items()}
        self.fitted = True

    def plot(self, new_coords=None, new_labels=None, new_image_names=None,
             class_names=None, x_range=None, y_range=None, output_path='style_space.png'):
        """
        Plot the style space

        Args:
            new_coords: Coordinates of new images to plot
            new_labels: Predicted labels for new images
            new_image_names: Names of new images
            class_names: List of class names
            x_range: Tuple of (min, max) for x-axis
            y_range: Tuple of (min, max) for y-axis
            output_path: Path to save the plot
        """
        if not self.fitted:
            raise ValueError("Visualizer must be fitted first or load reference data")

        fig, ax = plt.subplots(figsize=(14, 10))

        # Primary color map for classes (vivid colors for better visibility)
        primary_colors = [
            (1.0, 0.0, 0.0),    # Red
            (0.0, 0.0, 1.0),    # Blue
            (0.0, 0.8, 0.0),    # Green
            (1.0, 0.8, 0.0),    # Yellow
            (1.0, 0.0, 1.0),    # Magenta
            (0.0, 0.8, 0.8),    # Cyan
            (1.0, 0.5, 0.0),    # Orange
            (0.6, 0.0, 0.8),    # Purple
        ]
        color_map = {name: primary_colors[i % len(primary_colors)]
                     for i, name in enumerate(self.class_centers.keys())}

        # Plot training data distribution
        for class_name, coords in self.class_coords.items():
            if len(coords) > 0:
                ax.scatter(coords[:, 0], coords[:, 1],
                          c=[color_map[class_name]], alpha=0.3, s=30,
                          label=f'{class_name} (train)')

                # Calculate and plot confidence ellipse
                if len(coords) > 2:
                    cov = np.cov(coords.T)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

                    # 2-sigma ellipse
                    width, height = 2 * 2 * np.sqrt(eigenvalues)
                    ellipse = Ellipse(self.class_centers[class_name],
                                     width, height, angle=angle,
                                     facecolor='none',
                                     edgecolor=color_map[class_name],
                                     linewidth=2, linestyle='--')
                    ax.add_patch(ellipse)

        # Plot class centers
        for class_name, center in self.class_centers.items():
            ax.scatter(center[0], center[1],
                      c=[color_map[class_name]], s=300, marker='*',
                      edgecolors='black', linewidths=2,
                      label=f'{class_name} (center)', zorder=5)

        # Plot new images
        if new_coords is not None:
            new_coords_np = new_coords.cpu().numpy() if torch.is_tensor(new_coords) else new_coords

            if new_labels is not None and class_names is not None:
                new_labels_np = new_labels.cpu().numpy() if torch.is_tensor(new_labels) else new_labels

                for i, (coord, label) in enumerate(zip(new_coords_np, new_labels_np)):
                    class_name = class_names[label]
                    marker_color = color_map.get(class_name, 'gray')

                    ax.scatter(coord[0], coord[1],
                              c=[marker_color], s=200, marker='D',
                              edgecolors='red', linewidths=2,
                              zorder=6)

                    # Add image name annotation
                    if new_image_names is not None and i < len(new_image_names):
                        ax.annotate(new_image_names[i],
                                   (coord[0], coord[1]),
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=8, color='red',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Set axis ranges if specified
        if x_range is not None:
            ax.set_xlim(x_range)
        if y_range is not None:
            ax.set_ylim(y_range)

        ax.set_xlabel('Style Dimension 1', fontsize=12)
        ax.set_ylabel('Style Dimension 2', fontsize=12)
        ax.set_title('Art Style Space Visualization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Legend (remove duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                 loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Style space visualization saved to: {output_path}")
        plt.close()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model configuration
    class_names = checkpoint.get('class_names', [])
    use_msa = checkpoint.get('use_msa', False)
    resolution = checkpoint.get('resolution', 600)  # Default to 600 for backward compatibility
    hidden_dim = checkpoint.get('hidden_dim', 512)  # Default to 512 for old checkpoints

    # Create model
    model = EfficientNetWithMSA(
        num_classes=len(class_names),
        use_msa=use_msa,
        pretrained=False,
        hidden_dim=hidden_dim
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, class_names, resolution


def get_transform(resolution=600):
    """Get inference transform"""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def predict_single_image(model, image_path, transform, class_names, device):
    """Predict single image (full image mode)"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)

    # Add batch dimension: (1, C, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits, coords = model(input_tensor, return_features=True)
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, 1)

    result = {
        'predicted_class': class_names[predicted.item()],
        'confidence': confidence.item(),
        'probabilities': {class_names[i]: probs[0][i].item() for i in range(len(class_names))},
        'coordinates': coords[0].cpu().numpy()
    }

    return result


def predict_folder(model, folder_path, transform, class_names, device):
    """Predict all images in a folder"""
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in Path(folder_path).iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {folder_path}")
        return []

    results = []
    print(f"\nProcessing {len(image_files)} images...")

    for image_path in tqdm(image_files, desc="Inference"):
        try:
            result = predict_single_image(model, image_path, transform, class_names, device)
            result['image_path'] = str(image_path)
            result['image_name'] = image_path.name
            results.append(result)
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")

    return results


def build_style_space_reference(model, data_dir, class_names, device, output_path, resolution=600):
    """Build style space reference from training data"""
    transform = get_transform(resolution)

    all_coords = []
    all_labels = []

    print("\nBuilding style space reference from training data...")

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, 'train', class_name)

        if not os.path.exists(class_path):
            print(f"Warning: {class_path} not found")
            continue

        image_files = [f for f in Path(class_path).iterdir()
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]

        print(f"Processing {class_name}: {len(image_files)} images")

        for image_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
            try:
                image = Image.open(image_path).convert('RGB')
                input_tensor = transform(image)

                # Add batch dimension for full image mode
                input_tensor = input_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    _, coords = model(input_tensor, return_features=True)
                    all_coords.append(coords[0])
                    all_labels.append(idx)

            except Exception as e:
                print(f"\n  Error processing {image_path}: {e}")

    if not all_coords:
        raise ValueError("No coordinates extracted from training data")

    all_coords = torch.stack(all_coords)
    all_labels = torch.tensor(all_labels)

    # Create and fit visualizer
    visualizer = StyleSpaceVisualizer()
    visualizer.fit(all_coords, all_labels, class_names)
    visualizer.save_reference(output_path)

    print(f"\nStyle space reference saved to: {output_path}")

    return visualizer


def main():
    parser = argparse.ArgumentParser(description='Art Style Classification Inference')

    # Model and data paths
    parser.add_argument('--checkpoint', type=str, default='outputs/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--folder', type=str, default=None,
                       help='Path to folder for batch inference')

    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Enable style space visualization')
    parser.add_argument('--build_reference', action='store_true',
                       help='Build style space reference from training data')
    parser.add_argument('--data_dir', type=str, default='dataset',
                       help='Path to dataset directory (for building reference)')
    parser.add_argument('--reference', type=str, default='outputs/style_space_reference.json',
                       help='Path to style space reference file')
    parser.add_argument('--x_range', type=float, nargs=2, default=None,
                       metavar=('MIN', 'MAX'),
                       help='X-axis range for visualization (e.g., --x_range -5 5)')
    parser.add_argument('--y_range', type=float, nargs=2, default=None,
                       metavar=('MIN', 'MAX'),
                       help='Y-axis range for visualization (e.g., --y_range -5 5)')
    parser.add_argument('--output_plot', type=str, default='style_space.png',
                       help='Output path for style space plot')

    # Output options
    parser.add_argument('--output_json', type=str, default=None,
                       help='Save results to JSON file')

    args = parser.parse_args()

    # Validate arguments
    if not args.image and not args.folder and not args.build_reference:
        parser.error("Please specify --image, --folder, or --build_reference")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model, class_names, resolution = load_model(args.checkpoint, device)
    print(f"Model loaded. Classes: {', '.join(class_names)}")
    print(f"Using resolution: {resolution}x{resolution}")

    transform = get_transform(resolution)

    # Build style space reference if requested
    if args.build_reference:
        visualizer = build_style_space_reference(
            model, args.data_dir, class_names, device, args.reference, resolution
        )

        # Plot training data distribution
        if args.visualize:
            visualizer.plot(
                class_names=class_names,
                x_range=tuple(args.x_range) if args.x_range else None,
                y_range=tuple(args.y_range) if args.y_range else None,
                output_path=args.output_plot
            )
        return

    # Single image inference
    if args.image:
        print(f"\nPredicting: {args.image}")
        result = predict_single_image(model, args.image, transform, class_names, device)

        print("\n" + "=" * 60)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nClass Probabilities:")
        for class_name, prob in sorted(result['probabilities'].items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"  {class_name:15s}: {prob:6.2%} {'█' * int(prob * 50)}")
        print(f"\nStyle Coordinates: ({result['coordinates'][0]:.4f}, {result['coordinates'][1]:.4f})")
        print("=" * 60)

        # Visualization
        if args.visualize:
            if not os.path.exists(args.reference):
                print(f"\nWarning: Reference file {args.reference} not found.")
                print("Run with --build_reference first to create reference data.")
            else:
                visualizer = StyleSpaceVisualizer()
                visualizer.load_reference(args.reference)

                coords = torch.tensor([result['coordinates']])
                labels = torch.tensor([class_names.index(result['predicted_class'])])

                visualizer.plot(
                    new_coords=coords,
                    new_labels=labels,
                    new_image_names=[Path(args.image).name],
                    class_names=class_names,
                    x_range=tuple(args.x_range) if args.x_range else None,
                    y_range=tuple(args.y_range) if args.y_range else None,
                    output_path=args.output_plot
                )

    # Folder inference
    elif args.folder:
        results = predict_folder(model, args.folder, transform, class_names, device)

        if results:
            print("\n" + "=" * 80)
            print("Results Summary:")
            print("-" * 80)

            for result in results:
                print(f"{result['image_name']:30s} -> {result['predicted_class']:15s} "
                      f"({result['confidence']:.2%})")

            print("=" * 80)

            # Class distribution
            class_counts = {}
            for result in results:
                pred = result['predicted_class']
                class_counts[pred] = class_counts.get(pred, 0) + 1

            print("\nClass Distribution:")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name:15s}: {count:4d} ({count/len(results):.1%})")

            # Save to JSON if requested
            if args.output_json:
                with open(args.output_json, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    for result in results:
                        result['coordinates'] = result['coordinates'].tolist()
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {args.output_json}")

            # Visualization
            if args.visualize:
                if not os.path.exists(args.reference):
                    print(f"\nWarning: Reference file {args.reference} not found.")
                    print("Run with --build_reference first to create reference data.")
                else:
                    visualizer = StyleSpaceVisualizer()
                    visualizer.load_reference(args.reference)

                    coords = torch.tensor([r['coordinates'] if isinstance(r['coordinates'], list)
                                          else r['coordinates'] for r in results])
                    labels = torch.tensor([class_names.index(r['predicted_class']) for r in results])
                    image_names = [r['image_name'] for r in results]

                    visualizer.plot(
                        new_coords=coords,
                        new_labels=labels,
                        new_image_names=image_names,
                        class_names=class_names,
                        x_range=tuple(args.x_range) if args.x_range else None,
                        y_range=tuple(args.y_range) if args.y_range else None,
                        output_path=args.output_plot
                    )


if __name__ == '__main__':
    main()
