#!/usr/bin/env python3
"""
CLIP Feature Extraction Script for VIP5

This script extracts image features using CLIP model and saves them as .npy files
for use with the VIP5 training pipeline.

Usage:
    python scripts/extract_clip_features.py --input_dir photos/toys --output_dir features/vitb32_features/toys --clip_model ViT-B/32

Arguments:
    --input_dir: Directory containing input images (supports .jpg, .jpeg, .png, .gif, .webp)
    --output_dir: Directory to save extracted features as .npy files
    --clip_model: CLIP model to use (default: ViT-B/32)
                  Options: ViT-B/32, ViT-B/16, ViT-L/14, RN50, RN101
    --batch_size: Batch size for processing (default: 32)
    --device: Device to use (default: cuda if available, else cpu)

Example:
    # Extract features for toys dataset
    python scripts/extract_clip_features.py \\
        --input_dir photos/toys \\
        --output_dir features/vitb32_features/toys \\
        --clip_model ViT-B/32

    # Extract features for beauty dataset with custom batch size
    python scripts/extract_clip_features.py \\
        --input_dir photos/beauty \\
        --output_dir features/vitb32_features/beauty \\
        --batch_size 64

Author: VIP5 Project
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

try:
    import clip
except ImportError:
    print("Error: CLIP not installed. Please install with:")
    print("  pip install git+https://github.com/openai/CLIP.git")
    exit(1)


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract CLIP features from images for VIP5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing input images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save extracted .npy features'
    )
    parser.add_argument(
        '--clip_model',
        type=str,
        default='ViT-B/32',
        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'],
        help='CLIP model variant to use (default: ViT-B/32)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for feature extraction (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: cuda if available)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing feature files'
    )

    return parser.parse_args()


def get_image_files(input_dir: str) -> list:
    """Get all image files from input directory."""
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))

    return sorted(image_files)


def load_clip_model(model_name: str, device: str):
    """Load CLIP model and preprocessing function."""
    print(f"Loading CLIP model: {model_name}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def extract_features_batch(
    model,
    preprocess,
    image_paths: list,
    device: str
) -> dict:
    """Extract features for a batch of images."""
    features = {}
    valid_images = []
    valid_paths = []

    # Load and preprocess images
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = preprocess(image)
            valid_images.append(image_tensor)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue

    if not valid_images:
        return features

    # Stack images into batch
    image_batch = torch.stack(valid_images).to(device)

    # Extract features
    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()

    # Store features with filename as key
    for i, img_path in enumerate(valid_paths):
        # Use stem (filename without extension) as the key
        features[img_path.stem] = image_features[i]

    return features


def main():
    args = parse_args()

    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files
    image_files = get_image_files(args.input_dir)
    print(f"Found {len(image_files)} images in {args.input_dir}")

    if len(image_files) == 0:
        print("No images found! Supported formats:", IMAGE_EXTENSIONS)
        return

    # Filter out already processed files if not overwriting
    if not args.overwrite:
        original_count = len(image_files)
        image_files = [
            f for f in image_files
            if not (output_path / f"{f.stem}.npy").exists()
        ]
        skipped = original_count - len(image_files)
        if skipped > 0:
            print(f"Skipping {skipped} already processed images (use --overwrite to reprocess)")

    if len(image_files) == 0:
        print("All images already processed!")
        return

    # Load CLIP model
    model, preprocess = load_clip_model(args.clip_model, device)

    # Get feature dimension for info
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        feat_dim = model.encode_image(dummy).shape[-1]
    print(f"Feature dimension: {feat_dim}")

    # Process images in batches
    print(f"Extracting features for {len(image_files)} images...")

    processed = 0
    failed = 0

    for i in tqdm(range(0, len(image_files), args.batch_size)):
        batch_files = image_files[i:i + args.batch_size]

        # Extract features for batch
        batch_features = extract_features_batch(model, preprocess, batch_files, device)

        # Save features
        for name, feature in batch_features.items():
            output_file = output_path / f"{name}.npy"
            np.save(output_file, feature)
            processed += 1

        failed += len(batch_files) - len(batch_features)

    print(f"\nDone!")
    print(f"  Processed: {processed} images")
    print(f"  Failed: {failed} images")
    print(f"  Output directory: {args.output_dir}")

    # Verify output
    npy_files = list(output_path.glob("*.npy"))
    print(f"  Total .npy files: {len(npy_files)}")


if __name__ == '__main__':
    main()
