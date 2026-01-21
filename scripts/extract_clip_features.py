#!/usr/bin/env python3
"""
CLIP Feature Extraction Script for VIP5

This script extracts image features using CLIP model and saves them as .npy files
for use with the VIP5 training pipeline.

Supports two modes:
1. Direct mode: Extract features from images, using image filename as output name
2. Mapping mode: Use item2img_dict.pkl to map ASIN to image files (for VIP5 datasets)

Usage:
    # Mode 1: Direct extraction (filename -> filename.npy)
    python scripts/extract_clip_features.py \\
        --input_dir photos/toys \\
        --output_dir features/vitb32_features/toys

    # Mode 2: With ASIN mapping (ASIN -> ASIN.npy)
    python scripts/extract_clip_features.py \\
        --input_dir photos/toys \\
        --output_dir features/vitb32_features/toys \\
        --mapping_file data/toys/item2img_dict.pkl \\
        --dataset toys

Author: VIP5 Project
"""

import os
import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from urllib.parse import unquote

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
        '--mapping_file',
        type=str,
        default=None,
        help='Path to item2img_dict.pkl mapping file (ASIN -> image path)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['toys', 'beauty', 'sports', 'clothing'],
        help='Dataset name (used to parse mapping paths)'
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


def load_mapping(mapping_file: str, dataset: str, input_dir: str) -> dict:
    """
    Load ASIN to image path mapping from item2img_dict.pkl

    Returns:
        dict: {asin: full_image_path}
    """
    print(f"Loading mapping from: {mapping_file}")

    with open(mapping_file, 'rb') as f:
        item2img = pickle.load(f)

    print(f"Found {len(item2img)} ASIN mappings")

    # Convert mapping paths to actual file paths
    # Original format: "toys_photos/51XCjcMthML._SY300_.jpg"
    # We need: "photos/toys/51XCjcMthML._SY300_.jpg"

    asin_to_path = {}
    input_path = Path(input_dir)

    for asin, img_path in item2img.items():
        # Extract just the filename from the mapping path
        # e.g., "toys_photos/51XCjcMthML._SY300_.jpg" -> "51XCjcMthML._SY300_.jpg"
        img_filename = Path(img_path).name

        # URL decode the filename (handle %2B etc.)
        img_filename_decoded = unquote(img_filename)

        # Build the actual path
        actual_path = input_path / img_filename
        actual_path_decoded = input_path / img_filename_decoded

        # Check which version exists
        if actual_path.exists():
            asin_to_path[asin] = actual_path
        elif actual_path_decoded.exists():
            asin_to_path[asin] = actual_path_decoded
        else:
            # Try to find by partial match (sometimes filenames differ slightly)
            # This handles cases where the exact filename might have encoding differences
            pass  # Will be handled as missing

    found = len(asin_to_path)
    missing = len(item2img) - found
    print(f"Mapped {found} ASINs to existing images ({missing} images not found)")

    return asin_to_path


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


def extract_single_feature(model, preprocess, img_path: Path, device: str):
    """Extract feature for a single image."""
    try:
        image = Image.open(img_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model.encode_image(image_tensor)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            feature = feature.cpu().numpy().squeeze()

        return feature
    except Exception as e:
        print(f"Warning: Failed to process {img_path}: {e}")
        return None


def extract_features_batch(
    model,
    preprocess,
    items: list,  # List of (output_name, image_path)
    device: str
) -> dict:
    """Extract features for a batch of images."""
    features = {}
    valid_items = []
    valid_images = []

    # Load and preprocess images
    for output_name, img_path in items:
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = preprocess(image)
            valid_images.append(image_tensor)
            valid_items.append(output_name)
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

    # Store features
    for i, output_name in enumerate(valid_items):
        features[output_name] = image_features[i]

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

    # Determine mode: mapping or direct
    if args.mapping_file:
        # Mapping mode: use item2img_dict.pkl
        print("\n=== Mapping Mode ===")
        print(f"Using ASIN -> Image mapping from {args.mapping_file}")

        asin_to_path = load_mapping(args.mapping_file, args.dataset, args.input_dir)

        # Create list of (asin, image_path) to process
        items_to_process = list(asin_to_path.items())

        # Filter already processed
        if not args.overwrite:
            original_count = len(items_to_process)
            items_to_process = [
                (asin, path) for asin, path in items_to_process
                if not (output_path / f"{asin}.npy").exists()
            ]
            skipped = original_count - len(items_to_process)
            if skipped > 0:
                print(f"Skipping {skipped} already processed ASINs (use --overwrite to reprocess)")

        if len(items_to_process) == 0:
            print("All items already processed!")
            return

        print(f"Processing {len(items_to_process)} items...")

    else:
        # Direct mode: use filenames
        print("\n=== Direct Mode ===")
        image_files = get_image_files(args.input_dir)
        print(f"Found {len(image_files)} images in {args.input_dir}")

        if len(image_files) == 0:
            print("No images found! Supported formats:", IMAGE_EXTENSIONS)
            return

        # Create list of (filename_stem, image_path) to process
        items_to_process = [(f.stem, f) for f in image_files]

        # Filter already processed
        if not args.overwrite:
            original_count = len(items_to_process)
            items_to_process = [
                (name, path) for name, path in items_to_process
                if not (output_path / f"{name}.npy").exists()
            ]
            skipped = original_count - len(items_to_process)
            if skipped > 0:
                print(f"Skipping {skipped} already processed images (use --overwrite to reprocess)")

        if len(items_to_process) == 0:
            print("All images already processed!")
            return

    # Load CLIP model
    model, preprocess = load_clip_model(args.clip_model, device)

    # Get feature dimension for info
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        feat_dim = model.encode_image(dummy).shape[-1]
    print(f"Feature dimension: {feat_dim}")

    # Process in batches
    print(f"\nExtracting features for {len(items_to_process)} items...")

    processed = 0
    failed = 0

    for i in tqdm(range(0, len(items_to_process), args.batch_size)):
        batch_items = items_to_process[i:i + args.batch_size]

        # Extract features for batch
        batch_features = extract_features_batch(model, preprocess, batch_items, device)

        # Save features
        for name, feature in batch_features.items():
            output_file = output_path / f"{name}.npy"
            np.save(output_file, feature)
            processed += 1

        failed += len(batch_items) - len(batch_features)

    print(f"\nDone!")
    print(f"  Processed: {processed} items")
    print(f"  Failed: {failed} items")
    print(f"  Output directory: {args.output_dir}")

    # Verify output
    npy_files = list(output_path.glob("*.npy"))
    print(f"  Total .npy files: {len(npy_files)}")


if __name__ == '__main__':
    main()
