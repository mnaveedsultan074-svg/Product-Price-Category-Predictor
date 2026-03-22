#!/usr/bin/env python3
"""
Download datasets from Kaggle.
Requires KAGGLE_USERNAME and KAGGLE_KEY env vars or ~/.kaggle/kaggle.json
Falls back to synthetic sample data if credentials unavailable.
"""
import os
import sys
import json
import zipfile
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FLIPKART_CSV = RAW_DIR / "flipkart_com-ecommerce_sample.csv"
AMAZON_CSV   = RAW_DIR / "Amazon_Unlocked_Mobile.csv"


def download_via_kaggle(dataset_slug, dest_dir):
    """Download a Kaggle dataset using the kaggle CLI."""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug,
             "-p", str(dest_dir), "--unzip"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print(f"[OK] Downloaded {dataset_slug}")
            return True
        else:
            print(f"[WARN] kaggle CLI failed for {dataset_slug}: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"[WARN] kaggle CLI exception: {e}")
        return False


def create_synthetic_flipkart(path):
    """Generate realistic synthetic Flipkart-like data for development."""
    np.random.seed(42)
    n = 5000
    categories = [
        "Electronics >> Mobiles >> Smartphones",
        "Electronics >> Cameras >> DSLR",
        "Clothing >> Men >> T-Shirts",
        "Home & Kitchen >> Cookware",
        "Books >> Fiction",
        "Sports >> Fitness >> Gym Equipment",
        "Beauty >> Skincare",
        "Toys >> Action Figures",
    ]
    prices = np.concatenate([
        np.random.uniform(100, 800, n // 4),
        np.random.uniform(800, 3000, n // 4),
        np.random.uniform(3000, 10000, n // 4),
        np.random.uniform(10000, 50000, n // 4),
    ])
    np.random.shuffle(prices)
    adjectives = ["Premium", "Budget", "Professional", "Luxury", "Basic", "Advanced", "Smart", "Ultra"]
    nouns = ["Phone", "Camera", "Shirt", "Pan", "Novel", "Dumbbell", "Cream", "Toy"]
    df = pd.DataFrame({
        "product_name": [
            f"{np.random.choice(adjectives)} {np.random.choice(nouns)} {i}"
            for i in range(n)
        ],
        "retail_price": prices * 1.2,
        "discounted_price": prices,
        "description": [
            f"High quality product with excellent features. Rating: {np.random.uniform(1,5):.1f}. "
            f"Category: {np.random.choice(categories)}. Best seller in its category."
            for _ in range(n)
        ],
        "product_category_tree": np.random.choice(categories, n),
        "product_rating": np.random.uniform(1, 5, n).round(1),
        "overall_rating": np.random.uniform(1, 5, n).round(1),
    })
    df.to_csv(path, index=False)
    print(f"[SYNTHETIC] Created {path} with {n} rows")


def create_synthetic_amazon(path):
    """Generate realistic synthetic Amazon-like data for development."""
    np.random.seed(123)
    n = 5000
    brands = ["Samsung", "Apple", "Nokia", "Motorola", "LG", "Sony", "HTC", "Huawei", "OnePlus", "Xiaomi"]
    prices = np.concatenate([
        np.random.uniform(50, 200, n // 4),
        np.random.uniform(200, 500, n // 4),
        np.random.uniform(500, 800, n // 4),
        np.random.uniform(800, 1500, n // 4),
    ])
    np.random.shuffle(prices)
    df = pd.DataFrame({
        "Product Name": [
            f"{np.random.choice(brands)} {np.random.choice(['Galaxy', 'iPhone', 'Pixel', 'Edge', 'Note'])} {np.random.randint(1,20)}"
            for _ in range(n)
        ],
        "Brand Name": np.random.choice(brands, n),
        "Price": prices,
        "Rating": np.random.uniform(1, 5, n).round(1),
        "Reviews": [
            f"Great phone, excellent value. Battery life is {'amazing' if np.random.random() > 0.5 else 'average'}. "
            f"Camera quality is {'superb' if np.random.random() > 0.5 else 'decent'}."
            for _ in range(n)
        ],
        "Review Votes": np.random.randint(0, 500, n),
    })
    df.to_csv(path, index=False)
    print(f"[SYNTHETIC] Created {path} with {n} rows")


def main():
    print("=== Downloading / preparing datasets ===")

    # Flipkart
    if not FLIPKART_CSV.exists():
        ok = download_via_kaggle("PromptCloudHQ/flipkart-products", RAW_DIR)
        if not ok or not FLIPKART_CSV.exists():
            print("[INFO] Using synthetic Flipkart data")
            create_synthetic_flipkart(FLIPKART_CSV)
    else:
        print(f"[SKIP] {FLIPKART_CSV} already exists")

    # Amazon
    if not AMAZON_CSV.exists():
        ok = download_via_kaggle(
            "PromptCloudHQ/amazon-reviews-unlocked-mobile-phones", RAW_DIR
        )
        if not ok or not AMAZON_CSV.exists():
            print("[INFO] Using synthetic Amazon data")
            create_synthetic_amazon(AMAZON_CSV)
    else:
        print(f"[SKIP] {AMAZON_CSV} already exists")

    print("=== Dataset preparation complete ===")


if __name__ == "__main__":
    main()
