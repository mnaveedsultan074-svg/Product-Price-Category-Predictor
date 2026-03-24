#!/usr/bin/env python3
"""
Preprocessing pipeline for price category prediction.
Loads Flipkart and Amazon datasets, normalises columns,
assigns price tier labels, and saves processed data.
"""
import re
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_DIR       = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PRICE_BINS   = [0, 25, 50, 75, 100]
PRICE_LABELS = ["budget", "mid-range", "premium", "luxury"]


def clean_price(val):
    """Extract numeric price from various formats."""
    if pd.isna(val):
        return np.nan
    val = str(val).replace(",", "").replace("£", "").replace("$", "").replace("₹", "").strip()
    match = re.search(r"[\d.]+", val)
    return float(match.group()) if match else np.nan


def assign_price_tier(series: pd.Series) -> pd.Series:
    """Assign price tier based on percentiles of the series."""
    percentiles = np.percentile(series.dropna(), PRICE_BINS)
    # Ensure unique bin edges
    unique_pcts = sorted(set(percentiles))
    if len(unique_pcts) < 5:
        # Fall back to equal-width labels
        return pd.qcut(series, q=4, labels=PRICE_LABELS, duplicates="drop")
    return pd.cut(
        series,
        bins=[percentiles[0] - 0.01] + list(percentiles[1:]),
        labels=PRICE_LABELS,
        include_lowest=True,
    )


def load_flipkart(path: Path) -> pd.DataFrame:
    log.info(f"Loading Flipkart data from {path}")
    df = pd.read_csv(path, low_memory=False)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Price: prefer discounted_price, fall back to retail_price
    price_col = "discounted_price" if "discounted_price" in df.columns else "retail_price"
    df["price"] = df[price_col].apply(clean_price)

    # Rating
    rating_col = "product_rating" if "product_rating" in df.columns else "overall_rating"
    df["rating"] = pd.to_numeric(df.get(rating_col, np.nan), errors="coerce")

    # Title / description
    df["title"] = df.get("product_name", "").fillna("").astype(str)
    df["description"] = df.get("description", "").fillna("").astype(str)

    # Category
    def extract_top_category(tree_str):
        if pd.isna(tree_str):
            return "unknown"
        # format: "Electronics >> Mobiles >> Smartphones"
        parts = str(tree_str).split(">>")
        return parts[0].strip().lower() if parts else "unknown"

    df["category"] = df.get("product_category_tree", "").apply(extract_top_category)
    df["review_count"] = 0  # not available
    df["source"] = "flipkart"

    return df[["price", "rating", "review_count", "title", "description", "category", "source"]]


def load_amazon(path: Path) -> pd.DataFrame:
    log.info(f"Loading Amazon data from {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    df["price"] = df["Price"].apply(clean_price)
    df["rating"] = pd.to_numeric(df.get("Rating", np.nan), errors="coerce")
    df["review_count"] = pd.to_numeric(df.get("Review Votes", 0), errors="coerce").fillna(0)
    df["title"] = df.get("Product Name", "").fillna("").astype(str)
    df["description"] = df.get("Reviews", "").fillna("").astype(str)
    df["category"] = "mobile phones"
    df["source"] = "amazon"

    return df[["price", "rating", "review_count", "title", "description", "category", "source"]]


def main():
    flipkart_path = RAW_DIR / "flipkart_com-ecommerce_sample.csv"
    amazon_path   = RAW_DIR / "Amazon_Unlocked_Mobile.csv"

    frames = []
    if flipkart_path.exists():
        frames.append(load_flipkart(flipkart_path))
    if amazon_path.exists():
        frames.append(load_amazon(amazon_path))

    if not frames:
        raise FileNotFoundError("No raw data found. Run download_data.py first.")

    df = pd.concat(frames, ignore_index=True)
    log.info(f"Combined dataset: {len(df)} rows")

    # Drop rows without price
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    log.info(f"After price filter: {len(df)} rows")

    # Assign price tier
    df["price_tier"] = assign_price_tier(df["price"])
    df = df.dropna(subset=["price_tier"])

    # Fill remaining nulls
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["review_count"] = df["review_count"].fillna(0)
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["category"] = df["category"].fillna("unknown")

    # Save
    out_path = PROCESSED_DIR / "products.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved processed data to {out_path}")

    # Save tier distribution
    dist = df["price_tier"].value_counts().to_dict()
    log.info(f"Price tier distribution: {dist}")

    # Save percentile thresholds for later use in prediction
    thresholds = {
        "p25": float(np.percentile(df["price"], 25)),
        "p50": float(np.percentile(df["price"], 50)),
        "p75": float(np.percentile(df["price"], 75)),
        "labels": PRICE_LABELS,
    }
    with open(PROCESSED_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"Saved thresholds: {thresholds}")

    return df


if __name__ == "__main__":
    main()
