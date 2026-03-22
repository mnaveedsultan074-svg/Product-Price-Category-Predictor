#!/usr/bin/env python3
"""
Standalone evaluation script.

Loads the trained best_model.pkl + tfidf_vectorizer.pkl and produces:
  - Full classification report
  - Confusion matrix (saved as PNG)
  - Per-class F1 bar chart (saved as PNG)
  - JSON summary

Usage:
    python evaluate.py
    python evaluate.py --output_dir reports/
"""
import os
import sys
import json
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless rendering
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"

LABEL_MAP  = {"budget": 0, "mid-range": 1, "premium": 2, "luxury": 3}
LABEL_INV  = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = ["budget", "mid-range", "premium", "luxury"]
TIER_COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#8b5cf6"]


def load_artifacts():
    model_path = MODELS_DIR / "best_model.pkl"
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    meta_path  = MODELS_DIR / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, tfidf, meta


def build_features(df, tfidf, meta):
    df["rating"]       = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    df["log_review"]   = np.log1p(df["review_count"])

    struct = df[["rating", "log_review"]].values.astype(float)

    top_cats = meta.get("top_cats", [])
    df["category_clean"] = df["category"].apply(lambda x: x if x in top_cats else "other")
    cat_dummies = pd.get_dummies(df["category_clean"], prefix="cat")

    # Align to training columns
    cat_columns = meta.get("cat_columns", [])
    for col in cat_columns:
        if col not in cat_dummies.columns:
            cat_dummies[col] = 0
    cat_dummies = cat_dummies.reindex(columns=cat_columns, fill_value=0)

    struct_matrix = np.hstack([struct, cat_dummies.values])

    source_dummies = pd.get_dummies(df.get("source", "unknown"), prefix="src")
    struct_matrix = np.hstack([struct_matrix, source_dummies.values.reshape(-1, 1)
                                if source_dummies.shape[1] == 1 else source_dummies.values])

    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()
    text_matrix = tfidf.transform(df["text"])

    return hstack([csr_matrix(struct_matrix), text_matrix])


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved confusion matrix to {output_path}")


def plot_f1_bars(report_dict, output_path):
    classes = [c for c in CLASS_NAMES if c in report_dict]
    f1s     = [report_dict[c]["f1-score"] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(classes, f1s, color=TIER_COLORS[:len(classes)], edgecolor="white", linewidth=1.5,
                  width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score", fontsize=14, fontweight="bold")
    ax.axhline(0.6, color="red", linestyle="--", linewidth=1, label="Target (0.60)")
    ax.legend(fontsize=10)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved F1 bar chart to {output_path}")


def main(output_dir: str = "reports"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    model, tfidf, meta = load_artifacts()
    df = pd.read_csv(PROCESSED_DIR / "products.csv")
    df = df.dropna(subset=["price_tier"])
    df["label"] = df["price_tier"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])

    X = build_features(df.copy(), tfidf, meta)
    y = df["label"].values.astype(int)

    # Use same random split as training so we evaluate on the same test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_wgt = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report_str  = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                        zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    log.info("\n" + "="*60)
    log.info(f"Model: {meta.get('best_model_name', 'unknown')}")
    log.info(f"Accuracy:    {acc:.4f}")
    log.info(f"F1-Macro:    {f1_mac:.4f}")
    log.info(f"F1-Weighted: {f1_wgt:.4f}")
    log.info("\n" + report_str)

    # ROC-AUC (one-vs-rest) if predict_proba available
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba  = model.predict_proba(X_test)
            y_test_b = label_binarize(y_test, classes=[0, 1, 2, 3])
            roc_auc  = roc_auc_score(y_test_b, y_proba, multi_class="ovr", average="macro")
            log.info(f"ROC-AUC (macro OvR): {roc_auc:.4f}")
        except Exception as e:
            log.warning(f"ROC-AUC computation failed: {e}")

    # Save plots
    plot_confusion_matrix(cm, out / "confusion_matrix.png")
    plot_f1_bars(report_dict, out / "f1_per_class.png")

    # Save JSON summary
    summary = {
        "model":       meta.get("best_model_name", "unknown"),
        "accuracy":    round(acc, 4),
        "f1_macro":    round(f1_mac, 4),
        "f1_weighted": round(f1_wgt, 4),
        "roc_auc":     round(roc_auc, 4) if roc_auc else None,
        "per_class":   {k: {m: round(v, 4) for m, v in vs.items()}
                        for k, vs in report_dict.items()
                        if isinstance(vs, dict)},
        "confusion_matrix": cm.tolist(),
    }
    summary_path = out / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Evaluation summary saved to {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained price tier model")
    parser.add_argument("--output_dir", default="reports",
                        help="Directory for report files (default: reports/)")
    args = parser.parse_args()
    main(output_dir=args.output_dir)
