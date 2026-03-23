#!/usr/bin/env python3
"""
Flask REST API for product price tier prediction.
Endpoints:
  POST /predict  - predict price tier for a product
  GET  /health   - health check
  GET  /models   - list available models and their performance
"""
import os
import json
import logging
import traceback
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.sparse import hstack, csr_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODELS_DIR    = Path(__file__).parent / "models"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

# Global model state
MODEL      = None
TFIDF      = None
METADATA   = None
SHAP_EXPLAINER = None
THRESHOLDS = None

LABEL_INV = {0: "budget", 1: "mid-range", 2: "premium", 3: "luxury"}
TIER_COLORS = {
    "budget":    "#22c55e",
    "mid-range": "#3b82f6",
    "premium":   "#f59e0b",
    "luxury":    "#8b5cf6",
}
TIER_DESCRIPTIONS = {
    "budget":    "Entry-level pricing — great value for money, accessible to most consumers.",
    "mid-range": "Moderate pricing — balanced features and cost, targeting mainstream consumers.",
    "premium":   "Above-average pricing — enhanced features or brand positioning.",
    "luxury":    "Top-tier pricing — exclusive features, premium brand, or niche market.",
}


def load_artifacts():
    """Load all model artifacts on startup."""
    global MODEL, TFIDF, METADATA, SHAP_EXPLAINER, THRESHOLDS

    model_path = MODELS_DIR / "best_model.pkl"
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    meta_path  = MODELS_DIR / "metadata.json"
    shap_path  = MODELS_DIR / "shap_explainer.pkl"
    thresh_path = PROCESSED_DIR / "thresholds.json"

    if not model_path.exists():
        log.error(f"Model not found at {model_path}. Run train.py first.")
        return False

    MODEL    = joblib.load(model_path)
    TFIDF    = joblib.load(tfidf_path)

    with open(meta_path) as f:
        METADATA = json.load(f)

    if shap_path.exists():
        try:
            SHAP_EXPLAINER = joblib.load(shap_path)
        except Exception as e:
            log.warning(f"Could not load SHAP explainer: {e}")

    if thresh_path.exists():
        with open(thresh_path) as f:
            THRESHOLDS = json.load(f)

    log.info(f"Loaded model: {METADATA.get('best_model_name', 'unknown')}")
    log.info(f"F1-macro: {METADATA.get('f1_macro', 'N/A')}")
    return True


def build_feature_vector(title, description, rating, review_count, category, price=None, source=None):
    """Build a single feature vector for prediction (must match train.py build_features)."""
    top_cats       = METADATA.get("top_cats", [])
    cat_columns    = METADATA.get("cat_columns", [])
    source_columns = METADATA.get("source_columns", [])

    # Structured features
    rating       = float(rating or 0)
    review_count = float(review_count or 0)
    log_review   = np.log1p(review_count)

    struct = [rating, log_review]

    # Category one-hot
    category_clean = category.lower() if category.lower() in top_cats else "other"
    for col in cat_columns:
        expected_cat = col.replace("cat_", "")
        struct.append(1.0 if category_clean == expected_cat else 0.0)

    # Source one-hot (same columns as training); unknown / missing → all zeros
    source_norm = (source or "").strip().lower()
    if source_columns:
        for col in source_columns:
            suffix = col.replace("src_", "")
            struct.append(1.0 if source_norm == suffix else 0.0)
    else:
        struct.append(0.0)

    struct_matrix = csr_matrix(np.array(struct).reshape(1, -1))

    # TF-IDF
    text = (str(title) + " " + str(description)).lower()
    text_matrix = TFIDF.transform([text])

    X = hstack([struct_matrix, text_matrix])
    return X


def get_shap_values(X, feature_names=None):
    """Compute SHAP values for a prediction."""
    if SHAP_EXPLAINER is None:
        return None
    try:
        import shap
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X

        shap_values = SHAP_EXPLAINER.shap_values(X_dense)

        # Handle multi-class output
        if isinstance(shap_values, list):
            # List of arrays per class
            pred_class = int(MODEL.predict(X)[0])
            sv = shap_values[pred_class][0]
        else:
            sv = shap_values[0]

        # Build feature importance dict (top 10)
        if feature_names and len(feature_names) == len(sv):
            pairs = sorted(zip(feature_names, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)
            top_features = [{"feature": f, "shap_value": round(v, 4)} for f, v in pairs[:10]]
        else:
            idx = np.argsort(np.abs(sv))[::-1][:10]
            top_features = [{"feature": f"feature_{i}", "shap_value": round(float(sv[i]), 4)} for i in idx]

        return top_features
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")
        return None


@app.route("/health", methods=["GET"])
def health():
    if MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({
        "status": "ok",
        "model": METADATA.get("best_model_name", "unknown"),
        "f1_macro": METADATA.get("f1_macro", None),
    })


@app.route("/models", methods=["GET"])
def list_models():
    if METADATA is None:
        return jsonify({"error": "Metadata not loaded"}), 503
    return jsonify({
        "best_model": METADATA.get("best_model_name"),
        "metrics": {
            "f1_macro":    METADATA.get("f1_macro"),
            "accuracy":    METADATA.get("accuracy"),
            "f1_weighted": METADATA.get("f1_weighted"),
        },
        "all_models": METADATA.get("all_results", []),
        "thresholds":  THRESHOLDS,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict price tier for a product.
    Body (JSON):
      title        : str  (required)
      description  : str  (optional)
      rating       : float (optional, 0-5)
      review_count : int  (optional)
      category     : str  (optional)
      price        : float (optional, for reference)
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Run training pipeline first."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    title        = str(data.get("title", ""))
    description  = str(data.get("description", ""))
    rating       = float(data.get("rating", 0) or 0)
    review_count = float(data.get("review_count", 0) or 0)
    category     = str(data.get("category", "unknown"))
    price_input  = data.get("price")
    source_input = data.get("source")

    if not title and not description:
        return jsonify({"error": "At least title or description is required"}), 400

    try:
        X = build_feature_vector(
            title, description, rating, review_count, category, price_input, source=source_input
        )

        # Prediction
        pred_class = int(MODEL.predict(X)[0])
        pred_label = LABEL_INV.get(pred_class, "unknown")

        # Confidence
        proba = None
        confidence_scores = {}
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)[0]
            confidence_scores = {
                LABEL_INV[i]: round(float(p) * 100, 1)
                for i, p in enumerate(proba)
                if i in LABEL_INV
            }

        # SHAP
        shap_features = get_shap_values(X)

        # Build human-readable explanation
        top_driver = ""
        if shap_features:
            top = shap_features[0]
            direction = "increases" if top["shap_value"] > 0 else "decreases"
            top_driver = f"The feature '{top['feature']}' {direction} the prediction most strongly."

        response = {
            "prediction": pred_label,
            "prediction_code": pred_class,
            "confidence_scores": confidence_scores,
            "confidence_pct": round(float(proba[pred_class]) * 100, 1) if proba is not None else None,
            "tier_color": TIER_COLORS.get(pred_label, "#6b7280"),
            "tier_description": TIER_DESCRIPTIONS.get(pred_label, ""),
            "shap_features": shap_features or [],
            "explanation": top_driver,
            "model_used": METADATA.get("best_model_name", "unknown"),
            "thresholds": THRESHOLDS,
        }
        return jsonify(response)

    except Exception as e:
        log.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if not load_artifacts():
        log.error("Failed to load model artifacts. Start training first.")
        # Don't exit -- let Flask run so Rails gets a 503 response it can handle
    port = int(os.environ.get("ML_PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
