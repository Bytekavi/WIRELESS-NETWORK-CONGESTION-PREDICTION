"""
Wireless Network Congestion Prediction
Model Training & Evaluation Script
Author: Academic Machine Learning Project
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Configuration and Paths
DATA_DIR = "data"
MODELS_DIR = "models"
DATA_FILE = os.path.join(DATA_DIR, "network_congestion.csv")
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "congestion_model.joblib")
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")
COMPARISON_FILE = "model_comparison.csv"

FEATURE_COLUMNS = [
    "network_load_pct",
    "throughput_mbps",
    "latency_ms",
    "packet_loss_pct"
]
TARGET_COLUMN = "congestion_level"
TARGET_CLASSES = ["Low", "Medium", "High"]


def generate_synthetic_dataset(filepath=DATA_FILE, n_samples=3000, random_state=42):
    """
    Generates a realistic synthetic dataset representing wireless network traffic.
    Simulates radio access network features under Low, Medium, and High congestion conditions.
    """
    np.random.seed(random_state)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"[*] Generating {n_samples} synthetic wireless network records...")

    # Class distribution: ~35% Low, ~35% Medium, ~30% High
    n_low = int(n_samples * 0.35)
    n_med = int(n_samples * 0.35)
    n_high = n_samples - n_low - n_med

    # 1. Low Congestion Conditions
    # Low utilization, high throughput, low delay, near-zero loss
    load_low = np.random.uniform(5.0, 42.0, n_low)
    tp_low = np.random.normal(75.0, 15.0, n_low).clip(40.0, 140.0)
    lat_low = np.random.normal(22.0, 6.0, n_low).clip(8.0, 42.0)
    loss_low = np.random.exponential(0.3, n_low).clip(0.0, 1.2)

    # 2. Medium Congestion Conditions
    # Moderate load, intermediate throughput, increased latency, slight packet loss
    load_med = np.random.uniform(43.0, 74.0, n_med)
    tp_med = np.random.normal(36.0, 9.0, n_med).clip(18.0, 60.0)
    lat_med = np.random.normal(65.0, 14.0, n_med).clip(40.0, 105.0)
    loss_med = np.random.exponential(1.4, n_med).clip(0.8, 4.5)

    # 3. High Congestion Conditions
    # Near-saturation load, degraded throughput, severe bufferbloat/latency, notable packet loss
    load_high = np.random.uniform(75.0, 99.5, n_high)
    tp_high = np.random.normal(10.0, 5.0, n_high).clip(0.5, 24.0)
    lat_high = np.random.normal(160.0, 45.0, n_high).clip(95.0, 380.0)
    loss_high = np.random.exponential(4.8, n_high).clip(3.5, 22.0)

    # Introduce realistic boundary variance / noise (5% overlapping samples)
    # This ensures models learn genuine statistical boundaries rather than hard trivial cutoffs
    def inject_jitter(arr, scale):
        noise = np.random.normal(0, scale, len(arr))
        return arr + noise

    load_low = inject_jitter(load_low, 1.5).clip(0.0, 100.0)
    load_med = inject_jitter(load_med, 2.0).clip(0.0, 100.0)
    load_high = inject_jitter(load_high, 2.0).clip(0.0, 100.0)

    tp_low = inject_jitter(tp_low, 2.0).clip(1.0, 150.0)
    tp_med = inject_jitter(tp_med, 2.0).clip(1.0, 150.0)
    tp_high = inject_jitter(tp_high, 1.5).clip(0.1, 150.0)

    lat_low = inject_jitter(lat_low, 1.5).clip(5.0, 500.0)
    lat_med = inject_jitter(lat_med, 3.0).clip(5.0, 500.0)
    lat_high = inject_jitter(lat_high, 5.0).clip(5.0, 500.0)

    loss_low = inject_jitter(loss_low, 0.1).clip(0.0, 100.0)
    loss_med = inject_jitter(loss_med, 0.2).clip(0.0, 100.0)
    loss_high = inject_jitter(loss_high, 0.5).clip(0.0, 100.0)

    # Combine arrays
    load = np.concatenate([load_low, load_med, load_high])
    tp = np.concatenate([tp_low, tp_med, tp_high])
    lat = np.concatenate([lat_low, lat_med, lat_high])
    loss = np.concatenate([loss_low, loss_med, loss_high])
    labels = (
        ["Low"] * n_low +
        ["Medium"] * n_med +
        ["High"] * n_high
    )

    df = pd.DataFrame({
        "network_load_pct": np.round(load, 2),
        "throughput_mbps": np.round(tp, 2),
        "latency_ms": np.round(lat, 2),
        "packet_loss_pct": np.round(loss, 2),
        "congestion_level": labels
    })

    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    df.to_csv(filepath, index=False)
    print(f"[+] Synthetic dataset created and saved to: {filepath}")
    print(f"    Total records: {len(df)}")
    print(f"    Class distribution:\n{df['congestion_level'].value_counts().to_string()}\n")
    return df


def load_or_create_data(filepath=DATA_FILE):
    """Loads existing dataset or triggers generation if missing."""
    if not os.path.exists(filepath):
        print(f"[!] Dataset not found at '{filepath}'. Auto-generating synthetic data...")
        return generate_synthetic_dataset(filepath)
    else:
        print(f"[*] Loading existing dataset from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"    Loaded {len(df)} records.")
        return df


def train_and_evaluate(df):
    """
    Trains Decision Tree, Random Forest, and Gradient Boosting.
    Evaluates on held-out test split, prints metrics, and exports comparison.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Validate columns
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 80/20 Train/Test Split (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[*] Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"[*] Target classes: {TARGET_CLASSES}\n")

    # Candidate Algorithms
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=6,
            min_samples_leaf=4,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
    }

    comparison_records = []
    trained_pipelines = {}
    reports = {}
    confusion_matrices = {}

    print("=" * 78)
    print("                MACHINE LEARNING MODEL EVALUATION RESULTS                ")
    print("=" * 78)

    for name, clf in models.items():
        print(f"\n---> Training: {name}")

        # Construct Scikit-learn Pipeline with median imputation
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=TARGET_CLASSES)
        clf_report = classification_report(
            y_test, y_pred, labels=TARGET_CLASSES, output_dict=True, zero_division=0
        )

        trained_pipelines[name] = pipeline
        reports[name] = clf_report
        confusion_matrices[name] = cm.tolist()

        # Print detailed report
        print(f"Test Accuracy    : {acc * 100:.2f}%")
        print(f"Macro Precision  : {prec_macro:.4f}")
        print(f"Macro Recall     : {rec_macro:.4f}")
        print(f"Macro F1-Score   : {f1_macro:.4f}")
        print(f"Weighted F1-Score: {f1_weighted:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, labels=TARGET_CLASSES, zero_division=0))
        print("Confusion Matrix (rows: True, cols: Predicted):")
        print(f"Labels: {TARGET_CLASSES}")
        print(cm)
        print("-" * 78)

        # Extract feature importances if available
        feature_importances = {}
        if hasattr(pipeline.named_steps["classifier"], "feature_importances_"):
            importances = pipeline.named_steps["classifier"].feature_importances_
            feature_importances = {
                col: round(float(imp), 4) for col, imp in zip(FEATURE_COLUMNS, importances)
            }

        comparison_records.append({
            "Algorithm": name,
            "Accuracy": round(float(acc), 4),
            "Accuracy_pct": round(float(acc * 100), 2),
            "Precision_Macro": round(float(prec_macro), 4),
            "Recall_Macro": round(float(rec_macro), 4),
            "F1_Macro": round(float(f1_macro), 4),
            "F1_Weighted": round(float(f1_weighted), 4),
            "Feature_Importances": feature_importances
        })

    # Model comparison DataFrame
    comp_df = pd.DataFrame(comparison_records)
    comp_df_to_save = comp_df.drop(columns=["Feature_Importances"])
    comp_df_to_save.to_csv(COMPARISON_FILE, index=False)
    print(f"\n[+] Model comparison saved to: {COMPARISON_FILE}")

    # Select Best Model based on Test Accuracy (and Macro F1 as tie-breaker)
    best_record = comp_df.sort_values(
        by=["Accuracy", "F1_Macro"], ascending=[False, False]
    ).iloc[0]
    best_model_name = best_record["Algorithm"]
    best_pipeline = trained_pipelines[best_model_name]

    print("=" * 78)
    print(f"[*] BEST PERFORMING MODEL: {best_model_name}")
    print(f"    Accuracy: {best_record['Accuracy_pct']}% | Macro F1: {best_record['F1_Macro']}")
    print("=" * 78)

    # Bundle model and metadata for saving
    model_payload = {
        "pipeline": best_pipeline,
        "model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_classes": TARGET_CLASSES,
        "metrics": {
            "accuracy": best_record["Accuracy"],
            "accuracy_pct": best_record["Accuracy_pct"],
            "precision_macro": best_record["Precision_Macro"],
            "recall_macro": best_record["Recall_Macro"],
            "f1_macro": best_record["F1_Macro"],
            "f1_weighted": best_record["F1_Weighted"]
        },
        "all_models_summary": comparison_records
    }

    joblib.dump(model_payload, BEST_MODEL_FILE)
    print(f"[+] Best model pipeline saved to: {BEST_MODEL_FILE}")

    # Save metadata JSON for web dashboard
    metadata = {
        "best_model_name": best_model_name,
        "target_classes": TARGET_CLASSES,
        "feature_columns": FEATURE_COLUMNS,
        "comparison": comparison_records,
        "confusion_matrices": confusion_matrices,
        "reports": reports
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Model metadata exported to: {METADATA_FILE}\n")

    return best_model_name, comp_df


def main():
    parser = argparse.ArgumentParser(
        description="Train & Compare Machine Learning Models for Network Congestion Prediction"
    )
    parser.add_argument(
        "--regenerate", action="store_true", help="Force regenerate synthetic dataset"
    )
    args = parser.parse_args()

    if args.regenerate and os.path.exists(DATA_FILE):
        print(f"[*] Regenerating dataset at: {DATA_FILE}")
        generate_synthetic_dataset(DATA_FILE)

    df = load_or_create_data(DATA_FILE)
    train_and_evaluate(df)
    print("[v] Model training & evaluation completed successfully!")


if __name__ == "__main__":
    main()
