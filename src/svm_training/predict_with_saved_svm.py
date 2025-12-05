#!/usr/bin/env python3
"""
Predict with Pre-trained (Saved) SVM Model
============================================
Loads the pre-trained SVM model from pkl file and generates predictions.
This ensures we use the exact same model that was previously trained and optimized.

NOTE: This script requires a pre-trained model pkl file to exist.
      If you don't have one, use train_and_save_best_svm.py to train and save a new model.

Uses:
- output/svm_best_model.pkl (contains everything: model, scaler, config, features, thresholds)

Output: submission_svm_pretrained.csv
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / "src"))

from feature_engineering import engineer_all_features


def load_model_data():
    """Load everything from a single pkl file."""
    print("Loading saved model...")

    pkl_path = "output/svm_best_model.pkl"

    try:
        with open(pkl_path, "rb") as f:
            model_data = pickle.load(f)
        print(f"  ✓ Loaded from: {pkl_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find model file: {pkl_path}\n"
            "Please train a model first using train_best_svm_fresh.py"
        )

    # Check if it's the new format (dictionary) or old format (just model object)
    if isinstance(model_data, dict):
        # New format - everything in one dict
        model = model_data["model"]
        scaler_mean = model_data["scaler_mean"]
        scaler_std = model_data["scaler_std"]
        selected_features = model_data["selected_features"]
        thresholds = model_data["thresholds"]
        hyperparams = model_data.get("hyperparameters", {})

        print(
            f"  Hyperparameters: C={hyperparams.get('C')}, lr={hyperparams.get('lr')}, n_iter={hyperparams.get('n_iter')}"
        )
        print(f"  Features: {len(selected_features)}")
        print(
            f"  Thresholds: Dropout={thresholds['Dropout']}, Enrolled={thresholds['Enrolled']}, Graduate={thresholds['Graduate']}"
        )

        return model, selected_features, thresholds, scaler_mean, scaler_std
    else:
        # Old format - just the model, need to compute scaler
        raise ValueError(
            "Old model format detected. Please retrain using train_best_svm_fresh.py"
        )


def prepare_test_data(selected_features, scaler_mean, scaler_std):
    """Load and prepare test data with feature engineering."""
    print("\nLoading test data...")
    test_df = pd.read_csv("data/test.csv")
    test_ids = test_df["Student_ID"].copy()

    print("Applying feature engineering...")
    test_engineered = engineer_all_features(test_df)

    # Select features
    X_test = test_engineered[selected_features].values

    # Standardize using training statistics
    X_test = (X_test - scaler_mean) / scaler_std

    return X_test, test_ids


def predict_with_thresholds(model, X, thresholds):
    """Generate predictions using optimized thresholds."""
    print("\nGenerating predictions with optimized thresholds...")
    print(
        f"  Thresholds: Dropout={thresholds['Dropout']}, "
        f"Enrolled={thresholds['Enrolled']}, Graduate={thresholds['Graduate']}"
    )

    class_names = ["Dropout", "Enrolled", "Graduate"]

    # Get decision scores
    scores = model.decision_function(X)
    predictions = []

    for i in range(len(X)):
        sample_scores = scores[i]
        adjusted_scores = []

        for j, class_name in enumerate(class_names):
            adjusted_score = sample_scores[j] - thresholds[class_name]
            adjusted_scores.append(adjusted_score)

        pred_class_idx = np.argmax(adjusted_scores)
        predictions.append(class_names[pred_class_idx])

    return predictions


def main():
    """Main execution function."""
    print("=" * 60)
    print("PREDICTING WITH PRE-TRAINED (SAVED) SVM MODEL")
    print("=" * 60)

    # Load everything from single pkl file
    model, selected_features, thresholds, scaler_mean, scaler_std = load_model_data()

    # Prepare test data
    X_test, test_ids = prepare_test_data(selected_features, scaler_mean, scaler_std)

    # Generate predictions
    predictions = predict_with_thresholds(model, X_test, thresholds)

    # Create submission
    submission_df = pd.DataFrame({"Student_ID": test_ids, "Target": predictions})

    output_path = "output/submission_svm_pretrained.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")

    # Display prediction distribution
    pred_counts = pd.Series(predictions).value_counts()
    print("\nPrediction distribution:")
    class_names = ["Dropout", "Enrolled", "Graduate"]
    total = len(predictions)
    for class_name in class_names:
        count = pred_counts.get(class_name, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"  {class_name}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
