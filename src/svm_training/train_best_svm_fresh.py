#!/usr/bin/env python3
"""
Train Best SVM Model from Scratch
===================================
Trains a NEW SVM model from scratch using the best hyperparameters found during optimization.
Uses feature-engineered data with optimized hyperparameters and thresholds.

Saves:
- output/svm_best_model.pkl (everything in one file: model, scaler, config, features, thresholds)

Output: submission_svm_fresh_trained.csv
"""

# Import the SVM implementation
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
from svm import SVMOneVsAll


def load_and_prepare_data():
    """Load and prepare training data with feature engineering."""
    print("Loading training data...")
    train_df = pd.read_csv("data/train.csv")

    print("Applying feature engineering...")
    train_engineered = engineer_all_features(train_df)

    # Best selected features from optimization
    selected_features = [
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
        "Age at enrollment",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "pass_rate_1st",
        "pass_rate_2nd",
        "failed_1st",
        "failed_2nd",
        "academic_risk_score",
        "consistent_performance",
        "financial_burden",
        "economic_vulnerability",
        "financial_stability",
        "academic_economic_stress",
        "age_performance_risk",
        "grade_financial_stress",
        "displacement_academic_risk",
        "moderate_performance",
        "moderate_academic_risk",
        "has_financial_support",
        "enrolled_middle_ground",
    ]

    X_train = train_engineered[selected_features].values
    y_train = train_engineered["Target"].values

    # Standardize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train = (X_train - mean) / std

    return X_train, y_train, mean, std, selected_features


def train_best_model(X_train, y_train):
    """Train SVM with best hyperparameters."""
    print("Training SVM with best hyperparameters...")
    print("  C=20.0, lr=0.05, n_iter=1500")

    # Best hyperparameters from optimization
    model = SVMOneVsAll(C=20.0, lr=0.05, n_iter=1500)

    model.fit(X_train, y_train)
    return model


def save_model_and_config(model, selected_features, scaler_mean, scaler_std):
    """Save everything to a single pkl file."""
    print("\nSaving model (with scaler and config)...")

    # Save everything in one pkl file
    model_data = {
        "model": model,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "selected_features": selected_features,
        "thresholds": {"Dropout": -4.2, "Enrolled": -3.4, "Graduate": -4.2},
        "hyperparameters": {"C": 20.0, "lr": 0.05, "n_iter": 1500},
        "note": "Trained with train_best_svm_fresh.py",
    }

    model_path = "output/svm_best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  ✓ Everything saved to: {model_path}")


def predict_with_thresholds(model, X, class_names):
    """Predict using optimized thresholds."""
    # Best thresholds from optimization
    thresholds = {"Dropout": -4.2, "Enrolled": -3.4, "Graduate": -4.2}

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
    print("TRAINING FRESH SVM MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 60)

    # Load and prepare data
    X_train, y_train, mean, std, selected_features = load_and_prepare_data()

    # Train model
    model = train_best_model(X_train, y_train)

    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = np.mean(train_pred == y_train)
    print(f"\nTraining Accuracy: {train_acc:.4f}")

    # Save model, config, and scaler
    save_model_and_config(model, selected_features, mean, std)

    # Load and prepare test data
    print("\nLoading test data...")
    test_df = pd.read_csv("data/test.csv")
    test_ids = test_df["Student_ID"].copy()  # Preserve IDs before feature engineering
    test_engineered = engineer_all_features(test_df)
    X_test = test_engineered[selected_features].values

    # Standardize using training statistics
    X_test = (X_test - mean) / std

    # Predict with optimized thresholds
    print("Generating predictions with optimized thresholds...")
    class_names = ["Dropout", "Enrolled", "Graduate"]
    predictions = predict_with_thresholds(model, X_test, class_names)

    # Create submission
    submission_df = pd.DataFrame({"Student_ID": test_ids, "Target": predictions})

    output_path = "output/submission_svm_fresh_trained.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")

    # Display prediction distribution
    pred_counts = pd.Series(predictions).value_counts()
    print("\nPrediction distribution:")
    total = len(predictions)
    for class_name in class_names:
        count = pred_counts.get(class_name, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"  {class_name}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("DONE!")
    print("\nSaved files:")
    print("  • output/svm_best_model.pkl (contains everything)")
    print("  • output/submission_svm_fresh_trained.csv")
    print("\nYou can now use predict_with_saved_svm.py to load this model!")
    print("=" * 60)


if __name__ == "__main__":
    main()
