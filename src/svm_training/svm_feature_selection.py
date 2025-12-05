#!/usr/bin/env python3
"""
SVM-Only Feature Selection Optimizer
=====================================
This script finds the OPTIMAL subset of features for a SVM model.

Strategies:
1. Univariate feature selection (f_classif, mutual_info)
2. Correlation-based feature removal
3. L1-based feature selection (sparse models)
4. Feature importance from RF
5. Tests different feature counts
6. Cross-validation to avoid overfitting

Output files (separate from ensemble):
- submission_svm_only_feature_selected.csv
- svm_only_feature_selection_results.json
- svm_only_selected_features.txt
"""

import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / "src"))

from feature_engineering import engineer_all_features
from svm import SVMOneVsAll

warnings.filterwarnings("ignore")


def load_and_engineer_data(train_path, test_path):
    """Load data and apply feature engineering."""
    print("=" * 80)
    print("LOADING AND ENGINEERING FEATURES")
    print("=" * 80)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")

    y_train_raw = train_df["Target"].values
    test_ids = test_df["Student_ID"].values

    train_features = train_df.drop(columns=["Target"])
    if "Student_ID" in train_features.columns:
        train_features = train_features.drop(columns=["Student_ID"])

    test_features = test_df.copy()
    if "Student_ID" in test_features.columns:
        test_features = test_features.drop(columns=["Student_ID"])

    print(f"\nOriginal features: {train_features.shape[1]}")

    print("\nApplying feature engineering...")
    train_engineered = engineer_all_features(train_features)
    test_engineered = engineer_all_features(test_features)

    print(f"Engineered features: {train_engineered.shape[1]}")

    # Get feature names
    feature_names = train_engineered.columns.tolist()

    # Encode target
    unique_labels = np.unique(y_train_raw)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    y_train_encoded = np.array([label_to_int[label] for label in y_train_raw])

    print(f"\nTarget distribution:")
    for label in unique_labels:
        count = np.sum(y_train_raw == label)
        pct = 100 * count / len(y_train_raw)
        print(f"  {label}: {count} ({pct:.1f}%)")

    return (
        train_engineered,
        y_train_encoded,
        test_engineered,
        test_ids,
        label_to_int,
        int_to_label,
        feature_names,
    )


def remove_correlated_features(X, feature_names, threshold=0.95):
    """Remove highly correlated features."""
    print(f"\nRemoving features with correlation > {threshold}...")

    X_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = X_df.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []
    for column in upper.columns:
        if any(upper[column] > threshold):
            to_drop.append(column)

    print(f"  Found {len(to_drop)} highly correlated features to remove")

    if to_drop:
        print(
            f"  Removing: {to_drop[:10]}..."
            if len(to_drop) > 10
            else f"  Removing: {to_drop}"
        )

    keep_features = [f for f in feature_names if f not in to_drop]

    return keep_features


def select_features_univariate(X, y, feature_names, k=50, method="f_classif"):
    """Select top k features using univariate statistics."""
    print(f"\nSelecting top {k} features using {method}...")

    X_df = pd.DataFrame(X, columns=feature_names)

    if method == "f_classif":
        selector = SelectKBest(f_classif, k=min(k, len(feature_names)))
    elif method == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=min(k, len(feature_names)))
    else:
        raise ValueError(f"Unknown method: {method}")

    selector.fit(X_df, y)

    mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_names, mask) if m]

    scores = selector.scores_
    feature_scores = sorted(
        zip(feature_names, scores), key=lambda x: x[1], reverse=True
    )

    print(f"  Selected {len(selected_features)} features")
    print(f"  Top 10 by score:")
    for feat, score in feature_scores[:10]:
        print(f"    {feat}: {score:.2f}")

    return selected_features


def select_features_tree_importance(X, y, feature_names, k=50):
    """Select top k features using Random Forest importance."""
    print(f"\nSelecting top {k} features using Random Forest importance...")

    X_df = pd.DataFrame(X, columns=feature_names)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_df, y)

    importances = rf.feature_importances_
    feature_importance = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )

    selected_features = [f for f, _ in feature_importance[:k]]

    print(f"  Selected {len(selected_features)} features")
    print(f"  Top 10 by importance:")
    for feat, imp in feature_importance[:10]:
        print(f"    {feat}: {imp:.4f}")

    return selected_features


def evaluate_feature_subset_cv(
    X, y, feature_names, selected_features, svm_params, cv_folds=5
):
    """
    Evaluate feature subset using cross-validation.
    More robust than single train/val split.
    """
    # Get feature indices
    feature_indices = [feature_names.index(f) for f in selected_features]
    X_selected = X[:, feature_indices]

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_f1_scores = []
    cv_acc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y), 1):
        X_train_fold = X_selected[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X_selected[val_idx]
        y_val_fold = y[val_idx]

        # Train SVM
        model = SVMOneVsAll(**svm_params)
        model.fit(X_train_fold, y_train_fold)

        # Evaluate
        y_pred = model.predict(X_val_fold)
        f1 = f1_score(y_val_fold, y_pred, average="macro")
        acc = accuracy_score(y_val_fold, y_pred)

        cv_f1_scores.append(f1)
        cv_acc_scores.append(acc)

    mean_f1 = np.mean(cv_f1_scores)
    std_f1 = np.std(cv_f1_scores)
    mean_acc = np.mean(cv_acc_scores)

    return mean_f1, std_f1, mean_acc


def optimize_feature_selection_cv(X, y, feature_names, svm_params):
    """
    Test multiple feature selection strategies with CV.
    """
    print("\n" + "=" * 80)
    print("FEATURE SELECTION OPTIMIZATION (Cross-Validated)")
    print("=" * 80)

    results = []

    # Strategy 1: Remove highly correlated features
    print("\n[Strategy 1] Remove Highly Correlated Features")
    print("-" * 80)
    uncorrelated_features = remove_correlated_features(X, feature_names, threshold=0.95)

    mean_f1, std_f1, mean_acc = evaluate_feature_subset_cv(
        X, y, feature_names, uncorrelated_features, svm_params, cv_folds=5
    )
    print(f"Result: F1={mean_f1:.4f}±{std_f1:.4f}, Acc={mean_acc:.4f}")

    results.append(
        {
            "strategy": "remove_correlated",
            "n_features": len(uncorrelated_features),
            "features": uncorrelated_features,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_acc": mean_acc,
        }
    )

    # Get uncorrelated data for further selection
    uncorr_indices = [feature_names.index(f) for f in uncorrelated_features]
    X_uncorr = X[:, uncorr_indices]

    # Strategy 2: Univariate selection (F-test) on uncorrelated
    print("\n[Strategy 2] Univariate F-test on Uncorrelated Features")
    print("-" * 80)

    for k in [25, 30, 35, 40, 45, 50, 55, 60]:
        if k > len(uncorrelated_features):
            continue

        selected = select_features_univariate(
            X_uncorr, y, uncorrelated_features, k=k, method="f_classif"
        )

        mean_f1, std_f1, mean_acc = evaluate_feature_subset_cv(
            X, y, feature_names, selected, svm_params, cv_folds=5
        )
        print(f"k={k:2d}: F1={mean_f1:.4f}±{std_f1:.4f}, Acc={mean_acc:.4f}")

        results.append(
            {
                "strategy": f"univariate_f_k{k}",
                "n_features": len(selected),
                "features": selected,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_acc": mean_acc,
            }
        )

    # Strategy 3: Mutual Information on uncorrelated
    print("\n[Strategy 3] Mutual Information on Uncorrelated Features")
    print("-" * 80)

    for k in [25, 30, 35, 40, 45, 50, 55, 60]:
        if k > len(uncorrelated_features):
            continue

        selected = select_features_univariate(
            X_uncorr, y, uncorrelated_features, k=k, method="mutual_info"
        )

        mean_f1, std_f1, mean_acc = evaluate_feature_subset_cv(
            X, y, feature_names, selected, svm_params, cv_folds=5
        )
        print(f"k={k:2d}: F1={mean_f1:.4f}±{std_f1:.4f}, Acc={mean_acc:.4f}")

        results.append(
            {
                "strategy": f"mutual_info_k{k}",
                "n_features": len(selected),
                "features": selected,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_acc": mean_acc,
            }
        )

    # Strategy 4: Tree-based importance
    print("\n[Strategy 4] Random Forest Importance on All Features")
    print("-" * 80)

    for k in [25, 30, 35, 40, 45, 50, 55, 60]:
        selected = select_features_tree_importance(X, y, feature_names, k=k)

        mean_f1, std_f1, mean_acc = evaluate_feature_subset_cv(
            X, y, feature_names, selected, svm_params, cv_folds=5
        )
        print(f"k={k:2d}: F1={mean_f1:.4f}±{std_f1:.4f}, Acc={mean_acc:.4f}")

        results.append(
            {
                "strategy": f"tree_importance_k{k}",
                "n_features": len(selected),
                "features": selected,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_acc": mean_acc,
            }
        )

    # Strategy 5: All features (baseline)
    print("\n[Baseline] All Features")
    print("-" * 80)
    mean_f1, std_f1, mean_acc = evaluate_feature_subset_cv(
        X, y, feature_names, feature_names, svm_params, cv_folds=5
    )
    print(f"F1={mean_f1:.4f}±{std_f1:.4f}, Acc={mean_acc:.4f}")

    results.append(
        {
            "strategy": "all_features",
            "n_features": len(feature_names),
            "features": feature_names,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_acc": mean_acc,
        }
    )

    # Sort by mean F1
    results.sort(key=lambda x: x["mean_f1"], reverse=True)

    # Print top results
    print("\n" + "=" * 80)
    print("TOP 15 FEATURE SUBSETS")
    print("=" * 80)

    for i, res in enumerate(results[:15], 1):
        marker = " ★" if i == 1 else ""
        print(
            f"{i:2d}. {res['strategy']:30s} | n={res['n_features']:2d} | "
            f"F1={res['mean_f1']:.4f}±{res['std_f1']:.4f}, Acc={res['mean_acc']:.4f}{marker}"
        )

    best_result = results[0]
    print(f"\n★ BEST: {best_result['strategy']}")
    print(f"   Features: {best_result['n_features']}")
    print(f"   Mean F1: {best_result['mean_f1']:.4f} ± {best_result['std_f1']:.4f}")

    return best_result, results


def optimize_thresholds(model, X_val, y_val):
    """Optimize decision thresholds."""
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)

    scores = model.decision_function(X_val)

    best_f1 = -1
    best_thresholds = np.zeros(3)

    # Coarse search
    threshold_range = np.linspace(-2.0, 2.0, 21)

    print("Coarse search...")
    for t0 in threshold_range[::3]:
        for t1 in threshold_range[::3]:
            for t2 in threshold_range[::3]:
                thresholds = np.array([t0, t1, t2])
                adjusted_scores = scores - thresholds
                y_pred = np.argmax(adjusted_scores, axis=1)
                f1 = f1_score(y_val, y_pred, average="macro")

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = thresholds

    # Fine search
    print("Fine search...")
    fine_range = np.linspace(-0.5, 0.5, 11)

    for dt0 in fine_range:
        for dt1 in fine_range:
            for dt2 in fine_range:
                thresholds = best_thresholds + np.array([dt0, dt1, dt2])
                adjusted_scores = scores - thresholds
                y_pred = np.argmax(adjusted_scores, axis=1)
                f1 = f1_score(y_val, y_pred, average="macro")

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = thresholds

    print(f"\n✓ Optimal thresholds:")
    print(f"  Dropout:  {best_thresholds[0]:7.3f}")
    print(f"  Enrolled: {best_thresholds[1]:7.3f}")
    print(f"  Graduate: {best_thresholds[2]:7.3f}")
    print(f"  F1 Score: {best_f1:.4f}")

    return best_thresholds, best_f1


def evaluate_model(model, X, y, int_to_label, thresholds=None, name="Model"):
    """Evaluate model with optional thresholds."""
    print("\n" + "=" * 80)
    print(f"{name} EVALUATION")
    print("=" * 80)

    if thresholds is not None:
        scores = model.decision_function(X)
        adjusted_scores = scores - thresholds
        y_pred = np.argmax(adjusted_scores, axis=1)
    else:
        y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")

    f1_per_class = f1_score(y, y_pred, average=None)
    print(f"\nPer-class F1:")
    for i, f1_class in enumerate(f1_per_class):
        print(f"  {int_to_label[i]}: {f1_class:.4f}")

    y_labels = [int_to_label[y_int] for y_int in y]
    y_pred_labels = [int_to_label[y_int] for y_int in y_pred]

    print("\nClassification Report:")
    print(classification_report(y_labels, y_pred_labels, digits=4))

    return acc, f1


def make_predictions(
    model, X_test, test_ids, int_to_label, output_path, thresholds=None
):
    """Make predictions with optional thresholds."""
    print("\n" + "=" * 80)
    print("MAKING PREDICTIONS ON TEST SET")
    print("=" * 80)

    if thresholds is not None:
        scores = model.decision_function(X_test)
        adjusted_scores = scores - thresholds
        y_pred_encoded = np.argmax(adjusted_scores, axis=1)
    else:
        y_pred_encoded = model.predict(X_test)

    y_pred_labels = [int_to_label[y] for y in y_pred_encoded]

    submission = pd.DataFrame({"Student_ID": test_ids, "Target": y_pred_labels})

    print(f"\nPrediction distribution:")
    for target, count in submission["Target"].value_counts().items():
        pct = 100 * count / len(submission)
        print(f"  {target}: {count} ({pct:.1f}%)")

    submission.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")

    return submission


def main():
    """Main execution."""

    print("=" * 80)
    print("SVM-ONLY FEATURE SELECTION OPTIMIZER")
    print("=" * 80)

    # Configuration
    TRAIN_PATH = "data/train.csv"
    TEST_PATH = "data/test.csv"
    VAL_SIZE = 0.2
    RANDOM_STATE = 42

    # Best SVM hyperparameters (from previous tuning)
    SVM_PARAMS = {"C": 20.0, "lr": 0.05, "n_iter": 1500}

    print(f"\nSVM Parameters: {SVM_PARAMS}")

    # Load and engineer features
    (
        train_engineered,
        y_train_encoded,
        test_engineered,
        test_ids,
        label_to_int,
        int_to_label,
        feature_names,
    ) = load_and_engineer_data(TRAIN_PATH, TEST_PATH)

    # Split validation set
    print("\n" + "=" * 80)
    print("SPLITTING DATA")
    print("=" * 80)

    X_train, X_val, y_train, y_val = train_test_split(
        train_engineered,
        y_train_encoded,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_encoded,
    )

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")

    # Preprocess
    print("\nPreprocessing features...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Optimize feature selection with cross-validation
    best_result, all_results = optimize_feature_selection_cv(
        X_train_scaled, y_train, feature_names, SVM_PARAMS
    )

    # Train model with selected features on train set
    print("\n" + "=" * 80)
    print("TRAINING MODEL WITH SELECTED FEATURES")
    print("=" * 80)

    feature_indices = [feature_names.index(f) for f in best_result["features"]]
    X_train_selected = X_train_scaled[:, feature_indices]
    X_val_selected = X_val_scaled[:, feature_indices]

    print(f"Training with {len(best_result['features'])} selected features...")
    model = SVMOneVsAll(**SVM_PARAMS)
    model.fit(X_train_selected, y_train)
    print("✓ Training complete")

    # Baseline evaluation (no thresholds)
    evaluate_model(model, X_val_selected, y_val, int_to_label, None, "Baseline")

    # Optimize thresholds
    best_thresholds, threshold_f1 = optimize_thresholds(model, X_val_selected, y_val)

    # Evaluation with thresholds
    evaluate_model(
        model,
        X_val_selected,
        y_val,
        int_to_label,
        best_thresholds,
        "Threshold-Optimized",
    )

    # Retrain on full data
    print("\n" + "=" * 80)
    print("RETRAINING ON FULL DATA WITH SELECTED FEATURES")
    print("=" * 80)

    # Preprocess full data
    X_full_scaled = scaler.fit_transform(train_engineered)
    X_test_scaled = scaler.transform(test_engineered)

    # Select features
    X_full_selected = X_full_scaled[:, feature_indices]
    X_test_selected = X_test_scaled[:, feature_indices]

    print(f"Training on full data with {len(best_result['features'])} features...")
    production_model = SVMOneVsAll(**SVM_PARAMS)
    production_model.fit(X_full_selected, y_train_encoded)
    print("✓ Training complete")

    # Generate predictions
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

    # With thresholds (RECOMMENDED)
    make_predictions(
        production_model,
        X_test_selected,
        test_ids,
        int_to_label,
        "output/submission_svm_only_feature_selected.csv",
        best_thresholds,
    )

    # Without thresholds (for comparison)
    make_predictions(
        production_model,
        X_test_selected,
        test_ids,
        int_to_label,
        "output/submission_svm_only_feature_selected_baseline.csv",
        None,
    )

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save configuration
    config = {
        "svm_hyperparameters": SVM_PARAMS,
        "feature_selection": {
            "best_strategy": best_result["strategy"],
            "n_features_selected": best_result["n_features"],
            "total_features": len(feature_names),
            "cv_mean_f1": float(best_result["mean_f1"]),
            "cv_std_f1": float(best_result["std_f1"]),
        },
        "thresholds": {
            int_to_label[i]: float(best_thresholds[i]) for i in range(len(int_to_label))
        },
        "threshold_f1": float(threshold_f1),
        "selected_features": best_result["features"],
    }

    with open("output/svm_only_feature_selection_results.json", "w") as f:
        json.dump(config, f, indent=2)
    print("✓ Results saved to output/svm_only_feature_selection_results.json")

    # Save selected features list
    with open("output/svm_only_selected_features.txt", "w") as f:
        f.write("\n".join(best_result["features"]))
    print("✓ Selected features saved to output/svm_only_selected_features.txt")

    # Save model
    with open("output/svm_only_feature_selected_model.pkl", "wb") as f:
        pickle.dump(production_model, f)
    print("✓ Model saved to output/svm_only_feature_selected_model.pkl")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"✓ Tested {len(all_results)} feature selection strategies")
    print(f"✓ Best strategy: {best_result['strategy']}")
    print(
        f"✓ Selected {best_result['n_features']} features (from {len(feature_names)} total)"
    )
    print(
        f"✓ Reduction: {100 * (1 - best_result['n_features'] / len(feature_names)):.1f}%"
    )
    print(f"✓ CV F1: {best_result['mean_f1']:.4f} ± {best_result['std_f1']:.4f}")
    print(f"✓ Threshold-optimized F1: {threshold_f1:.4f}")

    # Compare to baseline (all features)
    baseline_result = [r for r in all_results if r["strategy"] == "all_features"][0]
    improvement = (best_result["mean_f1"] - baseline_result["mean_f1"]) * 100
    print(f"✓ Improvement over all features: {improvement:+.2f}%")

    print(f"\n✓ Generated submissions:")
    print(f"    1. output/submission_svm_only_feature_selected.csv (RECOMMENDED)")
    print(f"    2. output/submission_svm_only_feature_selected_baseline.csv")
    print(f"\n🎯 Try submission_svm_only_feature_selected.csv on Kaggle!")
    print("   Feature selection + thresholds should boost performance!")
    print("=" * 80)


if __name__ == "__main__":
    main()
