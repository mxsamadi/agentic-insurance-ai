
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report
)

# Project paths
ROOT = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(ROOT / "data/insurance_claims.csv")

# Feature engineering
df["claim_to_repair_ratio"] = (
    df["claim_amount"] / df["repair_estimate"]
)

# Separate features and target
X = df.drop(columns=["claim_id", "policy_id", "is_fraud"])
y = df["is_fraud"]

# Split by policy to avoid overlapping policies
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=df["policy_id"])
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Handle class imbalance
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(f"Training claims: {len(X_train):,}")
print(f"Testing claims: {len(X_test):,}")
print(f"Training fraud rate: {y_train.mean():.2%}")
print(f"Testing fraud rate: {y_test.mean():.2%}")
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["claim_type"]
        )
    ],
    remainder="passthrough"
)

# Define XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)

# Build modeling pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train model
pipeline.fit(X_train, y_train)

# Generate predictions
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

# Evaluate model
print("\n--- MODEL PERFORMANCE ---")

print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

print(
    f"Average Precision: "
    f"{average_precision_score(y_test, y_prob):.4f}"
)

print(f"Random ranking baseline: {y_test.mean():.4f}")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, zero_division=0))

# Evaluate top 500 suspicious claims
k = min(500, len(y_test))

top_k_indices = np.argsort(y_prob)[::-1][:k]

fraud_in_top_k = y_test.iloc[top_k_indices].sum()

precision_at_k = fraud_in_top_k / k
recall_at_k = fraud_in_top_k / y_test.sum()

print("\n--- INVESTIGATION PERFORMANCE ---")
print(f"Precision@{k}: {precision_at_k:.2%}")
print(f"Recall@{k}: {recall_at_k:.2%}")
print(f"Fraud cases in top {k}: {fraud_in_top_k}")

# Save trained pipeline
model_path = ROOT / "models/risk_model.joblib"

joblib.dump(pipeline, model_path)

print(f"\nModel saved to: {model_path}")
