
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

# Project paths
ROOT = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(ROOT / "data/insurance_claims.csv")

# Feature engineering
df["claim_to_repair_ratio"] = (
    df["claim_amount"] / df["repair_estimate"]
)

# Separate features and labels
X = df.drop(columns=["claim_id", "policy_id", "is_fraud"])
y = df["is_fraud"]

# Use the same split as XGBoost
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
y_test = y.iloc[test_idx]

# Preprocessing
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

# Define Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    max_samples=256,
    random_state=42,
    n_jobs=-1
)

# Build pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train without fraud labels
pipeline.fit(X_train)

# Higher scores indicate greater anomaly
anomaly_scores = -pipeline.score_samples(X_test)

# Identify anomalies
predictions = pipeline.predict(X_test)
anomalies = predictions == -1

# Evaluate against simulated fraud labels
print("\n--- ANOMALY MODEL PERFORMANCE ---")

print(f"Testing claims: {len(X_test):,}")
print(f"Detected anomalies: {anomalies.sum():,}")
print(f"Anomaly rate: {anomalies.mean():.2%}")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, anomaly_scores):.4f}"
)

print(
    f"Average Precision: "
    f"{average_precision_score(y_test, anomaly_scores):.4f}"
)

print(f"Random ranking baseline: {y_test.mean():.4f}")

# Evaluate top 500 anomalous claims
k = min(500, len(X_test))

top_k = np.argsort(anomaly_scores)[::-1][:k]

fraud_in_top_k = y_test.iloc[top_k].sum()

print("\n--- INVESTIGATION PERFORMANCE ---")
print(f"Precision@{k}: {fraud_in_top_k / k:.2%}")
print(f"Recall@{k}: {fraud_in_top_k / y_test.sum():.2%}")
print(f"Fraud cases in top {k}: {fraud_in_top_k}")

# Save trained pipeline
model_path = ROOT / "models/anomaly_model.joblib"

joblib.dump(pipeline, model_path)

print(f"\nModel saved to: {model_path}")
