import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score
)
import joblib

# Paths
CT_PATH = "data/ct_radiomics_samples/sample_ct_features.npy"
THERMAL_PATH = "data/thermal_features/sample_thermal_features.npy"
LABELS = [1]*5 + [0]*5  # Simulated: 5 RP+, 5 RP–

# Output directories
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load and prepare data
ct_features = np.load(CT_PATH)
thermal_features = np.load(THERMAL_PATH)
X = np.hstack([ct_features, thermal_features])
y = np.array(LABELS)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, digits=3, output_dict=True)
auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)

# Save results
with open("results/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred, digits=3))

with open("results/metrics.json", "w") as f:
    json.dump({"AUC": auc, "Accuracy": acc}, f, indent=2)

joblib.dump(clf, "models/rf_model.joblib")

print(f"[INFO] Training complete. AUC: {auc:.3f} | Accuracy: {acc:.3f}")
