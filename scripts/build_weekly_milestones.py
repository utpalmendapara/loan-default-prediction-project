"""
build_weekly_milestones.py
==========================
Generates all artifacts, models, visualizations, and Jupyter notebooks
required for Weeks 1 to 6 and Week 10 of the Computer Engineering Department
ML Project SOP.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

# Import custom scratch decision tree
import sys
sys.path.insert(0, os.path.abspath("."))
from backend.models.scratch_decision_tree import ScratchDecisionTreeClassifier

print("[STEP 1] Setting up paths and environment...")
os.makedirs("reports/visualizations", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# Set publication style for matplotlib
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

print("[STEP 2] Loading and sampling dataset...")
df_raw = pd.read_csv("data/Loan_default.csv")
print(f"Loaded raw dataset shape: {df_raw.shape}")

# Stratified sample of 40,000 for fast, high-quality notebook demonstration
df_sample, _ = train_test_split(df_raw, train_size=40000, random_state=42, stratify=df_raw['Default'])
df_sample = df_sample.copy()
df_sample = df_sample.drop(columns=['LoanID'])

# Preprocess categorical features identically to project standard
binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
for col in binary_cols:
    df_sample[col] = df_sample[col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)

nominal_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
df_encoded = pd.get_dummies(df_sample, columns=nominal_cols, drop_first=False)

# Features and target
X = df_encoded.drop(columns=['Default'])
y = df_encoded['Default'].values
feature_names = list(X.columns)

print(f"Encoded feature shape: {X.shape}, Target distribution: {np.bincount(y)}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.20, random_state=42, stratify=y
)

# Standard Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[STEP 3] Training library models and scratch model...")
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=7, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gaussian Naive Bayes": GaussianNB()
}

results = {}
for name, clf in models.items():
    print(f"  Training {name}...")
    if name in ["Logistic Regression", "Gaussian Naive Bayes"]:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        "model": clf,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    print(f"    -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# Train Scratch Decision Tree on subset for comparison
print("  Training Scratch Decision Tree (No Library)...")
scratch_tree = ScratchDecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10)
# Subsample for fast tree construction
scratch_X_tr, _, scratch_y_tr, _ = train_test_split(X_train, y_train, train_size=5000, random_state=42, stratify=y_train)
scratch_tree.fit(scratch_X_tr, scratch_y_tr)
scratch_preds = scratch_tree.predict(X_test[:2000])
scratch_acc = accuracy_score(y_test[:2000], scratch_preds)
print(f"    -> Scratch Decision Tree Test Accuracy: {scratch_acc:.4f}")

print("[STEP 4] Generating Week 6 Visualizations in reports/visualizations/...")

# 1. ROC Curves
plt.figure(figsize=(9, 7))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {res['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.title("ROC Curves - Model Comparison (SOP Week 6)")
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("reports/visualizations/roc_curve_comparison.png", dpi=300)
plt.close()

# 2. Precision-Recall Curves
plt.figure(figsize=(9, 7))
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
    ap = average_precision_score(y_test, res["y_prob"])
    plt.plot(rec, prec, lw=2, label=f"{name} (AP = {ap:.3f})")
plt.axhline(y=np.mean(y_test), color='k', linestyle='--', label=f'Baseline Proportion ({np.mean(y_test):.3f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for Default Class (SOP Week 6)")
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig("reports/visualizations/precision_recall_curves.png", dpi=300)
plt.close()

# 3. Confusion Matrix Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm_dt = confusion_matrix(y_test, results["Decision Tree"]["y_pred"])
cm_rf = confusion_matrix(y_test, results["Random Forest"]["y_pred"])

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
axes[0].set_title(f"Decision Tree Confusion Matrix\nAccuracy: {results['Decision Tree']['accuracy']*100:.1f}%")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
axes[1].set_title(f"Random Forest Confusion Matrix\nAccuracy: {results['Random Forest']['accuracy']*100:.1f}%")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("reports/visualizations/confusion_matrix_heatmaps.png", dpi=300)
plt.close()

# 4. Feature Importance Ranking
rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[top_idx], y=[feature_names[i] for i in top_idx], palette="viridis")
plt.title("Top 15 Most Predictive Features for Loan Default (SOP Week 6)")
plt.xlabel("Gini Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("reports/visualizations/feature_importance_ranking.png", dpi=300)
plt.close()

# 5. Overfitting vs Underfitting Learning Curve (Depth sweep)
depths = list(range(1, 17))
train_accs = []
test_accs = []
for d in depths:
    t_clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    t_clf.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, t_clf.predict(X_train)))
    test_accs.append(accuracy_score(y_test, t_clf.predict(X_test)))

plt.figure(figsize=(9, 6))
plt.plot(depths, train_accs, marker='o', lw=2, label="Training Accuracy", color="#2b5c8f")
plt.plot(depths, test_accs, marker='s', lw=2, label="Validation/Test Accuracy", color="#d95f02")
plt.axvline(x=7, color='grey', linestyle=':', label="Optimal Depth ~ 7 (Sweet Spot)")
plt.annotate('High Bias\n(Underfitting)', xy=(2, 0.88), xytext=(2, 0.93),
             arrowprops=dict(arrowstyle="->", color="red"), color="red", fontweight="bold")
plt.annotate('High Variance\n(Overfitting)', xy=(14, 0.94), xytext=(12, 0.97),
             arrowprops=dict(arrowstyle="->", color="red"), color="red", fontweight="bold")
plt.title("Diagnosis of Overfitting vs Underfitting across Tree Depth (SOP Week 4 & 6)")
plt.xlabel("Max Tree Depth")
plt.ylabel("Accuracy Score")
plt.legend(loc="center right")
plt.tight_layout()
plt.savefig("reports/visualizations/overfitting_learning_curves.png", dpi=300)
plt.close()

# 6. Model Calibration Curves
plt.figure(figsize=(9, 6))
for name in ["Random Forest", "Gradient Boosting", "Logistic Regression"]:
    prob_true, prob_pred = calibration_curve(y_test, results[name]["y_prob"], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', lw=2, label=name)
plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")
plt.xlabel("Mean Predicted Default Probability")
plt.ylabel("Fraction of Positives (Actual Default Rate)")
plt.title("Reliability / Calibration Curves (SOP Week 6)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("reports/visualizations/model_calibration_curves.png", dpi=300)
plt.close()

# 7. 5-Fold Stratified Cross-Validation Boxplot
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
print("  Running 5-fold Stratified Cross-Validation...")
# Subsample for CV speed
X_cv, y_cv = X_train[:8000], y_train[:8000]
for name, clf in [("Decision Tree", DecisionTreeClassifier(max_depth=7, random_state=42)),
                  ("Random Forest", RandomForestClassifier(n_estimators=50, max_depth=7, random_state=42, n_jobs=-1)),
                  ("Gradient Boosting", GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42))]:
    scores = cross_val_score(clf, X_cv, y_cv, cv=cv, scoring='roc_auc')
    cv_scores[name] = scores
    print(f"    {name} 5-Fold ROC-AUC: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

plt.figure(figsize=(8, 5))
sns.boxplot(data=pd.DataFrame(cv_scores), palette="Set2")
plt.title("5-Fold Stratified Cross-Validation ROC-AUC Distribution (SOP Week 5 & 6)")
plt.ylabel("ROC-AUC Score")
plt.xlabel("Model Architecture")
plt.tight_layout()
plt.savefig("reports/visualizations/cross_validation_boxplots.png", dpi=300)
plt.close()

print("[STEP 5] Saving benchmark metrics JSON...")
summary_metrics = {
    model_name: {
        "accuracy": float(res["accuracy"]),
        "precision": float(res["precision"]),
        "recall": float(res["recall"]),
        "f1": float(res["f1"]),
        "roc_auc": float(res["roc_auc"])
    }
    for model_name, res in results.items()
}
summary_metrics["Scratch Decision Tree"] = {
    "accuracy": float(scratch_acc),
    "notes": "Built from scratch using pure Python + NumPy with Gini Impurity calculation"
}

with open("reports/summary_metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary_metrics, f, indent=2)

print("Visualizations and summary metrics successfully generated!")
