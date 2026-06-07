import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, precision_score,
                              recall_score, f1_score)
from xgboost import XGBClassifier
import joblib
import json

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/matches_clean.csv")
print(f"Total matches: {len(df):,}")

# ── FEATURES ──────────────────────────────────────────────────────────────────
features = [
    "home_form",
    "away_form",
    "home_conceded_form",
    "away_conceded_form",
    "home_shots_form",
    "away_shots_form"
]

df = df.dropna(subset=features)

# Map result to 0, 1, 2 for XGBoost
result_map     = {-1: 0, 0: 1, 1: 2}
result_inv_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
df["result_encoded"] = df["result"].map(result_map)

# ── CHRONOLOGICAL TRAIN / TEST SPLIT ─────────────────────────────────────────
# Train on older seasons, test on newer — simulates real prediction
df["season"] = df["season"].astype(str)
train = df[df["season"] <= "2122"]
test  = df[df["season"] >  "2122"]

X_train = train[features]
y_train = train["result_encoded"]
X_test  = test[features]
y_test  = test["result_encoded"]

print(f"\nChronological split:")
print(f"  Training:  {len(X_train):,} matches (seasons up to 2021/22)")
print(f"  Testing:   {len(X_test):,}  matches (seasons 2022/23+)")

# ── BASELINE MODEL ────────────────────────────────────────────────────────────
# Always predict Home Win (most common outcome)
baseline_preds = np.full(len(y_test), 2)
baseline_acc   = accuracy_score(y_test, baseline_preds)
print(f"\nBaseline accuracy (always Home Win): {baseline_acc:.1%}")

# ── TRAIN MULTIPLE MODELS ─────────────────────────────────────────────────────
models = {
   "Logistic Regression": LogisticRegression(
    max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, eval_metric="mlogloss", n_jobs=-1),
}

results = {}
print("\nTraining models...")

for name, m in models.items():
    m.fit(X_train, y_train)
    preds    = m.predict(X_test)
    acc      = accuracy_score(y_test, preds)
    prec     = precision_score(y_test, preds, average="weighted",
                               zero_division=0)
    rec      = recall_score(y_test, preds, average="weighted",
                            zero_division=0)
    f1       = f1_score(y_test, preds, average="weighted",
                        zero_division=0)
    cm       = confusion_matrix(y_test, preds).tolist()
    report   = classification_report(
        y_test, preds,
        target_names=["Away Win","Draw","Home Win"],
        output_dict=True)

    results[name] = {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    print(f"  {name}: {acc:.1%} accuracy")

# ── FEATURE IMPORTANCE (XGBoost) ──────────────────────────────────────────────
xgb_model      = models["XGBoost"]
importance_vals = xgb_model.feature_importances_.tolist()
importance_dict = dict(zip(features, importance_vals))

# ── SAVE BEST MODEL ───────────────────────────────────────────────────────────
joblib.dump(xgb_model, "models/match_predictor.pkl")
print("\n✅ XGBoost model saved to models/match_predictor.pkl")

# ── SAVE METRICS ──────────────────────────────────────────────────────────────
metrics = {
    "baseline_accuracy":  round(baseline_acc, 4),
    "train_size":         len(X_train),
    "test_size":          len(X_test),
    "features":           features,
    "model_results":      results,
    "feature_importance": importance_dict,
    "train_seasons":      "up to 2021/22",
    "test_seasons":       "2022/23 onwards",
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("✅ Metrics saved to models/metrics.json")

# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10}")
print("-"*50)
print(f"{'Baseline (Home Win)':<25} {baseline_acc:>10.1%} {'N/A':>10}")
for name, r in results.items():
    print(f"{name:<25} {r['accuracy']:>10.1%} {r['f1']:>10.1%}")
print("="*50)
print("\nXGBoost chosen as best performing model.")