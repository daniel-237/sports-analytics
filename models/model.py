"""
Advanced Match Prediction Model
Trained on 35 engineered features including Elo ratings,
head to head records, form streaks, and attack/defence strength.
"""
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
print("Loading data...")
df = pd.read_csv("data/processed/matches_clean.csv")
print(f"Total matches: {len(df):,}")

# ── FEATURES ──────────────────────────────────────────────────────────────────
ALL_FEATURES = [
    "home_goals_form", "away_goals_form",
    "home_conceded_form", "away_conceded_form",
    "home_shots_form", "away_shots_form",
    "home_sot_form", "away_sot_form",
    "home_gd_form", "away_gd_form",
    "home_points_form5", "away_points_form5",
    "home_points_form10", "away_points_form10",
    "home_win_rate", "away_win_rate",
    "home_cs_rate", "away_cs_rate",
    "home_fts_rate", "away_fts_rate",
    "home_elo", "away_elo", "elo_diff",
    "home_streak", "away_streak",
    "home_attack_strength", "away_attack_strength",
    "home_defence_strength", "away_defence_strength",
    "season_stage",
    "h2h_home_wins", "h2h_away_wins",
    "h2h_draws", "h2h_total", "h2h_home_rate",
]

# Keep only features that exist
FEATURES = [f for f in ALL_FEATURES if f in df.columns]
print(f"\nUsing {len(FEATURES)} features")

# Drop rows with missing values
df = df.dropna(subset=FEATURES)
print(f"Matches after dropping nulls: {len(df):,}")

# ── ENCODE TARGET ─────────────────────────────────────────────────────────────
result_map = {-1: 0, 0: 1, 1: 2}
df["result_encoded"] = df["result"].map(result_map)
df = df.dropna(subset=["result_encoded"])
df["result_encoded"] = df["result_encoded"].astype(int)

# ── CHRONOLOGICAL SPLIT ───────────────────────────────────────────────────────
df["season"] = df["season"].astype(str)
train = df[df["season"] <= "2122"]
test  = df[df["season"] >  "2122"]

X_train = train[FEATURES]
y_train = train["result_encoded"]
X_test  = test[FEATURES]
y_test  = test["result_encoded"]

print(f"\nTraining:  {len(X_train):,} matches (up to 2021/22)")
print(f"Testing:   {len(X_test):,} matches (2022/23+)")

# ── BASELINE ──────────────────────────────────────────────────────────────────
baseline_preds = np.full(len(y_test), 2)
baseline_acc   = accuracy_score(y_test, baseline_preds)
print(f"\nBaseline (always Home Win): {baseline_acc:.1%}")

# ── TRAIN MODELS ──────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, C=0.1),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3,
        random_state=42, eval_metric="mlogloss",
        n_jobs=-1
    ),
}

results = {}
print("\nTraining models...")

for name, m in models.items():
    print(f"  {name}...")
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1   = f1_score(y_test, preds, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, preds).tolist()
    rep  = classification_report(
        y_test, preds,
        target_names=["Away Win", "Draw", "Home Win"],
        output_dict=True, zero_division=0
    )
    results[name] = {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "confusion_matrix": cm, "classification_report": rep,
    }
    print(f"  {name}: {acc:.1%}")

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
xgb_model       = models["XGBoost"]
importance_dict = dict(zip(FEATURES, xgb_model.feature_importances_.tolist()))

# ── SAVE ──────────────────────────────────────────────────────────────────────
joblib.dump(xgb_model, "models/match_predictor.pkl")
print("\n✅ Model saved!")

metrics_out = {
    "baseline_accuracy":  round(baseline_acc, 4),
    "train_size":         len(X_train),
    "test_size":          len(X_test),
    "features":           FEATURES,
    "model_results":      results,
    "feature_importance": importance_dict,
    "train_seasons":      "up to 2021/22",
    "test_seasons":       "2022/23 onwards",
    "num_features":       len(FEATURES),
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print("✅ Metrics saved!")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("MODEL COMPARISON SUMMARY")
print("="*55)
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10}")
print("-"*55)
print(f"{'Baseline':<25} {baseline_acc:>10.1%} {'N/A':>10}")
for name, r in results.items():
    print(f"{name:<25} {r['accuracy']:>10.1%} {r['f1']:>10.1%}")
print("="*55)
best       = max(results, key=lambda x: results[x]["accuracy"])
improvement = results[best]["accuracy"] - baseline_acc
print(f"\n🏆 Best: {best}")
print(f"📈 Improvement over baseline: +{improvement:.1%}")
print(f"🔧 Features: {len(FEATURES)}")

# Check PL only accuracy
print("\n── Premier League Only ──")
pl_test = test[test["league"] == "Premier League"]
if len(pl_test) > 0:
    pl_preds = xgb_model.predict(pl_test[FEATURES])
    pl_acc   = accuracy_score(pl_test["result_encoded"], pl_preds)
    print(f"PL Accuracy: {pl_acc:.1%} on {len(pl_test):,} matches")