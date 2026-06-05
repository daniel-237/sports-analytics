import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# Load the big cleaned dataset
df = pd.read_csv("data/processed/matches_clean.csv")
print(f"Training on {len(df):,} matches!")

# Features the model learns from
features = [
    "home_form",
    "away_form",
    "home_conceded_form",
    "away_conceded_form",
    "home_shots_form",
    "away_shots_form"
]

# Drop rows with missing values
df = df.dropna(subset=features)

# Map result to 0, 1, 2 (XGBoost needs 0-indexed labels)
result_map = {-1: 0, 0: 1, 1: 2}
df["result_encoded"] = df["result"].map(result_map)

X = df[features]
y = df["result_encoded"]

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train):,} matches...")
print(f"Testing on {len(X_test):,} matches...")

# Train XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n✅ Model Accuracy: {accuracy:.1%}")
print("\nDetailed Results:")
print(classification_report(y_test, predictions,
      target_names=["Away Win", "Draw", "Home Win"]))

# Save the model
joblib.dump(model, "models/match_predictor.pkl")
print("Model saved ✓")