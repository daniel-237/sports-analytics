import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import METRICS_PATH, MODEL_FEATURES, MODEL_PATH, RANDOM_STATE
from src.data_loader import load_match_data
from src.feature_engineering import build_match_features


def prepare_target(frame: pd.DataFrame) -> pd.Series:
    return frame["result"].map({-1: 0, 0: 1, 1: 2}).astype(int)


def get_season_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(frame["season"].dropna().astype(str).unique().tolist())

    if len(seasons) < 4:
        train_frame, test_frame = train_test_split(
            frame,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=prepare_target(frame),
        )
        return train_frame, test_frame

    test_seasons = seasons[-3:]
    train_frame = frame[~frame["season"].astype(str).isin(test_seasons)].copy()
    test_frame = frame[frame["season"].astype(str).isin(test_seasons)].copy()

    if train_frame.empty or test_frame.empty:
        train_frame, test_frame = train_test_split(
            frame,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=prepare_target(frame),
        )

    return train_frame, test_frame


def train_model() -> None:
    matches = load_match_data()
    features = build_match_features(matches)

    features = features.dropna(subset=["result"]).copy()
    features[MODEL_FEATURES] = features[MODEL_FEATURES].fillna(0)

    train_frame, test_frame = get_season_split(features)

    x_train = train_frame[MODEL_FEATURES]
    y_train = prepare_target(train_frame)

    x_test = test_frame[MODEL_FEATURES]
    y_test = prepare_target(test_frame)

    model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Away Win", "Draw", "Home Win"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, predictions).tolist()

    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    metrics = {
        "accuracy": accuracy,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_seasons": sorted(train_frame["season"].astype(str).unique().tolist()),
        "test_seasons": sorted(test_frame["season"].astype(str).unique().tolist()),
        "target_mapping": {
            "0": "Away Win",
            "1": "Draw",
            "2": "Home Win",
        },
        "classification_report": report,
        "confusion_matrix": matrix,
        "feature_importance": feature_importance.to_dict(orient="records"),
        "model_features": MODEL_FEATURES,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Model training complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training rows: {len(train_frame):,}")
    print(f"Testing rows: {len(test_frame):,}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print()
    print("Top 10 features:")
    print(feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    train_model()