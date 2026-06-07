"""
Advanced Feature Engineering for Football Match Prediction.
Adds Elo ratings, head to head records, form streaks,
points form, goal difference trends, and attack/defence strength.
"""
import pandas as pd
import numpy as np

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/raw/matches_all_leagues.csv")
print(f"Total matches loaded: {len(df):,}")

# ── KEEP NEEDED COLUMNS ───────────────────────────────────────────────────────
df = df[["Date", "HomeTeam", "AwayTeam",
         "FTHG", "FTAG", "FTR",
         "HTHG", "HTAG",
         "HS", "AS", "HST", "AST",
         "season", "league"]].copy()

df.columns = ["date", "home_team", "away_team",
              "home_goals", "away_goals", "result",
              "home_goals_ht", "away_goals_ht",
              "home_shots", "away_shots",
              "home_shots_target", "away_shots_target",
              "season", "league"]

# Drop rows with missing scores
df = df.dropna(subset=["home_goals", "away_goals", "result"])

# Map result to numbers
result_map = {"H": 1, "D": 0, "A": -1}
df["result"] = df["result"].map(result_map)

# Sort by date
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date").reset_index(drop=True)

print(f"Matches after cleaning: {len(df):,}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ── HELPER: ROLLING AVERAGE ───────────────────────────────────────────────────
def rolling_avg(df, team_col, stat_col, window=5):
    return df.groupby(team_col)[stat_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# ── BASIC FORM FEATURES ───────────────────────────────────────────────────────
print("\nCalculating basic form features...")

df["home_goals_form"]    = rolling_avg(df, "home_team", "home_goals")
df["away_goals_form"]    = rolling_avg(df, "away_team", "away_goals")
df["home_conceded_form"] = rolling_avg(df, "home_team", "away_goals")
df["away_conceded_form"] = rolling_avg(df, "away_team", "home_goals")
df["home_shots_form"]    = rolling_avg(df, "home_team", "home_shots")
df["away_shots_form"]    = rolling_avg(df, "away_team", "away_shots")
df["home_sot_form"]      = rolling_avg(df, "home_team", "home_shots_target")
df["away_sot_form"]      = rolling_avg(df, "away_team", "away_shots_target")

# ── GOAL DIFFERENCE FORM ─────────────────────────────────────────────────────
df["home_gd"] = df["home_goals"] - df["away_goals"]
df["away_gd"] = df["away_goals"] - df["home_goals"]
df["home_gd_form"] = rolling_avg(df, "home_team", "home_gd")
df["away_gd_form"] = rolling_avg(df, "away_team", "away_gd")

# ── POINTS FORM (last 5 and last 10) ─────────────────────────────────────────
df["home_points"] = df["result"].map({1: 3, 0: 1, -1: 0})
df["away_points"] = df["result"].map({1: 0, 0: 1, -1: 3})

df["home_points_form5"]  = rolling_avg(df, "home_team", "home_points", window=5)
df["away_points_form5"]  = rolling_avg(df, "away_team", "away_points", window=5)
df["home_points_form10"] = rolling_avg(df, "home_team", "home_points", window=10)
df["away_points_form10"] = rolling_avg(df, "away_team", "away_points", window=10)

# ── WIN RATE (last 10) ────────────────────────────────────────────────────────
df["home_win"] = (df["result"] == 1).astype(int)
df["away_win"] = (df["result"] == -1).astype(int)
df["home_win_rate"] = rolling_avg(df, "home_team", "home_win", window=10)
df["away_win_rate"] = rolling_avg(df, "away_team", "away_win", window=10)

# ── CLEAN SHEET RATE ──────────────────────────────────────────────────────────
df["home_clean_sheet"] = (df["away_goals"] == 0).astype(int)
df["away_clean_sheet"] = (df["home_goals"] == 0).astype(int)
df["home_cs_rate"] = rolling_avg(df, "home_team", "home_clean_sheet", window=10)
df["away_cs_rate"] = rolling_avg(df, "away_team", "away_clean_sheet", window=10)

# ── FAILED TO SCORE RATE ──────────────────────────────────────────────────────
df["home_failed_score"] = (df["home_goals"] == 0).astype(int)
df["away_failed_score"] = (df["away_goals"] == 0).astype(int)
df["home_fts_rate"] = rolling_avg(df, "home_team", "home_failed_score", window=10)
df["away_fts_rate"] = rolling_avg(df, "away_team", "away_failed_score", window=10)

# ── ELO RATINGS ───────────────────────────────────────────────────────────────
print("Calculating Elo ratings...")

K      = 32    # K-factor: how much each result changes Elo
BASE   = 1500  # Starting Elo for all teams
elo_dict = {}  # team -> current Elo

home_elo_before = []
away_elo_before = []

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

for _, row in df.iterrows():
    ht = row["home_team"]
    at = row["away_team"]

    # Get current Elo (default to base if new team)
    elo_h = elo_dict.get(ht, BASE)
    elo_a = elo_dict.get(at, BASE)

    # Store pre-match Elo
    home_elo_before.append(elo_h)
    away_elo_before.append(elo_a)

    # Actual scores for Elo update
    if row["result"] == 1:    s_h, s_a = 1.0, 0.0
    elif row["result"] == 0:  s_h, s_a = 0.5, 0.5
    else:                     s_h, s_a = 0.0, 1.0

    # Expected scores
    e_h = expected_score(elo_h, elo_a)
    e_a = expected_score(elo_a, elo_h)

    # Update Elo
    elo_dict[ht] = elo_h + K * (s_h - e_h)
    elo_dict[at] = elo_a + K * (s_a - e_a)

df["home_elo"] = home_elo_before
df["away_elo"] = away_elo_before
df["elo_diff"] = df["home_elo"] - df["away_elo"]

print(f"  Elo ratings calculated for {len(elo_dict)} teams")

# ── FORM STREAKS ──────────────────────────────────────────────────────────────
print("Calculating form streaks...")

home_streaks = []
away_streaks = []
streak_dict  = {}  # team -> current streak (+ve = wins, -ve = losses)

for _, row in df.iterrows():
    ht = row["home_team"]
    at = row["away_team"]

    # Record pre-match streaks
    home_streaks.append(streak_dict.get(ht, 0))
    away_streaks.append(streak_dict.get(at, 0))

    # Update streaks after match
    if row["result"] == 1:
        streak_dict[ht] = max(streak_dict.get(ht, 0), 0) + 1
        streak_dict[at] = min(streak_dict.get(at, 0), 0) - 1
    elif row["result"] == -1:
        streak_dict[ht] = min(streak_dict.get(ht, 0), 0) - 1
        streak_dict[at] = max(streak_dict.get(at, 0), 0) + 1
    else:
        streak_dict[ht] = 0
        streak_dict[at] = 0

df["home_streak"] = home_streaks
df["away_streak"] = away_streaks

# ── ATTACK & DEFENCE STRENGTH ─────────────────────────────────────────────────
print("Calculating attack and defence strength...")

# League average goals per game (rolling, to avoid data leakage)
df["total_goals"] = df["home_goals"] + df["away_goals"]
league_avg = df.groupby("league")["total_goals"].transform(
    lambda x: x.shift(1).rolling(50, min_periods=10).mean()
).fillna(2.5)

df["home_attack_strength"]  = df["home_goals_form"]    / (league_avg / 2).clip(lower=0.1)
df["away_attack_strength"]  = df["away_goals_form"]    / (league_avg / 2).clip(lower=0.1)
df["home_defence_strength"] = df["home_conceded_form"] / (league_avg / 2).clip(lower=0.1)
df["away_defence_strength"] = df["away_conceded_form"] / (league_avg / 2).clip(lower=0.1)

# ── SEASON STAGE ──────────────────────────────────────────────────────────────
df["matchday"]     = df.groupby(["season", "league", "home_team"]).cumcount() + 1
df["season_stage"] = pd.cut(df["matchday"], bins=[0, 12, 25, 50],
                             labels=[0, 1, 2]).astype(float)

# ── HEAD TO HEAD ──────────────────────────────────────────────────────────────
print("Calculating head to head records (this takes a few minutes)...")

h2h_home_wins = []
h2h_away_wins = []
h2h_draws     = []
h2h_total     = []

for idx, row in df.iterrows():
    past = df[
        (df.index < idx) &
        (
            ((df["home_team"] == row["home_team"]) &
             (df["away_team"] == row["away_team"])) |
            ((df["home_team"] == row["away_team"]) &
             (df["away_team"] == row["home_team"]))
        )
    ].tail(10)  # Last 10 h2h meetings

    total = len(past)
    if total > 0:
        hw = len(past[past["result"] == 1])
        aw = len(past[past["result"] == -1])
        dr = len(past[past["result"] == 0])
    else:
        hw, aw, dr = 0, 0, 0

    h2h_home_wins.append(hw)
    h2h_away_wins.append(aw)
    h2h_draws.append(dr)
    h2h_total.append(total)

    if idx % 5000 == 0:
        print(f"  Processing match {idx:,}/{len(df):,}...")

df["h2h_home_wins"] = h2h_home_wins
df["h2h_away_wins"] = h2h_away_wins
df["h2h_draws"]     = h2h_draws
df["h2h_total"]     = h2h_total
df["h2h_home_rate"] = (df["h2h_home_wins"] / df["h2h_total"].clip(lower=1)).round(3)

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("\nSaving processed data...")
df.to_csv("data/processed/matches_clean.csv", index=False)

print(f"\n✅ {len(df):,} matches saved with advanced features!")
print(f"\nFeature list:")
feature_cols = [c for c in df.columns if c not in
                ["date","home_team","away_team","home_goals","away_goals",
                 "result","home_goals_ht","away_goals_ht","home_shots",
                 "away_shots","home_shots_target","away_shots_target",
                 "season","league","home_gd","away_gd","home_points",
                 "away_points","home_win","away_win","home_clean_sheet",
                 "away_clean_sheet","home_failed_score","away_failed_score",
                 "total_goals","matchday"]]
for c in feature_cols:
    print(f"  {c}")
print(f"\nTotal: {len(feature_cols)} features")