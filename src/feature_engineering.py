import pandas as pd
import numpy as np

# Load the big historical dataset
df = pd.read_csv("data/raw/matches_all_leagues.csv")
print(f"Total matches loaded: {len(df):,}")

# Keep only the columns we need
df = df[["Date", "HomeTeam", "AwayTeam", 
         "FTHG", "FTAG", "FTR", 
         "HTHG", "HTAG", "HTR",
         "HS", "AS", "HST", "AST",
         "season", "league"]].copy()

# Rename to readable names
df.columns = ["date", "home_team", "away_team",
              "home_goals", "away_goals", "result",
              "home_goals_ht", "away_goals_ht", "result_ht",
              "home_shots", "away_shots", 
              "home_shots_target", "away_shots_target",
              "season", "league"]

# Drop rows with missing values
df = df.dropna(subset=["home_goals", "away_goals", "result"])

# Convert result to numbers
# H = Home Win, D = Draw, A = Away Win
result_map = {"H": 1, "D": 0, "A": -1}
df["result"] = df["result"].map(result_map)

# Sort by date
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date").reset_index(drop=True)

# Rolling form - last 5 games average goals
def rolling_form(df, team_col, goals_col, window=5):
    return df.groupby(team_col)[goals_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

print("Calculating rolling form...")
df["home_form"]          = rolling_form(df, "home_team", "home_goals")
df["away_form"]          = rolling_form(df, "away_team", "away_goals")
df["home_conceded_form"] = rolling_form(df, "home_team", "away_goals")
df["away_conceded_form"] = rolling_form(df, "away_team", "home_goals")
df["home_shots_form"]    = rolling_form(df, "home_team", "home_shots")
df["away_shots_form"]    = rolling_form(df, "away_team", "away_shots")

# Save processed data
df.to_csv("data/processed/matches_clean.csv", index=False)
print(f"\n✅ Processed {len(df):,} matches saved!")
print(f"Leagues: {df['league'].unique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nSample:")
print(df[["date", "home_team", "away_team", 
          "home_goals", "away_goals", "result",
          "home_form", "away_form"]].head())