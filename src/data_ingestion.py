import requests
import pandas as pd
from dotenv import load_dotenv
import os
import time

# Load API key
load_dotenv()
API_KEY = os.getenv("FOOTBALL_API_KEY")
headers = {"X-Auth-Token": API_KEY}

# Fetch all available seasons
seasons = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
all_matches = []

for season in seasons:
    print(f"Fetching season {season}...")
    url = f"https://api.football-data.org/v4/competitions/PL/matches?season={season}"
    response = requests.get(url, headers=headers)
    data = response.json()
    
    if "matches" in data:
        matches = pd.json_normalize(data["matches"])
        matches["season"] = season
        all_matches.append(matches)
        print(f"✅ Season {season}: {len(matches)} matches fetched")
    else:
        print(f"⚠️ Season {season} not available: {data.get('message', 'skipping...')}")
    
    # Pause between requests - free tier limit
    time.sleep(7)

# Combine all seasons that worked
all_df = pd.concat(all_matches, ignore_index=True)
all_df.to_csv("data/raw/matches_historical.csv", index=False)
print(f"\n🎉 Total matches saved: {len(all_df)}")
print(f"Seasons collected: {all_df['season'].unique()}")