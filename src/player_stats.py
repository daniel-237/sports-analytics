import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("APIFOOTBALL_KEY")

headers = {
    "x-apisports-key": API_KEY,
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

BASE_URL = "https://v3.football.api-sports.io"

# Get Premier League top scorers 2023
print("Fetching Premier League player stats...")

response = requests.get(
    f"{BASE_URL}/players/topscorers",
    headers=headers,
    params={"league": "39", "season": "2023"}
)

data = response.json()
players = []

for item in data["response"]:
    player = item["player"]
    stats = item["statistics"][0]
    players.append({
        "name":             player["name"],
        "age":              player["age"],
        "nationality":      player["nationality"],
        "team":             stats["team"]["name"],
        "goals":            stats["goals"]["total"],
        "assists":          stats["goals"]["assists"],
        "appearances":      stats["games"]["appearences"],
        "minutes":          stats["games"]["minutes"],
        "yellow_cards":     stats["cards"]["yellow"],
        "red_cards":        stats["cards"]["red"],
        "shots_on_target":  stats["shots"]["on"],
        "rating":           stats["games"]["rating"]
    })

df = pd.DataFrame(players)
print(df[["name", "team", "goals", 
          "assists", "yellow_cards"]].to_string())

df.to_csv("data/processed/player_stats.csv", index=False)
print(f"\n✅ {len(df)} players saved to player_stats.csv!")