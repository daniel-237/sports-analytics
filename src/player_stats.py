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

def get_players(league_id, season, page=1):
    response = requests.get(
        f"{BASE_URL}/players",
        headers=headers,
        params={"league": league_id, "season": season, "page": page}
    )
    return response.json()

print("Fetching Premier League players 2023...")
all_players = []

# Fetch first 3 pages (free tier allows 100 requests/day)
for page in range(1, 4):
    print(f"  Page {page}...")
    data = get_players("39", "2023", page)
    
    if "response" not in data:
        print(f"  Error: {data}")
        break
        
    for item in data["response"]:
        player = item["player"]
        stats  = item["statistics"][0]
        all_players.append({
            "id":               player["id"],
            "name":             player["name"],
            "age":              player["age"],
            "nationality":      player["nationality"],
            "position":         stats["games"]["position"],
            "team":             stats["team"]["name"],
            "appearances":      stats["games"]["appearences"] or 0,
            "minutes":          stats["games"]["minutes"] or 0,
            "goals":            stats["goals"]["total"] or 0,
            "assists":          stats["goals"]["assists"] or 0,
            "shots_total":      stats["shots"]["total"] or 0,
            "shots_on_target":  stats["shots"]["on"] or 0,
            "passes_total":     stats["passes"]["total"] or 0,
            "pass_accuracy":    stats["passes"]["accuracy"] or 0,
            "dribbles":         stats["dribbles"]["success"] or 0,
            "tackles":          stats["tackles"]["total"] or 0,
            "interceptions":    stats["tackles"]["interceptions"] or 0,
            "yellow_cards":     stats["cards"]["yellow"] or 0,
            "red_cards":        stats["cards"]["red"] or 0,
            "rating":           stats["games"]["rating"] or 0,
            "duels_won":        stats["duels"]["won"] or 0,
        })
    
    print(f"  Got {len(data['response'])} players")
    
    # Check if more pages exist
    total_pages = data["paging"]["total"]
    if page >= total_pages:
        break

df = pd.DataFrame(all_players)
df.to_csv("data/processed/player_stats.csv", index=False)
print(f"\n✅ {len(df)} players saved!")
print(df[["name", "team", "goals", "assists", "rating"]].head(10))