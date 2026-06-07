import requests
import pandas as pd
from dotenv import load_dotenv
import os
import time
import json

load_dotenv()
API_KEY = os.getenv("APIFOOTBALL_KEY")

headers = {
    "x-apisports-key": API_KEY,
    "x-rapidapi-key":  API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

BASE_URL = "https://v3.football.api-sports.io"

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Premier League = 39
# Championship   = 40
# League One     = 41
# League Two     = 42

LEAGUES_TO_FETCH = [
    {"id": "39", "name": "Premier League"},
    {"id": "40", "name": "Championship"},
    {"id": "41", "name": "League One"},
    {"id": "42", "name": "League Two"},
]

SEASONS_TO_FETCH = ["2021", "2022", "2023", "2024", "2025"]

# How many pages per league/season (20 players per page, free tier = 100 req/day)
# 5 seasons x 4 leagues x 5 pages = 100 requests — within free tier
# Cached data is reused on subsequent runs so requests drop to near zero
PAGES_PER_FETCH = 5

# ── CACHE FILE ────────────────────────────────────────────────────────────────
CACHE_FILE = "data/processed/player_stats_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# ── FETCH PLAYERS ─────────────────────────────────────────────────────────────
def fetch_players(league_id, season, page=1):
    response = requests.get(
        f"{BASE_URL}/players",
        headers=headers,
        params={"league": league_id, "season": season, "page": page}
    )
    return response.json()

# ── PARSE PLAYER ──────────────────────────────────────────────────────────────
def parse_player(item, season, league_name):
    player = item["player"]
    stats  = item["statistics"][0]
    return {
        "id":               player["id"],
        "name":             player["name"],
        "age":              player["age"],
        "nationality":      player["nationality"],
        "position":         stats["games"].get("position") or "Unknown",
        "team":             stats["team"]["name"],
        "league":           league_name,
        "season":           season,
        "appearances":      stats["games"].get("appearences") or 0,
        "minutes":          stats["games"].get("minutes") or 0,
        "goals":            stats["goals"].get("total") or 0,
        "assists":          stats["goals"].get("assists") or 0,
        "shots_total":      stats["shots"].get("total") or 0,
        "shots_on_target":  stats["shots"].get("on") or 0,
        "passes_total":     stats["passes"].get("total") or 0,
        "pass_accuracy":    stats["passes"].get("accuracy") or 0,
        "dribbles":         stats["dribbles"].get("success") or 0,
        "tackles":          stats["tackles"].get("total") or 0,
        "interceptions":    stats["tackles"].get("interceptions") or 0,
        "yellow_cards":     stats["cards"].get("yellow") or 0,
        "red_cards":        stats["cards"].get("red") or 0,
        "rating":           stats["games"].get("rating") or 0,
        "duels_won":        stats["duels"].get("won") or 0,
    }

# ── MAIN FETCH LOOP ───────────────────────────────────────────────────────────
cache      = load_cache()
all_players = []
requests_used = 0

print("=" * 55)
print("Football Player Stats — Multi-Season Fetcher")
print("=" * 55)

for league in LEAGUES_TO_FETCH:
    for season in SEASONS_TO_FETCH:
        cache_key = f"{league['id']}_{season}"

        # Use cached data if available
        if cache_key in cache:
            print(f"  📦 {league['name']} {season}: loaded from cache ({len(cache[cache_key])} players)")
            all_players.extend(cache[cache_key])
            continue

        print(f"\n📥 Fetching {league['name']} {season}...")
        season_players = []

        for page in range(1, PAGES_PER_FETCH + 1):
            print(f"   Page {page}...", end=" ")
            data = fetch_players(league["id"], season, page)
            requests_used += 1

            if "response" not in data or not data["response"]:
                print("no data")
                break

            for item in data["response"]:
                try:
                    season_players.append(parse_player(item, season, league["name"]))
                except Exception as e:
                    pass

            total_pages = data.get("paging", {}).get("total", 1)
            print(f"got {len(data['response'])} players (page {page}/{total_pages})")

            if page >= total_pages:
                break

            # Respect API rate limit — 30 requests per minute
            time.sleep(2.5)

        print(f"  ✅ {league['name']} {season}: {len(season_players)} players")

        # Save to cache
        cache[cache_key] = season_players
        save_cache(cache)
        all_players.extend(season_players)

        # Longer pause between league/season combos
        time.sleep(3)

# ── BUILD DATAFRAME ───────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Total records fetched: {len(all_players):,}")

df = pd.DataFrame(all_players)

# Convert numeric columns
for col in ["goals","assists","minutes","appearances","shots_total",
            "shots_on_target","pass_accuracy","dribbles","tackles",
            "interceptions","rating","yellow_cards","red_cards","duels_won"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ── PER 90 STATS ──────────────────────────────────────────────────────────────
df["mins_safe"]    = df["minutes"].replace(0, float("nan"))
df["goals_p90"]    = (df["goals"]          / df["mins_safe"] * 90).round(2)
df["assists_p90"]  = (df["assists"]         / df["mins_safe"] * 90).round(2)
df["shots_p90"]    = (df["shots_total"]     / df["mins_safe"] * 90).round(2)
df["sot_p90"]      = (df["shots_on_target"] / df["mins_safe"] * 90).round(2)
df["tackles_p90"]  = (df["tackles"]         / df["mins_safe"] * 90).round(2)
df["interc_p90"]   = (df["interceptions"]   / df["mins_safe"] * 90).round(2)
df["contrib_p90"]  = ((df["goals"] + df["assists"]) / df["mins_safe"] * 90).round(2)
df["cards_p90"]    = ((df["yellow_cards"] + df["red_cards"]) / df["mins_safe"] * 90).round(2)
df = df.fillna(0)

# ── SAVE ──────────────────────────────────────────────────────────────────────
df.to_csv("data/processed/player_stats.csv", index=False)

print(f"\n✅ Saved {len(df):,} player records!")
print(f"API requests used: {requests_used}")
print(f"\nBreakdown:")
print(df.groupby(["league","season"])["name"].count().to_string())
print(f"\nSample:")
print(df[["name","team","league","season","goals","assists","rating"]].head(10).to_string())