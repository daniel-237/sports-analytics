import pandas as pd
import requests
import time

seasons = [
    "9394","9495","9596","9697","9798","9899",
    "9900","0001","0102","0203","0304","0405",
    "0506","0607","0708","0809","0910","1011",
    "1112","1213","1314","1415","1516","1617",
    "1718","1819","1920","2021","2122","2223"
]

# All English football leagues
leagues = {
    "E0": "Premier League",
    "E1": "Championship", 
    "E2": "League One",
    "E3": "League Two",
    "EC": "Conference"
}

all_data = []

for league_code, league_name in leagues.items():
    print(f"\n📥 Downloading {league_name}...")
    
    for season in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
        
        try:
            df = pd.read_csv(url)
            df["season"] = season
            df["league"] = league_name
            df["league_code"] = league_code
            all_data.append(df)
            print(f"  ✅ {season}: {len(df)} matches")
        except:
            print(f"  ⚠️ {season}: not available")
        
        time.sleep(0.5)

# Combine everything
combined = pd.concat(all_data, ignore_index=True)
combined.to_csv("data/raw/matches_all_leagues.csv", index=False)

print(f"\n🎉 DONE!")
print(f"Total matches: {len(combined):,}")
print(f"Leagues: {combined['league'].unique()}")
print(f"Seasons: {combined['season'].nunique()} seasons")
seasons = [
    "9394","9495","9596","9697","9798","9899",
    "9900","0001","0102","0203","0304","0405",
    "0506","0607","0708","0809","0910","1011",
    "1112","1213","1314","1415","1516","1617",
    "1718","1819","1920","2021","2122","2223",
    "2324","2425"  
]