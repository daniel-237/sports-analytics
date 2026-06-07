import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("APIFOOTBALL_KEY")

headers = {
    "x-apisports-key": API_KEY,
    "x-rapidapi-key":  API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

BASE_URL = "https://v3.football.api-sports.io"

# Check available seasons for Premier League (id=39)
print("Checking available seasons for Premier League...")
response = requests.get(
    f"{BASE_URL}/leagues",
    headers=headers,
    params={"id": "39"}
)
data = response.json()

if "response" in data and data["response"]:
    league = data["response"][0]
    seasons = league.get("seasons", [])
    print(f"\nAvailable seasons:")
    for s in seasons:
        print(f"  {s['year']} — coverage: {s['coverage']}")
else:
    print("Error:", data)