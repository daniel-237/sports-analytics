![CI](https://github.com/daniel-237/sports-analytics/actions/workflows/ci.yml/badge.svg)
# ⚽ Football Analytics Platform

A machine learning powered football intelligence dashboard for analysing team performance, player output, match predictions, and recruitment insights across English football.

**[🚀 Live Demo](https://daniel237-football-analytics.streamlit.app)** · **[GitHub](https://github.com/daniel-237/sports-analytics)**

---

## Screenshots

> Overview · Match Predictor · Player Comparison · Transfer Analysis

---

## Problem Statement

Football clubs, analysts, and enthusiasts need tools to make sense of decades of match data. This platform turns 30 years of English football into actionable insights — predicting match outcomes, comparing players, identifying hidden gems, and analysing team weaknesses.

---

## Features

| Page | Description |
|---|---|
| ⚽ Overview | 55,143 matches across 5 leagues and 29 seasons with dynamic insight cards |
| 🔮 Match Predictor | XGBoost model predicting Home Win / Draw / Away Win with confidence scores |
| 👤 Player Stats | Goals, assists, cards, ratings with search and per 90 filtering |
| ⚔️ Player Comparison | Radar charts, percentile rankings, and cosine similarity player finder |
| 🏟️ Team Analysis | Form, home vs away splits, goals trend, and team strength profiles |
| 💰 Transfer Analysis | Attack vs defence scatter, hidden gems, scouting shortlist builder |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Core language |
| Pandas | Data manipulation and feature engineering |
| Scikit-learn | Preprocessing, model evaluation, cosine similarity |
| XGBoost | Match outcome prediction model |
| Streamlit | Interactive web dashboard |
| Plotly | Charts and radar visualisations |
| football-data.org API | Live Premier League match data |
| api-football.com | Player statistics |
| football-data.co.uk | 30 years of historical match CSVs |

---

## Dataset

- **Source:** football-data.co.uk, football-data.org API, api-football.com
- **Coverage:** 1993/94 to 2023/24
- **Leagues:** Premier League, Championship, League One, League Two, National League
- **Total matches:** 55,143
- **Teams:** 157
- **Seasons:** 29
- **Player data:** Premier League 2023/24 via API Football (100 req/day free tier)

### Data Cleaning
- Removed rows with missing scorelines
- Standardised team names across seasons
- Converted date formats to datetime
- Filled missing form stats with rolling league averages
- Excluded matches with status other than FINISHED

### Known Limitations
- Injuries and lineups are not included
- Transfers mid-season are not accounted for
- Some seasons have gaps due to missing source data
- Player data is limited to 60 players on the free API tier
- Weather and tactical changes are not modelled

---

## Machine Learning

### Model
XGBoost multi-class classifier predicting three outcomes: Home Win, Draw, Away Win.

### Features
| Feature | Description |
|---|---|
| home_form | Rolling 5-game avg goals scored at home |
| away_form | Rolling 5-game avg goals scored away |
| home_conceded_form | Rolling 5-game avg goals conceded at home |
| away_conceded_form | Rolling 5-game avg goals conceded away |
| home_shots_form | Rolling 5-game avg shots at home |
| away_shots_form | Rolling 5-game avg shots away |

### Train / Test Split
Chronological split — trained on seasons up to 2021, tested on 2022 onwards. This simulates real-world prediction where future matches are unknown.

### Model Performance
| Metric | Score |
|---|---|
| Accuracy | 44.7% |
| Baseline (always predict home win) | 33.3% |
| Home Win Recall | 86% |

### Why XGBoost?
- Handles non-linear relationships between features
- Robust to missing values
- Faster and more accurate than Random Forest on this dataset
- Industry standard for tabular sports prediction

### Why Not 100% Accuracy?
Football is genuinely unpredictable. Even professional betting models with GPS tracking, heart rate data, and tactical analysis sit at 60-65%. A model beating 33% random baseline on real historical data is meaningful.

---

## App Pages

### ⚽ Overview
High-level stats with dynamic insight cards. League filter applies across all charts. Shows match outcome distribution, goals per season trend, and matches per league.

### 🔮 Match Predictor
Select any two teams. The model returns Home Win / Draw / Away Win probabilities. Shows recent form for both teams alongside the prediction.

### 👤 Player Stats
Top scorers, assisters, and disciplinary records. Player search returns a dedicated profile when exactly one player matches. Full stats table sortable by any column.

### ⚔️ Player Comparison
Select any two players. Radar chart shows percentile rankings across 6 dimensions. Cosine similarity engine finds the 5 most similar players in the dataset.

### 🏟️ Team Analysis
Select any team across any league. Shows home vs away performance, last 5 matches form, and goals per season bar chart. Insight cards summarise win rate and goal difference automatically.

### 💰 Transfer Analysis
Attack vs defence scatter plot across all teams. Hover for team details, search to highlight a specific team. Top 10 attack and defence tables included.

---

## How To Run Locally

```bash
git clone https://github.com/daniel-237/sports-analytics
cd sports-analytics
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
FOOTBALL_API_KEY=your_key
APIFOOTBALL_KEY=your_key
```

Fetch data:
```bash
python src/download_historical.py
python src/data_ingestion.py
python src/feature_engineering.py
python src/model.py
python src/player_stats.py
```

Run the dashboard:
```bash
streamlit run src/dashboard.py
```

---

## Folder Structure

```
sports-analytics/
├── data/
│   ├── raw/                  # Downloaded CSVs
│   └── processed/            # Cleaned and engineered data
├── models/
│   └── match_predictor.pkl   # Saved XGBoost model
├── notebooks/                # Exploratory analysis
├── src/
│   ├── dashboard.py          # Streamlit app
│   ├── data_ingestion.py     # API data fetching
│   ├── download_historical.py# Historical CSV download
│   ├── feature_engineering.py# Feature creation
│   ├── model.py              # Model training
│   └── player_stats.py       # Player data fetching
├── .env                      # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Key Technical Decisions

| Decision | Reason |
|---|---|
| Streamlit over Flask/Django | Faster iteration for data dashboards, no frontend code needed |
| XGBoost over Random Forest | Better performance on tabular data, handles missing values natively |
| Chronological train/test split | Simulates real prediction — future matches must be unknown to the model |
| Rolling averages over season totals | Captures recent form rather than cumulative bias |
| Per 90 stats over raw totals | Fairer comparison across players with different minutes |
| Cosine similarity for player matching | Efficient, interpretable, and works well on scaled stat vectors |

---

## Key Learnings

- Football prediction is genuinely hard — draws are nearly impossible to predict reliably
- Feature engineering matters more than model choice for sports data
- Chronological splits are essential for time-series sports models
- Rolling averages of 5 games capture form better than season averages
- Per 90 stats change player rankings significantly compared to raw totals

---

## Future Improvements

- Add xG (expected goals) as a feature
- Add Elo ratings for team strength
- Add lineup and injury data
- Add SHAP explanations for individual predictions
- Add a recruitment shortlist builder
- Add league table probability predictions
- Automate daily data updates
- Add Docker containerisation
- Add CI/CD pipeline with GitHub Actions

---

## Data Sources

- [football-data.co.uk](https://www.football-data.co.uk) — Historical match results
- [football-data.org](https://www.football-data.org) — Live Premier League API
- [api-football.com](https://www.api-football.com) — Player statistics API

---

*Built by Daniel Olutade*
