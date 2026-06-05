import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Sports Analytics Dashboard",
    page_icon="⚽",
    layout="wide"
)

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/matches_clean.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/match_predictor.pkl")

df = load_data()
model = load_model()

# ── SIDEBAR ──────────────────────────────────────
st.sidebar.title("⚽ Sports Analytics")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔮 Match Predictor",
    "👤 Player Stats",
    "💰 Transfer Analysis"
])

# ── OVERVIEW PAGE ─────────────────────────────────
if page == "🏠 Overview":
    st.title("Premier League Analytics Dashboard")
    st.markdown("### 30 Years of English Football Data")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", f"{len(df):,}")
    col2.metric("Leagues", df["league"].nunique())
    col3.metric("Teams", df["home_team"].nunique())
    col4.metric("Seasons", df["season"].nunique())

    st.divider()

    # Results breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Match Results Breakdown")
        result_counts = df["result"].map(
            {1: "Home Win", 0: "Draw", -1: "Away Win"}
        ).value_counts()
        fig = px.pie(
            values=result_counts.values,
            names=result_counts.index,
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Matches Per League")
        league_counts = df["league"].value_counts()
        fig = px.bar(
            x=league_counts.index,
            y=league_counts.values,
            color=league_counts.index,
            labels={"x": "League", "y": "Matches"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Goals over time
    st.subheader("Average Goals Per Season")
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    goals_by_season = df.groupby("season")["total_goals"].mean().reset_index()
    fig = px.line(
        goals_by_season,
        x="season", y="total_goals",
        markers=True,
        labels={"total_goals": "Avg Goals", "season": "Season"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ── MATCH PREDICTOR PAGE ──────────────────────────
elif page == "🔮 Match Predictor":
    st.title("🔮 Match Outcome Predictor")
    st.markdown("Select two teams and predict the match outcome")

    teams = sorted(df["home_team"].unique())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, index=0)
    with col2:
        away_team = st.selectbox("✈️ Away Team", teams, index=1)

    if st.button("Predict Match", type="primary"):
        # Get recent form for each team
        home_stats = df[df["home_team"] == home_team].tail(5)
        away_stats = df[df["away_team"] == away_team].tail(5)

        home_form = home_stats["home_goals"].mean()
        away_form = away_stats["away_goals"].mean()
        home_conceded = home_stats["away_goals"].mean()
        away_conceded = away_stats["home_goals"].mean()
        home_shots = home_stats["home_shots_form"].mean()
        away_shots = away_stats["away_shots_form"].mean()

        # Fill NaN with league average
        features = pd.DataFrame([[
            home_form or df["home_form"].mean(),
            away_form or df["away_form"].mean(),
            home_conceded or df["home_conceded_form"].mean(),
            away_conceded or df["away_conceded_form"].mean(),
            home_shots or df["home_shots_form"].mean(),
            away_shots or df["away_shots_form"].mean()
        ]], columns=[
            "home_form", "away_form",
            "home_conceded_form", "away_conceded_form",
            "home_shots_form", "away_shots_form"
        ])

        probs = model.predict_proba(features)[0]

        st.divider()
        st.subheader(f"🏟️ {home_team} vs {away_team}")

        col1, col2, col3 = st.columns(3)
        col1.metric("🏠 Home Win", f"{probs[2]:.1%}")
        col2.metric("🤝 Draw", f"{probs[1]:.1%}")
        col3.metric("✈️ Away Win", f"{probs[0]:.1%}")

        # Probability bar chart
        fig = go.Figure(go.Bar(
            x=["Home Win", "Draw", "Away Win"],
            y=[probs[2], probs[1], probs[0]],
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"]
        ))
        fig.update_layout(
            yaxis_tickformat=".0%",
            yaxis_title="Probability"
        )
        st.plotly_chart(fig, use_container_width=True)

# ── TEAM STATS PAGE ───────────────────────────────
elif page == "👤 Player Stats":
    st.title("📊 Team Performance Stats")

    league_filter = st.selectbox("Select League", 
                                  ["All"] + list(df["league"].unique()))
    
    if league_filter != "All":
        filtered_df = df[df["league"] == league_filter]
    else:
        filtered_df = df

    teams = sorted(filtered_df["home_team"].unique())
    selected_team = st.selectbox("Select Team", teams)

    team_home = filtered_df[filtered_df["home_team"] == selected_team]
    team_away = filtered_df[filtered_df["away_team"] == selected_team]

    col1, col2, col3, col4 = st.columns(4)
    total_games = len(team_home) + len(team_away)
    home_wins = len(team_home[team_home["result"] == 1])
    away_wins = len(team_away[team_away["result"] == -1])
    total_goals = team_home["home_goals"].sum() + team_away["away_goals"].sum()

    col1.metric("Total Games", total_games)
    col2.metric("Home Wins", home_wins)
    col3.metric("Away Wins", away_wins)
    col4.metric("Goals Scored", int(total_goals))

    # Goals per season
    st.subheader(f"{selected_team} — Goals Per Season")
    home_goals = team_home.groupby("season")["home_goals"].sum()
    away_goals = team_away.groupby("season")["away_goals"].sum()
    goals_combined = (home_goals.add(away_goals, fill_value=0)).reset_index()
    goals_combined.columns = ["season", "goals"]

    fig = px.bar(goals_combined, x="season", y="goals",
                 labels={"goals": "Goals", "season": "Season"})
    st.plotly_chart(fig, use_container_width=True)

# ── TRANSFER ANALYSIS PAGE ────────────────────────
elif page == "💰 Transfer Analysis":
    st.title("💰 Transfer Market Analysis")
    st.markdown("Teams ranked by attacking and defensive performance")

    league_filter = st.selectbox("Select League",
                                  ["All"] + list(df["league"].unique()))

    if league_filter != "All":
        filtered_df = df[df["league"] == league_filter]
    else:
        filtered_df = df

    # Calculate team performance scores
    home_stats = filtered_df.groupby("home_team").agg(
        home_goals=("home_goals", "mean"),
        home_conceded=("away_goals", "mean")
    )
    away_stats = filtered_df.groupby("away_team").agg(
        away_goals=("away_goals", "mean"),
        away_conceded=("home_goals", "mean")
    )

    team_stats = pd.DataFrame({
        "attack": (home_stats["home_goals"] + 
                   away_stats["away_goals"]) / 2,
        "defence": (home_stats["home_conceded"] + 
                    away_stats["away_conceded"]) / 2
    }).dropna()

    team_stats["performance"] = (
        team_stats["attack"] - team_stats["defence"]
    )

    st.subheader("⚔️ Attack vs Defence — Team Scatter")
    fig = px.scatter(
        team_stats.reset_index(),
        x="defence", y="attack",
        text="home_team",
        color="performance",
        color_continuous_scale="RdYlGn",
        labels={
            "defence": "Avg Goals Conceded (lower = better)",
            "attack": "Avg Goals Scored (higher = better)"
        }
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Top 10 Attacking Teams")
    top_attack = team_stats.nlargest(10, "attack")[["attack", "defence"]]
    st.dataframe(top_attack.style.format("{:.2f}"), use_container_width=True)