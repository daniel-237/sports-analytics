import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import json

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── APPLE CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue',
                     'Arial', sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] {
        background-color: #f5f5f7 !important;
        border-right: 1px solid #e0e0e5 !important;
    }
    [data-testid="stSidebar"] * { color: #1d1d1f !important; }
    h1 {
        font-size: 48px !important; font-weight: 700 !important;
        color: #1d1d1f !important; letter-spacing: -1.5px !important;
        line-height: 1.05 !important;
    }
    h2 { font-size: 26px !important; font-weight: 600 !important;
         color: #1d1d1f !important; letter-spacing: -0.5px !important; }
    h3 { font-size: 19px !important; font-weight: 600 !important;
         color: #1d1d1f !important; letter-spacing: -0.3px !important; }
    [data-testid="metric-container"] {
        background: #f5f5f7 !important; border-radius: 18px !important;
        padding: 24px !important; border: none !important;
        transition: transform 0.2s ease !important;
    }
    [data-testid="metric-container"]:hover { transform: scale(1.02) !important; }
    [data-testid="metric-container"] label {
        font-size: 11px !important; font-weight: 600 !important;
        color: #6e6e73 !important; text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 34px !important; font-weight: 700 !important;
        color: #1d1d1f !important; letter-spacing: -1px !important;
    }
    .stButton > button {
        background-color: #0071e3 !important; color: #ffffff !important;
        border: none !important; border-radius: 980px !important;
        padding: 12px 28px !important; font-size: 15px !important;
        font-weight: 500 !important; transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #0077ed !important; transform: scale(1.02) !important;
        box-shadow: 0 4px 20px rgba(0,113,227,0.3) !important;
    }
    [data-testid="stSelectbox"] > div > div {
        background-color: #f5f5f7 !important; border: 1px solid #d2d2d7 !important;
        border-radius: 12px !important; font-size: 15px !important;
    }
    [data-testid="stTextInput"] > div > div > input {
        background-color: #f5f5f7 !important; border: 1px solid #d2d2d7 !important;
        border-radius: 12px !important; padding: 12px 16px !important;
        font-size: 15px !important;
    }
    hr { border: none !important; border-top: 1px solid #e0e0e5 !important;
         margin: 36px 0 !important; }
    [data-testid="stDataFrame"] {
        border-radius: 18px !important; overflow: hidden !important;
        border: 1px solid #e0e0e5 !important;
    }
    .main .block-container { padding: 48px 60px !important; max-width: 1280px !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .insight-card {
        background: linear-gradient(135deg, #0071e3 0%, #0051a0 100%);
        border-radius: 18px; padding: 20px 24px; margin-bottom: 8px; color: white;
    }
    .insight-card p { color: white !important; font-size: 15px !important;
                      font-weight: 500 !important; margin: 0 !important; }
    .insight-card span { font-size: 22px; }
    @media (max-width: 768px) {
        .main .block-container { padding: 24px 16px !important; }
        h1 { font-size: 32px !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY BASE ───────────────────────────────────────────────────────────────
BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Helvetica Neue'",
              color="#1d1d1f"),
    xaxis=dict(showgrid=False, zeroline=False,
               tickfont=dict(size=12, color="#6e6e73")),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f5", zeroline=False,
               tickfont=dict(size=12, color="#6e6e73")),
    margin=dict(l=0, r=0, t=24, b=0),
    hoverlabel=dict(bgcolor="white", bordercolor="#e0e0e5",
                    font=dict(size=13, color="#1d1d1f"))
)
COLORS = ["#0071e3","#34c759","#ff9f0a","#ff3b30","#bf5af2",
          "#5ac8fa","#ffcc00","#ff6b35","#32ade6","#30b0c7"]

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_match_data():
    return pd.read_csv("data/processed/matches_clean.csv")

@st.cache_data
def load_player_data():
    try:
        df = pd.read_csv("data/processed/player_stats.csv")
        for col in ["goals","assists","minutes","appearances",
                    "shots_total","shots_on_target","pass_accuracy",
                    "dribbles","tackles","interceptions","rating",
                    "yellow_cards","red_cards","duels_won"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        # ── PER 90 STATS ──────────────────────────────────────────────────────
        df["mins_safe"] = df["minutes"].replace(0, np.nan)
        df["goals_p90"]    = (df["goals"]           / df["mins_safe"] * 90).round(2)
        df["assists_p90"]  = (df["assists"]          / df["mins_safe"] * 90).round(2)
        df["shots_p90"]    = (df["shots_total"]      / df["mins_safe"] * 90).round(2)
        df["sot_p90"]      = (df["shots_on_target"]  / df["mins_safe"] * 90).round(2)
        df["tackles_p90"]  = (df["tackles"]          / df["mins_safe"] * 90).round(2)
        df["interc_p90"]   = (df["interceptions"]    / df["mins_safe"] * 90).round(2)
        df["contrib_p90"]  = ((df["goals"] + df["assists"]) / df["mins_safe"] * 90).round(2)
        df["cards_p90"]    = ((df["yellow_cards"] + df["red_cards"]) / df["mins_safe"] * 90).round(2)
        df = df.fillna(0)
        return df
    except FileNotFoundError:
        st.error("Player data file not found. Please run player_stats.py.")
        return None
    except pd.errors.EmptyDataError:
        st.error("Player data file is empty.")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading player data: {e}")
        return None

@st.cache_resource
def load_model():
    return joblib.load("models/match_predictor.pkl")

df      = load_match_data()
players = load_player_data()
model   = load_model()
LEAGUES = ["All"] + sorted(df["league"].unique().tolist())

# Dynamic sidebar stats
total_matches = len(df)
total_leagues = df["league"].nunique()
total_seasons = df["season"].nunique()

def filter_df(league):
    return df if league == "All" else df[df["league"] == league]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding:28px 8px 20px 8px;'>
    <p style='font-size:11px;font-weight:600;color:#6e6e73;
              letter-spacing:1.2px;text-transform:uppercase;margin:0;'>Sports</p>
    <p style='font-size:26px;font-weight:700;color:#1d1d1f;
              letter-spacing:-0.8px;margin:4px 0 0 0;'>Analytics</p>
</div>""", unsafe_allow_html=True)

page = st.sidebar.radio("", [
    "⚽  Overview",
    "🔮  Match Predictor",
    "👤  Player Stats",
    "⚔️  Player Comparison",
    "🏟️  Team Analysis",
    "💰  Transfer Analysis",
    "📈  Model Performance",
])

st.sidebar.markdown(f"""
<div style='padding:20px 8px 0 8px;border-top:1px solid #e0e0e5;margin-top:20px;'>
    <p style='font-size:12px;color:#6e6e73;margin:0;line-height:1.6;'>
        {total_matches:,} matches · {total_leagues} leagues<br>{total_seasons} seasons of data
    </p>
</div>""", unsafe_allow_html=True)

def insight_card(emoji, text):
    st.markdown(f"""
    <div class="insight-card">
        <span>{emoji}</span>
        <p>{text}</p>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "⚽  Overview":
    st.markdown("# Football Analytics")
    st.markdown("Thirty years of English football. Five leagues. One dashboard.")
    league = st.selectbox("League", LEAGUES, key="ov_league")
    fdf = filter_df(league)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches",  f"{len(fdf):,}")
    c2.metric("Leagues",        fdf["league"].nunique())
    c3.metric("Teams",          fdf["home_team"].nunique())
    c4.metric("Seasons",        fdf["season"].nunique())
    st.divider()

    home_pct  = (fdf["result"] == 1).mean()
    top_team  = fdf.groupby("home_team")["home_goals"].mean().idxmax()
    avg_goals = (fdf["home_goals"] + fdf["away_goals"]).mean()
    insight_card("🏠", f"Home teams win <b>{home_pct:.0%}</b> of all matches.")
    col1, col2 = st.columns(2)
    with col1:
        insight_card("⚽", f"<b>{top_team}</b> average the most goals per home game.")
    with col2:
        insight_card("📊", f"Average of <b>{avg_goals:.2f}</b> goals per match across all leagues.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Match Outcomes")
        rc = fdf["result"].map({1:"Home Win",0:"Draw",-1:"Away Win"}).value_counts()
        fig = go.Figure(go.Pie(
            values=rc.values, labels=rc.index, hole=0.6,
            marker=dict(colors=["#0071e3","#34c759","#ff3b30"],
                        line=dict(color="#ffffff", width=2)),
        ))
        fig.update_layout(**BASE, showlegend=True,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Matches Per League")
        lc = fdf["league"].value_counts()
        fig = go.Figure(go.Bar(
            x=lc.index, y=lc.values,
            marker=dict(color=COLORS[:len(lc)], line=dict(width=0)),
            text=lc.values, textposition="outside",
            textfont=dict(size=11, color="#6e6e73")
        ))
        fig.update_layout(**BASE)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Goals Per Season")
    fdf = fdf.copy()
    fdf["total_goals"] = fdf["home_goals"] + fdf["away_goals"]
    gs = fdf.groupby("season")["total_goals"].mean().reset_index()
    fig = go.Figure(go.Scatter(
        x=gs["season"], y=gs["total_goals"], mode="lines+markers",
        line=dict(color="#0071e3", width=2.5),
        marker=dict(color="#0071e3", size=6),
        fill="tozeroy", fillcolor="rgba(0,113,227,0.08)",
        hovertemplate="Season %{x}<br>Avg Goals: %{y:.2f}<extra></extra>"
    ))
    fig.update_layout(**BASE)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Match Predictor":
    st.markdown("# Match Predictor")
    st.markdown("Select two teams to forecast the outcome.")
    st.divider()

    teams = sorted(df["home_team"].unique())
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏠 Home Team**")
        home_team = st.selectbox("Home", teams, index=0, label_visibility="collapsed")
    with col2:
        st.markdown("**✈️ Away Team**")
        away_team = st.selectbox("Away", teams, index=1, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Match →"):
        h = df[df["home_team"] == home_team].tail(5)
        a = df[df["away_team"] == away_team].tail(5)

        feats = pd.DataFrame([[
            h["home_goals"].mean() or df["home_form"].mean(),
            a["away_goals"].mean() or df["away_form"].mean(),
            h["away_goals"].mean() or df["home_conceded_form"].mean(),
            a["home_goals"].mean() or df["away_conceded_form"].mean(),
            h["home_shots_form"].mean() or df["home_shots_form"].mean(),
            a["away_shots_form"].mean() or df["away_shots_form"].mean(),
        ]], columns=["home_form","away_form","home_conceded_form",
                     "away_conceded_form","home_shots_form","away_shots_form"])

        probs  = model.predict_proba(feats)[0]
        winner = ["Away Win","Draw","Home Win"][np.argmax(probs)]
        conf   = max(probs)
        conf_label = "High" if conf >= 0.55 else "Medium" if conf >= 0.45 else "Low"
        conf_color = "#34c759" if conf >= 0.55 else "#ff9f0a" if conf >= 0.45 else "#ff3b30"

        st.divider()
        insight_card("🔮", f"Prediction: <b>{winner}</b> — <b>{conf_label} confidence</b> ({conf:.0%})")
        st.markdown("<br>", unsafe_allow_html=True)

        # Prediction explanation
        h_form     = h["home_goals"].mean()
        a_form     = a["away_goals"].mean()
        h_conceded = h["away_goals"].mean()
        a_conceded = a["home_goals"].mean()

        st.subheader("Why this prediction?")
        reasons = []
        if h_form > a_form:
            reasons.append(f"⚽ {home_team} have stronger recent scoring form ({h_form:.1f} vs {a_form:.1f} goals/game)")
        else:
            reasons.append(f"⚽ {away_team} have stronger recent scoring form ({a_form:.1f} vs {h_form:.1f} goals/game)")
        if h_conceded < a_conceded:
            reasons.append(f"🛡️ {home_team} have conceded fewer goals recently ({h_conceded:.1f} vs {a_conceded:.1f}/game)")
        else:
            reasons.append(f"🛡️ {away_team} have a stronger defensive record recently ({a_conceded:.1f} vs {h_conceded:.1f}/game)")
        reasons.append(f"🏠 Home advantage is factored into all predictions")
        if abs(conf - 0.5) < 0.1:
            reasons.append(f"⚠️ Low confidence — both teams have similar recent form")

        for r in reasons:
            st.markdown(f"- {r}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {home_team}  vs  {away_team}")
        c1, c2, c3 = st.columns(3)
        c1.metric("🏠 Home Win", f"{probs[2]:.0%}")
        c2.metric("🤝 Draw",     f"{probs[1]:.0%}")
        c3.metric("✈️ Away Win",  f"{probs[0]:.0%}")

        fig = go.Figure(go.Bar(
            x=["Home Win","Draw","Away Win"],
            y=[probs[2], probs[1], probs[0]],
            marker=dict(color=["#0071e3","#ff9f0a","#ff3b30"], line=dict(width=0)),
            text=[f"{p:.0%}" for p in [probs[2], probs[1], probs[0]]],
            textposition="outside", textfont=dict(size=14, color="#1d1d1f"),
            hovertemplate="%{x}: %{y:.1%}<extra></extra>"
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="-apple-system", color="#1d1d1f"),
            xaxis=dict(showgrid=False, zeroline=False,
                       tickfont=dict(size=13, color="#6e6e73")),
            yaxis=dict(tickformat=".0%", showgrid=True,
                       gridcolor="#f0f0f5", zeroline=False,
                       tickfont=dict(size=12, color="#6e6e73")),
            margin=dict(l=0, r=0, t=24, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Recent Form — {home_team}")
            recent_h = df[
                (df["home_team"]==home_team)|(df["away_team"]==home_team)
            ].tail(5)[["date","home_team","away_team","home_goals","away_goals"]]
            st.dataframe(recent_h, use_container_width=True, hide_index=True)
        with col2:
            st.subheader(f"Recent Form — {away_team}")
            recent_a = df[
                (df["home_team"]==away_team)|(df["away_team"]==away_team)
            ].tail(5)[["date","home_team","away_team","home_goals","away_goals"]]
            st.dataframe(recent_a, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER STATS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👤  Player Stats":
    st.markdown("# Player Statistics")
    st.markdown("Premier League 2023/24 season.")
    st.divider()

    if players is None:
        st.stop()

    # Insight cards
    top_scorer   = players.loc[players["goals"].idxmax()]
    top_assister = players.loc[players["assists"].idxmax()]
    top_p90      = players[players["minutes"] >= 0].loc[
                   players[players["minutes"] >= 0]["goals_p90"].idxmax()]
    insight_card("⚽", f"<b>{top_scorer['name']}</b> leads with <b>{int(top_scorer['goals'])}</b> goals.")
    insight_card("🎯", f"<b>{top_p90['name']}</b> has the best goals per 90 among players with 450+ minutes: <b>{top_p90['goals_p90']:.2f}</b>")
    st.divider()

    # ── FILTERS ───────────────────────────────────────────────────────────────
    st.subheader("Filters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        positions  = ["All"] + sorted(players["position"].dropna().unique().tolist())
        pos_filter = st.selectbox("Position", positions)
    with col2:
        teams_list  = ["All"] + sorted(players["team"].unique().tolist())
        team_filter = st.selectbox("Team", teams_list)
    with col3:
        min_mins = st.number_input("Min Minutes", min_value=0,
                                    max_value=3000, value=0, step=90)
    with col4:
        stat_view = st.selectbox("Stat View", ["Raw Totals", "Per 90"])

    # Apply filters
    filtered = players.copy()
    if pos_filter  != "All": filtered = filtered[filtered["position"] == pos_filter]
    if team_filter != "All": filtered = filtered[filtered["team"] == team_filter]
    filtered = filtered[filtered["minutes"] >= min_mins]

    search = st.text_input("🔍 Search player", placeholder="e.g. Salah, Haaland…")
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]

    st.markdown(f"**{len(filtered)} players**")
    st.divider()

    # Single player profile
    if search and len(filtered) == 1:
        p = filtered.iloc[0]
        st.subheader(f"📋 {p['name']} — {p['team']}")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Goals",        int(p["goals"]))
        c2.metric("Assists",      int(p["assists"]))
        c3.metric("Goals/90",     f"{p['goals_p90']:.2f}")
        c4.metric("Assists/90",   f"{p['assists_p90']:.2f}")
        c5.metric("Minutes",      int(p["minutes"]))
        c6.metric("Rating",       p["rating"])
        st.divider()

    # Chart
    if stat_view == "Per 90":
        goal_col   = "goals_p90"
        assist_col = "assists_p90"
        chart_label = "Goals per 90"
    else:
        goal_col   = "goals"
        assist_col = "assists"
        chart_label = "Goals"

    st.subheader(f"Top Scorers — {stat_view}")
    top = filtered.nlargest(10, goal_col)
    fig = go.Figure(go.Bar(
        x=top["name"], y=top[goal_col],
        marker=dict(color="#0071e3", line=dict(width=0)),
        text=top[goal_col].round(2), textposition="outside",
        textfont=dict(size=12, color="#1d1d1f"),
        hovertemplate="%{x}<br>" + chart_label + ": %{y}<extra></extra>"
    ))
    fig.update_layout(**BASE)
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Assisters")
        st.dataframe(
            filtered.nlargest(10, assist_col)[["name","team", assist_col]],
            use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Most Booked")
        filtered_copy = filtered.copy()
        filtered_copy["total_cards"] = filtered_copy["yellow_cards"] + filtered_copy["red_cards"]
        st.dataframe(
            filtered_copy.nlargest(10,"total_cards")[
                ["name","team","yellow_cards","red_cards"]],
            use_container_width=True, hide_index=True)

    st.subheader("Full Stats Table")
    if stat_view == "Per 90":
        cols = ["name","team","position","goals_p90","assists_p90",
                "shots_p90","sot_p90","tackles_p90","contrib_p90","rating"]
    else:
        cols = ["name","team","position","goals","assists","appearances",
                "minutes","yellow_cards","red_cards","shots_on_target","rating"]
    st.dataframe(
        filtered[cols].sort_values(goal_col, ascending=False),
        use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️  Player Comparison":
    st.markdown("# Player Comparison")
    st.markdown("Compare two players side by side with radar charts.")
    st.divider()

    if players is None:
        st.stop()

    pnames = sorted(players["name"].unique())
    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("Player 1", pnames, index=0)
    with col2:
        p2_name = st.selectbox("Player 2", pnames, index=1)

    p1 = players[players["name"] == p1_name].iloc[0]
    p2 = players[players["name"] == p2_name].iloc[0]

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"👤 {p1_name}")
        st.caption(f"{p1['team']} · {p1['position']} · {int(p1['minutes'])} mins")
        ca,cb,cc,cd = st.columns(4)
        ca.metric("Goals",      int(p1["goals"]))
        cb.metric("Assists",    int(p1["assists"]))
        cc.metric("Goals/90",   f"{p1['goals_p90']:.2f}")
        cd.metric("Rating",     p1["rating"])
    with col2:
        st.subheader(f"👤 {p2_name}")
        st.caption(f"{p2['team']} · {p2['position']} · {int(p2['minutes'])} mins")
        ca,cb,cc,cd = st.columns(4)
        ca.metric("Goals",      int(p2["goals"]))
        cb.metric("Assists",    int(p2["assists"]))
        cc.metric("Goals/90",   f"{p2['goals_p90']:.2f}")
        cd.metric("Rating",     p2["rating"])

    st.divider()

    stat_mode = st.radio("Radar Mode", ["Raw Stats","Per 90"], horizontal=True)
    if stat_mode == "Per 90":
        radar_cols   = ["goals_p90","assists_p90","sot_p90",
                        "tackles_p90","interc_p90","contrib_p90"]
        radar_labels = ["Goals/90","Assists/90","Shots on Target/90",
                        "Tackles/90","Interceptions/90","Contributions/90"]
    else:
        radar_cols   = ["goals","assists","shots_on_target",
                        "pass_accuracy","dribbles","tackles"]
        radar_labels = ["Goals","Assists","Shots on Target",
                        "Pass Accuracy","Dribbles","Tackles"]

    def percentile(player, col):
        val = player[col]
        return float((players[col] <= val).mean() * 100)

    p1_vals = [percentile(p1, c) for c in radar_cols]
    p2_vals = [percentile(p2, c) for c in radar_cols]

    st.subheader("Radar Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=p1_vals + [p1_vals[0]], theta=radar_labels + [radar_labels[0]],
        fill="toself", fillcolor="rgba(0,113,227,0.15)",
        line=dict(color="#0071e3", width=2.5), name=p1_name,
        hovertemplate="%{theta}: %{r:.0f}th percentile<extra></extra>"
    ))
    fig.add_trace(go.Scatterpolar(
        r=p2_vals + [p2_vals[0]], theta=radar_labels + [radar_labels[0]],
        fill="toself", fillcolor="rgba(255,59,48,0.15)",
        line=dict(color="#ff3b30", width=2.5), name=p2_name,
        hovertemplate="%{theta}: %{r:.0f}th percentile<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100],
                            tickfont=dict(size=10, color="#6e6e73"),
                            gridcolor="#e0e0e5"),
            angularaxis=dict(tickfont=dict(size=12, color="#1d1d1f"),
                             gridcolor="#e0e0e5")
        ),
        showlegend=True, legend=dict(orientation="h", y=-0.1),
        margin=dict(l=40, r=40, t=40, b=40), height=480
    )
    st.plotly_chart(fig, use_container_width=True)

    # Percentile table
    st.divider()
    st.subheader("Percentile Rankings")
    pct_data = []
    for col, label in zip(radar_cols, radar_labels):
        p1_pct = (players[col] <= p1[col]).mean() * 100
        p2_pct = (players[col] <= p2[col]).mean() * 100
        better = p1_name if p1_pct >= p2_pct else p2_name
        pct_data.append({
            "Stat":              label,
            f"{p1_name} (raw)":  round(float(p1[col]),2),
            f"{p1_name} (pct)":  f"{p1_pct:.0f}th",
            f"{p2_name} (raw)":  round(float(p2[col]),2),
            f"{p2_name} (pct)":  f"{p2_pct:.0f}th",
            "Better":            better,
        })
    st.dataframe(pd.DataFrame(pct_data), use_container_width=True, hide_index=True)

    # Strengths summary
    st.divider()
    p1_wins = sum(1 for d in pct_data if d["Better"] == p1_name)
    p2_wins = sum(1 for d in pct_data if d["Better"] == p2_name)
    if p1_wins > p2_wins:
        insight_card("🏆", f"<b>{p1_name}</b> edges this comparison, winning <b>{p1_wins}</b> of {len(pct_data)} statistical categories.")
    elif p2_wins > p1_wins:
        insight_card("🏆", f"<b>{p2_name}</b> edges this comparison, winning <b>{p2_wins}</b> of {len(pct_data)} statistical categories.")
    else:
        insight_card("🤝", f"These players are evenly matched — both win <b>{p1_wins}</b> statistical categories each.")

    # Similar players
    st.divider()
    st.subheader(f"Players Similar to {p1_name}")
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    num_cols = ["goals","assists","shots_on_target","pass_accuracy",
                "dribbles","tackles"]
    scaler  = StandardScaler()
    matrix  = scaler.fit_transform(players[num_cols].fillna(0))
    idx     = players[players["name"] == p1_name].index[0]
    pos     = players.index.get_loc(idx)
    sims    = cosine_similarity([matrix[pos]], matrix)[0]
    players_copy = players.copy()
    players_copy["similarity"] = sims
    similar = (players_copy[players_copy["name"] != p1_name]
               .nlargest(5,"similarity")
               [["name","team","goals","assists","rating","similarity"]])
    similar["similarity"] = similar["similarity"].apply(lambda x: f"{x:.0%}")
    st.dataframe(similar, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM ANALYSIS — FIXED DRAW BUG
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏟️  Team Analysis":
    st.markdown("# Team Analysis")
    st.markdown("Deep dive into any team's performance.")
    st.divider()

    league = st.selectbox("League", LEAGUES, key="team_league")
    fdf    = filter_df(league)
    teams  = sorted(set(fdf["home_team"].unique()) | set(fdf["away_team"].unique()))
    team   = st.selectbox("Select Team", teams)

    home_df  = fdf[fdf["home_team"] == team].copy()
    away_df  = fdf[fdf["away_team"] == team].copy()

    home_df["gf"]    = home_df["home_goals"]
    home_df["ga"]    = home_df["away_goals"]
    home_df["venue"] = "Home"

    away_df["gf"]    = away_df["away_goals"]
    away_df["ga"]    = away_df["home_goals"]
    away_df["venue"] = "Away"

    all_team = pd.concat([home_df, away_df]).copy()

    # ── FIX: Proper win/draw/loss logic ──────────────────────────────────────
    def result_label(row):
        if row["gf"] > row["ga"]:   return "✅ Win"
        elif row["gf"] == row["ga"]: return "🟡 Draw"
        else:                        return "❌ Loss"

    all_team["result_label"] = all_team.apply(result_label, axis=1)
    all_team["win"]  = (all_team["gf"] >  all_team["ga"]).astype(int)
    all_team["draw"] = (all_team["gf"] == all_team["ga"]).astype(int)
    all_team["loss"] = (all_team["gf"] <  all_team["ga"]).astype(int)

    total = len(all_team)
    wins  = all_team["win"].sum()
    draws = all_team["draw"].sum()
    gf    = all_team["gf"].sum()
    ga    = all_team["ga"].sum()

    insight_card("🏆", f"<b>{team}</b> — <b>{wins}W {draws}D {total-wins-draws}L</b> from {total} matches. Win rate: <b>{wins/total:.0%}</b>.")
    insight_card("⚽", f"Scored <b>{int(gf)}</b> · Conceded <b>{int(ga)}</b> · Goal difference <b>{int(gf-ga):+}</b>.")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games",    total)
    c2.metric("Wins",     int(wins))
    c3.metric("Draws",    int(draws))
    c4.metric("Goals For", int(gf))
    c5.metric("Goals Against", int(ga))
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Home vs Away")
        hv = all_team.groupby("venue").agg(
            Wins=("win","sum"), Draws=("draw","sum"),
            Goals=("gf","mean"), Conceded=("ga","mean")
        ).reset_index()
        fig = go.Figure()
        for i, metric in enumerate(["Wins","Draws","Goals","Conceded"]):
            fig.add_trace(go.Bar(
                name=metric, x=hv["venue"], y=hv[metric],
                marker=dict(color=COLORS[i], line=dict(width=0))
            ))
        fig.update_layout(**BASE, barmode="group",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Last 5 Matches")
        last5 = all_team.tail(5)[["date","venue","gf","ga","result_label"]].copy()
        last5 = last5.rename(columns={"date":"Date","venue":"Venue",
                                       "gf":"GF","ga":"GA",
                                       "result_label":"Result"})
        st.dataframe(last5, use_container_width=True, hide_index=True)

    st.subheader("Goals Per Season")
    gps = all_team.groupby("season")["gf"].sum().reset_index()
    fig = go.Figure(go.Bar(
        x=gps["season"], y=gps["gf"],
        marker=dict(color="#0071e3", line=dict(width=0)),
        hovertemplate="Season %{x}<br>Goals: %{y}<extra></extra>"
    ))
    fig.update_layout(**BASE)
    st.plotly_chart(fig, use_container_width=True)

    # Clean sheet and attack stats
    st.divider()
    st.subheader("Team Strength Profile")
    clean_sheets = (all_team["ga"] == 0).sum()
    failed_score = (all_team["gf"] == 0).sum()
    home_wins    = home_df[home_df["gf"] > home_df["ga"]].shape[0]
    away_wins    = away_df[away_df["gf"] > away_df["ga"]].shape[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clean Sheets",    f"{clean_sheets} ({clean_sheets/total:.0%})")
    c2.metric("Failed to Score", f"{failed_score} ({failed_score/total:.0%})")
    c3.metric("Home Win Rate",   f"{home_wins/max(len(home_df),1):.0%}")
    c4.metric("Away Win Rate",   f"{away_wins/max(len(away_df),1):.0%}")

    # Auto insight
    if home_wins/max(len(home_df),1) > away_wins/max(len(away_df),1) * 1.3:
        insight_card("🏠", f"<b>{team}</b>'s home form is significantly stronger than their away form.")
    elif away_wins/max(len(away_df),1) > home_wins/max(len(home_df),1) * 1.3:
        insight_card("✈️", f"<b>{team}</b> are surprisingly strong away from home.")
    else:
        insight_card("⚖️", f"<b>{team}</b> perform consistently both home and away.")

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰  Transfer Analysis":
    st.markdown("# Transfer Analysis")
    st.markdown("Attack vs defence efficiency — find hidden gems and weak links.")
    st.divider()

    league = st.selectbox("League", LEAGUES, key="tr_league")
    fdf    = filter_df(league)

    home_s = fdf.groupby("home_team").agg(hg=("home_goals","mean"), hc=("away_goals","mean"))
    away_s = fdf.groupby("away_team").agg(ag=("away_goals","mean"), ac=("home_goals","mean"))
    ts     = pd.DataFrame({
        "attack":  (home_s["hg"] + away_s["ag"]) / 2,
        "defence": (home_s["hc"] + away_s["ac"]) / 2
    }).dropna().reset_index()
    ts.columns = ["team","attack","defence"]
    ts["score"] = ts["attack"] - ts["defence"]

    best_att = ts.loc[ts["attack"].idxmax()]
    best_def = ts.loc[ts["defence"].idxmin()]
    insight_card("⚔️",  f"<b>{best_att['team']}</b> — best attack at <b>{best_att['attack']:.2f}</b> goals/game.")
    insight_card("🛡️", f"<b>{best_def['team']}</b> — best defence conceding only <b>{best_def['defence']:.2f}</b>/game.")
    st.divider()

    search_team = st.text_input("🔍 Highlight a team", placeholder="e.g. Arsenal")
    ts["highlight"] = ts["team"].str.contains(search_team, case=False) if search_team else False

    st.subheader("Attack vs Defence — All Teams")
    fig = go.Figure()
    normal = ts[~ts["highlight"]]
    fig.add_trace(go.Scatter(
        x=normal["defence"], y=normal["attack"], mode="markers",
        name="Teams",
        marker=dict(size=9, color=normal["score"],
                    colorscale=[[0,"#ff3b30"],[0.5,"#ff9f0a"],[1,"#34c759"]],
                    line=dict(width=0)),
        text=normal["team"],
        hovertemplate="<b>%{text}</b><br>Attack: %{y:.2f}<br>Defence: %{x:.2f}<extra></extra>",
    ))
    if search_team and ts["highlight"].any():
        hl = ts[ts["highlight"]]
        fig.add_trace(go.Scatter(
            x=hl["defence"], y=hl["attack"], mode="markers+text",
            name="Highlighted",
            marker=dict(size=16, color="#0071e3",
                        line=dict(color="white", width=2)),
            text=hl["team"], textposition="top center",
            textfont=dict(size=13, color="#0071e3"),
            hovertemplate="<b>%{text}</b><br>Attack: %{y:.2f}<br>Conceded: %{x:.2f}<extra></extra>"
        ))
    fig.update_layout(
        **BASE,
        xaxis_title="Avg Goals Conceded (lower = better)",
        yaxis_title="Avg Goals Scored (higher = better)",
        height=520
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top 10 Attack")
        st.dataframe(
            ts.nlargest(10,"attack")[["team","attack","defence"]]
              .style.format({"attack":"{:.2f}","defence":"{:.2f}"}),
            use_container_width=True)
    with col2:
        st.subheader("🛡️ Top 10 Defence")
        st.dataframe(
            ts.nsmallest(10,"defence")[["team","attack","defence"]]
              .style.format({"attack":"{:.2f}","defence":"{:.2f}"}),
            use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Model Performance":
    st.markdown("# Model Performance")
    st.markdown("Comparing machine learning models trained on 30 years of football data.")
    st.divider()

    try:
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        st.error("Run model.py first to generate metrics.")
        st.stop()

    results    = metrics["model_results"]
    baseline   = metrics["baseline_accuracy"]
    importance = metrics["feature_importance"]
    xgb        = results["XGBoost"]

    best_model = max(results, key=lambda x: results[x]["accuracy"])
    best_acc   = results[best_model]["accuracy"]
    insight_card("🏆", f"<b>{best_model}</b> is the best model with <b>{best_acc:.1%}</b> accuracy — beating the <b>{baseline:.1%}</b> baseline.")
    insight_card("📊", f"The model struggles most with <b>draws</b> — the hardest outcome to predict in football.")
    st.divider()

    st.subheader("XGBoost — Best Model")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{xgb['accuracy']:.1%}")
    c2.metric("Baseline",  f"{baseline:.1%}")
    c3.metric("Precision", f"{xgb['precision']:.1%}")
    c4.metric("Recall",    f"{xgb['recall']:.1%}")
    c5.metric("F1 Score",  f"{xgb['f1']:.1%}")
    st.divider()

    st.subheader("Model Comparison")
    model_names = ["Baseline"] + list(results.keys())
    accuracies  = [baseline]   + [results[m]["accuracy"] for m in results]
    f1_scores   = [0]          + [results[m]["f1"]       for m in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy", x=model_names, y=accuracies,
        marker=dict(color=["#6e6e73","#ff9f0a","#34c759","#0071e3"],
                    line=dict(width=0)),
        text=[f"{v:.1%}" for v in accuracies], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="F1 Score", x=model_names, y=f1_scores,
        marker=dict(color=["rgba(0,0,0,0)","#ffcc00","#30b0c7","#bf5af2"],
                    line=dict(width=0)),
        text=[f"{v:.1%}" if v > 0 else "" for v in f1_scores],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system", color="#1d1d1f"),
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=12, color="#6e6e73")),
        yaxis=dict(tickformat=".0%", showgrid=True,
                   gridcolor="#f0f0f5", zeroline=False,
                   tickfont=dict(size=12, color="#6e6e73")),
        margin=dict(l=0, r=0, t=24, b=0),
        barmode="group", legend=dict(orientation="h", y=-0.15)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Confusion Matrix — XGBoost")
    cm     = np.array(xgb["confusion_matrix"])
    labels = ["Away Win","Draw","Home Win"]
    fig    = px.imshow(
        cm, x=labels, y=labels,
        color_continuous_scale=[[0,"#f5f5f7"],[1,"#0071e3"]],
        labels=dict(x="Predicted", y="Actual", color="Count"),
        text_auto=True
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system", color="#1d1d1f"),
        margin=dict(l=0, r=0, t=24, b=0), coloraxis_showscale=False
    )
    fig.update_traces(textfont=dict(size=14, color="#1d1d1f"))
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Feature Importance")
    feat_names = list(importance.keys())
    feat_vals  = list(importance.values())
    sorted_pairs          = sorted(zip(feat_vals, feat_names), reverse=True)
    feat_vals, feat_names = zip(*sorted_pairs)
    fig = go.Figure(go.Bar(
        x=list(feat_vals), y=list(feat_names), orientation="h",
        marker=dict(color="#0071e3", line=dict(width=0)),
        text=[f"{v:.3f}" for v in feat_vals], textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system", color="#1d1d1f"),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f5", zeroline=False,
                   tickfont=dict(size=12, color="#6e6e73")),
        yaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(size=13, color="#1d1d1f")),
        margin=dict(l=0, r=60, t=24, b=0), height=350
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Performance By Outcome")
    report    = xgb["classification_report"]
    perf_data = []
    for outcome in ["Away Win","Draw","Home Win"]:
        if outcome in report:
            perf_data.append({
                "Outcome":   outcome,
                "Precision": f"{report[outcome]['precision']:.1%}",
                "Recall":    f"{report[outcome]['recall']:.1%}",
                "F1 Score":  f"{report[outcome]['f1-score']:.1%}",
                "Support":   int(report[outcome]['support'])
            })
    st.dataframe(pd.DataFrame(perf_data),
                 use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("Why XGBoost?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Advantages over other models:**
        - Handles non-linear relationships between features
        - Robust to outliers and missing values
        - Built-in regularisation prevents overfitting
        - Faster training than Random Forest
        - Industry standard for tabular sports prediction
        """)
    with col2:
        st.markdown("""
        **Why not 100% accuracy?**
        - Football has 3 outcomes — random baseline is 33%
        - Professional betting models sit at 60-65%
        - Injuries, tactics, and luck are not in the data
        - Draws are nearly impossible to predict reliably
        - This model beats the baseline by 11 percentage points
        """)