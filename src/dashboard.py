import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

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
    /* Insight card */
    .insight-card {
        background: linear-gradient(135deg, #0071e3 0%, #0051a0 100%);
        border-radius: 18px; padding: 20px 24px; margin-bottom: 8px;
        color: white;
    }
    .insight-card p { color: white !important; font-size: 15px !important;
                      font-weight: 500 !important; margin: 0 !important; }
    .insight-card span { font-size: 22px; }
    /* Mobile */
    @media (max-width: 768px) {
        .main .block-container { padding: 24px 16px !important; }
        h1 { font-size: 32px !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY BASE LAYOUT ────────────────────────────────────────────────────────
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
        # Convert numeric cols safely
        for col in ["goals","assists","minutes","appearances",
                    "shots_total","shots_on_target","pass_accuracy",
                    "dribbles","tackles","interceptions","rating",
                    "yellow_cards","red_cards","duels_won"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    except:
        return None

@st.cache_resource
def load_model():
    return joblib.load("models/match_predictor.pkl")

df      = load_match_data()
players = load_player_data()
model   = load_model()

LEAGUES = ["All"] + sorted(df["league"].unique().tolist())

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
])

st.sidebar.markdown("""
<div style='padding:20px 8px 0 8px;border-top:1px solid #e0e0e5;margin-top:20px;'>
    <p style='font-size:12px;color:#6e6e73;margin:0;line-height:1.6;'>
        55,143 matches · 5 leagues<br>30 seasons of data
    </p>
</div>""", unsafe_allow_html=True)

# ── INSIGHT CARD HELPER ───────────────────────────────────────────────────────
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

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Matches",  f"{len(fdf):,}")
    c2.metric("Leagues",        fdf["league"].nunique())
    c3.metric("Teams",          fdf["home_team"].nunique())
    c4.metric("Seasons",        fdf["season"].nunique())

    st.divider()

    # Insight cards
    home_pct = (fdf["result"] == 1).mean()
    top_team  = (fdf.groupby("home_team")["home_goals"].mean()
                   .idxmax())
    avg_goals = (fdf["home_goals"] + fdf["away_goals"]).mean()
    insight_card("🏠", f"Home teams win <b>{home_pct:.0%}</b> of all matches.")
    col1, col2 = st.columns(2)
    with col1:
        insight_card("⚽", f"<b>{top_team}</b> average the most goals per home game.")
    with col2:
        insight_card("📊", f"Average of <b>{avg_goals:.2f}</b> goals per match across all leagues.")

    st.divider()

    # Charts
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
        x=gs["season"], y=gs["total_goals"],
        mode="lines+markers",
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
        home_team = st.selectbox("Home", teams, index=0,
                                  label_visibility="collapsed")
    with col2:
        st.markdown("**✈️ Away Team**")
        away_team = st.selectbox("Away", teams, index=1,
                                  label_visibility="collapsed")

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

        probs = model.predict_proba(feats)[0]
        winner = (["Away Win","Draw","Home Win"])[np.argmax(probs)]
        confidence = max(probs)

        st.divider()
        insight_card("🔮", f"Prediction: <b>{winner}</b> with <b>{confidence:.0%}</b> confidence.")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"### {home_team}  vs  {away_team}")
        c1, c2, c3 = st.columns(3)
        c1.metric("🏠 Home Win", f"{probs[2]:.0%}")
        c2.metric("🤝 Draw",     f"{probs[1]:.0%}")
        c3.metric("✈️ Away Win",  f"{probs[0]:.0%}")
        st.markdown("<br>", unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=["Home Win","Draw","Away Win"],
            y=[probs[2], probs[1], probs[0]],
            marker=dict(color=["#0071e3","#ff9f0a","#ff3b30"],
                        line=dict(width=0)),
            text=[f"{p:.0%}" for p in [probs[2], probs[1], probs[0]]],
            textposition="outside",
            textfont=dict(size=14, color="#1d1d1f"),
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

        # Recent form
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
        st.error("Run player_stats.py first to fetch player data.")
        st.stop()

    # Insight cards
    top_scorer   = players.loc[players["goals"].idxmax()]
    top_assister = players.loc[players["assists"].idxmax()]
    insight_card("⚽", f"<b>{top_scorer['name']}</b> leads the league with <b>{int(top_scorer['goals'])}</b> goals.")
    insight_card("🎯", f"<b>{top_assister['name']}</b> tops the assists chart with <b>{int(top_assister['assists'])}</b>.")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players",       len(players))
    c2.metric("Total Goals",   int(players["goals"].sum()))
    c3.metric("Total Assists", int(players["assists"].sum()))
    c4.metric("Avg Rating",    f"{players['rating'].mean():.2f}")

    st.divider()

    # Search
    search = st.text_input("🔍 Search player",
                            placeholder="e.g. Salah, Haaland, Kane…")
    filtered = (players[players["name"].str.contains(search, case=False, na=False)]
                if search else players)

    # Player profile on search
    if search and len(filtered) == 1:
        p = filtered.iloc[0]
        st.subheader(f"📋 {p['name']} — {p['team']}")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Goals",       int(p["goals"]))
        c2.metric("Assists",     int(p["assists"]))
        c3.metric("Appearances", int(p["appearances"]))
        c4.metric("Minutes",     int(p["minutes"]))
        c5.metric("Rating",      p["rating"])
        st.divider()

    st.subheader("Top Scorers")
    top = filtered.nlargest(10, "goals")
    fig = go.Figure(go.Bar(
        x=top["name"], y=top["goals"],
        marker=dict(color="#0071e3", line=dict(width=0)),
        text=top["goals"], textposition="outside",
        textfont=dict(size=12, color="#1d1d1f"),
        hovertemplate="%{x}<br>Goals: %{y}<extra></extra>"
    ))
    fig.update_layout(**BASE)
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Assisters")
        st.dataframe(
            filtered.nlargest(10,"assists")[["name","team","assists"]],
            use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Most Booked")
        players["total_cards"] = players["yellow_cards"] + players["red_cards"]
        st.dataframe(
            players.nlargest(10,"total_cards")[
                ["name","team","yellow_cards","red_cards"]],
            use_container_width=True, hide_index=True)

    st.subheader("Full Stats Table")
    st.dataframe(
        filtered[["name","team","goals","assists","appearances",
                  "minutes","yellow_cards","red_cards",
                  "shots_on_target","rating"]]
        .sort_values("goals", ascending=False),
        use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️  Player Comparison":
    st.markdown("# Player Comparison")
    st.markdown("Compare two players side by side with radar charts.")
    st.divider()

    if players is None:
        st.error("Run player_stats.py first to fetch player data.")
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

    # Side by side metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"👤 {p1_name}")
        st.caption(p1["team"])
        ca, cb, cc = st.columns(3)
        ca.metric("Goals",       int(p1["goals"]))
        cb.metric("Assists",     int(p1["assists"]))
        cc.metric("Rating",      p1["rating"])
    with col2:
        st.subheader(f"👤 {p2_name}")
        st.caption(p2["team"])
        ca, cb, cc = st.columns(3)
        ca.metric("Goals",       int(p2["goals"]))
        cb.metric("Assists",     int(p2["assists"]))
        cc.metric("Rating",      p2["rating"])

    st.divider()

    # Radar chart
    st.subheader("Radar Comparison")

    # Normalise stats to 0-100 percentile
    radar_cols = ["goals","assists","shots_on_target",
                  "pass_accuracy","dribbles","tackles"]
    radar_labels = ["Goals","Assists","Shots on Target",
                    "Pass Accuracy","Dribbles","Tackles"]

    def percentile(player, col):
        val = player[col]
        return float(np.percentile(
            [100 * (val >= players[col]).mean()], 50))

    p1_vals = [percentile(p1, c) for c in radar_cols]
    p2_vals = [percentile(p2, c) for c in radar_cols]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=p1_vals + [p1_vals[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        fillcolor="rgba(0,113,227,0.15)",
        line=dict(color="#0071e3", width=2.5),
        name=p1_name,
        hovertemplate="%{theta}: %{r:.0f}th percentile<extra></extra>"
    ))
    fig.add_trace(go.Scatterpolar(
        r=p2_vals + [p2_vals[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        fillcolor="rgba(255,59,48,0.15)",
        line=dict(color="#ff3b30", width=2.5),
        name=p2_name,
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
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=480
    )
    st.plotly_chart(fig, use_container_width=True)

    # Percentile table
    st.divider()
    st.subheader("Percentile Rankings")
    pct_data = []
    for col, label in zip(radar_cols, radar_labels):
        p1_pct = (players[col] <= p1[col]).mean() * 100
        p2_pct = (players[col] <= p2[col]).mean() * 100
        pct_data.append({
            "Stat": label,
            f"{p1_name} (raw)": int(p1[col]),
            f"{p1_name} (pct)": f"{p1_pct:.0f}th",
            f"{p2_name} (raw)": int(p2[col]),
            f"{p2_name} (pct)": f"{p2_pct:.0f}th",
        })
    st.dataframe(pd.DataFrame(pct_data),
                 use_container_width=True, hide_index=True)

    # Similar players
    st.divider()
    st.subheader(f"Players Similar to {p1_name}")
    num_cols = ["goals","assists","shots_on_target","pass_accuracy",
                "dribbles","tackles"]
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    scaler   = StandardScaler()
    matrix   = scaler.fit_transform(players[num_cols].fillna(0))
    idx      = players[players["name"] == p1_name].index[0]
    pos      = players.index.get_loc(idx)
    sims     = cosine_similarity([matrix[pos]], matrix)[0]
    players  = players.copy()
    players["similarity"] = sims
    similar  = (players[players["name"] != p1_name]
                .nlargest(5, "similarity")
                [["name","team","goals","assists","rating","similarity"]])
    similar["similarity"] = similar["similarity"].apply(lambda x: f"{x:.0%}")
    st.dataframe(similar, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏟️  Team Analysis":
    st.markdown("# Team Analysis")
    st.markdown("Deep dive into any team's performance.")
    st.divider()

    league = st.selectbox("League", LEAGUES, key="team_league")
    fdf = filter_df(league)
    teams = sorted(set(fdf["home_team"].unique()) |
                   set(fdf["away_team"].unique()))
    team  = st.selectbox("Select Team", teams)

    home_df  = fdf[fdf["home_team"] == team]
    away_df  = fdf[fdf["away_team"] == team]
    all_team = pd.concat([
        home_df.assign(venue="Home",
                       gf=home_df["home_goals"],
                       ga=home_df["away_goals"],
                       win=(home_df["result"]==1).astype(int)),
        away_df.assign(venue="Away",
                       gf=away_df["away_goals"],
                       ga=away_df["home_goals"],
                       win=(away_df["result"]==-1).astype(int))
    ])

    total  = len(all_team)
    wins   = all_team["win"].sum()
    gf     = all_team["gf"].sum()
    ga     = all_team["ga"].sum()

    # Insight cards
    insight_card("🏆", f"<b>{team}</b> have won <b>{wins}</b> of their <b>{total}</b> matches — a win rate of <b>{wins/total:.0%}</b>.")
    insight_card("⚽", f"They've scored <b>{int(gf)}</b> goals and conceded <b>{int(ga)}</b> — a goal difference of <b>{int(gf-ga):+}</b>.")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Games",  total)
    c2.metric("Wins",         int(wins))
    c3.metric("Goals Scored", int(gf))
    c4.metric("Goals Conceded", int(ga))

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Home vs Away")
        hv = all_team.groupby("venue").agg(
            Wins=("win","sum"),
            Goals=("gf","mean"),
            Conceded=("ga","mean")
        ).reset_index()
        fig = go.Figure()
        for i, metric in enumerate(["Wins","Goals","Conceded"]):
            fig.add_trace(go.Bar(
                name=metric, x=hv["venue"], y=hv[metric],
                marker=dict(color=COLORS[i], line=dict(width=0))
            ))
        fig.update_layout(**BASE, barmode="group",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Last 5 Matches Form")
        last5 = all_team.tail(5)[["date","venue","gf","ga","win"]].copy()
        last5["Result"] = last5["win"].map({1:"✅ Win", 0:"❌ Loss"})
        last5 = last5.rename(columns={"date":"Date","venue":"Venue",
                                       "gf":"GF","ga":"GA"})
        st.dataframe(last5[["Date","Venue","GF","GA","Result"]],
                     use_container_width=True, hide_index=True)

    st.subheader("Goals Per Season")
    gps = all_team.groupby("season")["gf"].sum().reset_index()
    fig = go.Figure(go.Bar(
        x=gps["season"], y=gps["gf"],
        marker=dict(color="#0071e3", line=dict(width=0)),
        hovertemplate="Season %{x}<br>Goals: %{y}<extra></extra>"
    ))
    fig.update_layout(**BASE)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰  Transfer Analysis":
    st.markdown("# Transfer Analysis")
    st.markdown("Attack vs defence efficiency — find hidden gems and weak links.")
    st.divider()

    league = st.selectbox("League", LEAGUES, key="tr_league")
    fdf = filter_df(league)

    home_s = fdf.groupby("home_team").agg(
        hg=("home_goals","mean"), hc=("away_goals","mean"))
    away_s = fdf.groupby("away_team").agg(
        ag=("away_goals","mean"), ac=("home_goals","mean"))
    ts = pd.DataFrame({
        "attack":  (home_s["hg"] + away_s["ag"]) / 2,
        "defence": (home_s["hc"] + away_s["ac"]) / 2
    }).dropna().reset_index()
    ts.columns = ["team","attack","defence"]
    ts["score"] = ts["attack"] - ts["defence"]

    # Insight cards
    best_att = ts.loc[ts["attack"].idxmax()]
    best_def = ts.loc[ts["defence"].idxmin()]
    insight_card("⚔️",  f"<b>{best_att['team']}</b> have the best attack — averaging <b>{best_att['attack']:.2f}</b> goals per game.")
    insight_card("🛡️", f"<b>{best_def['team']}</b> have the best defence — conceding only <b>{best_def['defence']:.2f}</b> goals per game.")

    st.divider()

    # Team filter
    search_team = st.text_input("🔍 Highlight a team", placeholder="e.g. Arsenal")
    ts["highlight"] = ts["team"].str.contains(
        search_team, case=False) if search_team else False

    st.subheader("Attack vs Defence — All Teams")
    fig = go.Figure()

    # Non-highlighted teams
    normal = ts[~ts["highlight"]]
    fig.add_trace(go.Scatter(
        x=normal["defence"], y=normal["attack"],
        mode="markers", name="Teams",
        marker=dict(size=9, color=normal["score"],
                    colorscale=[[0,"#ff3b30"],[0.5,"#ff9f0a"],[1,"#34c759"]],
                    line=dict(width=0), showscale=False),
        text=normal["team"],
        hovertemplate="<b>%{text}</b><br>Attack: %{y:.2f}<br>Defence: %{x:.2f}<extra></extra>",
        customdata=normal["team"]
    ))

    # Highlighted team
    if search_team and ts["highlight"].any():
        hl = ts[ts["highlight"]]
        fig.add_trace(go.Scatter(
            x=hl["defence"], y=hl["attack"],
            mode="markers+text", name="Highlighted",
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