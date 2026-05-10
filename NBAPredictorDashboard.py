"""
NBA Matchup Predictor Dashboard

Run with:
    streamlit run app.py

Requires:
    matchups_v2.csv in the same directory.
    pip install streamlit xgboost scikit-learn pandas numpy scipy
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind


# ============================================================================
# Configuration
# ============================================================================

DATA_PATH = "matchups_v2.csv"
ELO_HOME_ADV = 100.0
ELO_TO_NETRATING = 0.04
ALTITUDE_TEAM_IDS = {1610612743, 1610612762}  # Nuggets, Jazz

# 30 NBA team IDs (stats.nba.com convention)
NBA_TEAMS = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612739: "Cleveland Cavaliers",
    1610612740: "New Orleans Pelicans",
    1610612741: "Chicago Bulls",
    1610612742: "Dallas Mavericks",
    1610612743: "Denver Nuggets",
    1610612744: "Golden State Warriors",
    1610612745: "Houston Rockets",
    1610612746: "LA Clippers",
    1610612747: "Los Angeles Lakers",
    1610612748: "Miami Heat",
    1610612749: "Milwaukee Bucks",
    1610612750: "Minnesota Timberwolves",
    1610612751: "Brooklyn Nets",
    1610612752: "New York Knicks",
    1610612753: "Orlando Magic",
    1610612754: "Indiana Pacers",
    1610612755: "Philadelphia 76ers",
    1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers",
    1610612758: "Sacramento Kings",
    1610612759: "San Antonio Spurs",
    1610612760: "Oklahoma City Thunder",
    1610612761: "Toronto Raptors",
    1610612762: "Utah Jazz",
    1610612763: "Memphis Grizzlies",
    1610612764: "Washington Wizards",
    1610612765: "Detroit Pistons",
    1610612766: "Charlotte Hornets",
}
NAME_TO_ID = {v: k for k, v in NBA_TEAMS.items()}

# Curated 50-feature list (matches FEATURE_LIST from build_features_nba_final.ipynb)
FEATURE_LIST = [
    'Elo_Win_Prob', 'Diff_Pregame_Elo', 'Diff_Power_Rating',
    'Diff_Pythag_Win_Pct', 'Diff_Pyth_Luck_Gap',
    'Diff_Net_Rating', 'Diff_Avg_Off_Eff', 'Diff_Avg_Def_Eff',
    'Diff_Win_Pct', 'Diff_Home_Win_Pct', 'Diff_Away_Win_Pct',
    'Diff_Last_14_Days_Win_Pct',
    'Diff_Win_Streak', 'Diff_Last5_Win_Pct', 'Diff_Last10_Win_Pct',
    'Diff_Form_Gap_5', 'Diff_Form_Gap_10',
    'H2H_Win_Pct', 'Is_Home', 'Team_Altitude_Dis', 'Opp_Altitude_Dis',
    'Diff_Avg_Score',
    'Diff_Days_Rest',
    'Team_Is_B2B', 'Opp_Is_B2B',
    'Team_Is_3in4', 'Opp_Is_3in4',
    'Team_Is_4in5', 'Opp_Is_4in5',
    'Diff_Avg_FGA', 'Diff_Avg_FGM3', 'Diff_Avg_FGA3', 'Diff_Avg_FTA',
    'Diff_Avg_OR', 'Diff_Avg_DR',
    'Diff_Avg_Ast', 'Diff_Avg_TO', 'Diff_Avg_Stl', 'Diff_Avg_Blk',
    'Diff_Avg_Opp_Score',
    'Diff_Avg_Opp_FGA', 'Diff_Avg_Opp_FGM3', 'Diff_Avg_Opp_FGA3', 'Diff_Avg_Opp_FTA',
    'Diff_Avg_Opp_OR', 'Diff_Avg_Opp_DR',
    'Diff_Avg_Opp_Ast', 'Diff_Avg_Opp_TO', 'Diff_Avg_Opp_Stl', 'Diff_Avg_Opp_Blk',
]

# 22 actionable features — what teams CAN influence by game-plan / effort.
# Excluded: Elo, Power Rating, Net Rating, Off/Def Eff, Pythagorean (cumulative-state),
#          all Win_Pct variants, win streaks, form gaps, H2H (history),
#          Is_Home, schedule density, Days_Rest, altitude (geographic / fixed).
ACTIONABLE_FEATURES = [
    'Diff_Avg_Score', 'Diff_Avg_FGA', 'Diff_Avg_FGM3', 'Diff_Avg_FGA3', 'Diff_Avg_FTA',
    'Diff_Avg_OR', 'Diff_Avg_DR',
    'Diff_Avg_Ast', 'Diff_Avg_TO', 'Diff_Avg_Stl', 'Diff_Avg_Blk',
    'Diff_Avg_Opp_Score',
    'Diff_Avg_Opp_FGA', 'Diff_Avg_Opp_FGM3', 'Diff_Avg_Opp_FGA3', 'Diff_Avg_Opp_FTA',
    'Diff_Avg_Opp_OR', 'Diff_Avg_Opp_DR',
    'Diff_Avg_Opp_Ast', 'Diff_Avg_Opp_TO', 'Diff_Avg_Opp_Stl', 'Diff_Avg_Opp_Blk',
]

# Plain-English labels and a hint about whether higher or lower is intuitively "good" for offense
FEATURE_DESCRIPTIONS = {
    'Diff_Avg_Score':         'Points scored',
    'Diff_Avg_FGA':           'Field goals attempted',
    'Diff_Avg_FGM3':          '3-pointers made',
    'Diff_Avg_FGA3':          '3-pointers attempted',
    'Diff_Avg_FTA':           'Free throws attempted',
    'Diff_Avg_OR':            'Offensive rebounds',
    'Diff_Avg_DR':            'Defensive rebounds',
    'Diff_Avg_Ast':           'Assists',
    'Diff_Avg_TO':            'Turnovers committed',
    'Diff_Avg_Stl':           'Steals',
    'Diff_Avg_Blk':           'Blocks',
    'Diff_Avg_Opp_Score':     'Points allowed (def)',
    'Diff_Avg_Opp_FGA':       'Opp FG attempts allowed',
    'Diff_Avg_Opp_FGM3':      'Opp 3-pointers allowed',
    'Diff_Avg_Opp_FGA3':      'Opp 3-attempts allowed',
    'Diff_Avg_Opp_FTA':       'Opp FT attempts (fouls)',
    'Diff_Avg_Opp_OR':        'Opp offensive rebs allowed',
    'Diff_Avg_Opp_DR':        'Opp defensive rebs',
    'Diff_Avg_Opp_Ast':       'Opp assists allowed',
    'Diff_Avg_Opp_TO':        'Opp turnovers (forced)',
    'Diff_Avg_Opp_Stl':       'Opp steals',
    'Diff_Avg_Opp_Blk':       'Opp blocks',
}


# ============================================================================
# Cached loaders (run once per session)
# ============================================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=['DayDate'])


@st.cache_resource
def train_xgboost(df):
    """One-shot XGBoost fit on all data (no CV — this is for inference, not evaluation)."""
    df_train = df.dropna(subset=['Diff_Avg_Score']).copy()
    X = df_train[FEATURE_LIST]
    y = df_train['Target_Win'].astype(int)
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.005,
        max_depth=2,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="rmse",
    )
    model.fit(X, y)
    return model


# ============================================================================
# XGBoost: synthesise "T1 vs T2 right now" feature row and predict
# ============================================================================

def build_synthetic_row(df, t1_id, t2_id, t1_is_home=True, recent_n=30):
    """Construct a feature vector representing a hypothetical T1 vs T2 game.

    Symmetric Diff_X features are estimated as `mean(T1's Diff_X over recent N games) -
    mean(T2's Diff_X over recent N games)`. Since each Diff_X averages out to roughly
    `team_X - league_avg_X` over many opponents, the subtraction approximates `T1_X - T2_X`,
    which is exactly the Diff_X we'd see in a hypothetical T1 vs T2 game.
    """
    t1_recent = (df[df['TeamID'] == t1_id]
                 .dropna(subset=['Diff_Avg_Score'])
                 .sort_values('DayDate').tail(recent_n))
    t2_recent = (df[df['TeamID'] == t2_id]
                 .dropna(subset=['Diff_Avg_Score'])
                 .sort_values('DayDate').tail(recent_n))
    if len(t1_recent) == 0 or len(t2_recent) == 0:
        return None

    feats = {}

    # Symmetric Diff_* features (skip ones we recompute below)
    skip = {'Diff_Days_Rest', 'Diff_Power_Rating'}
    diff_cols = [c for c in FEATURE_LIST if c.startswith('Diff_') and c not in skip]
    for c in diff_cols:
        feats[c] = float(t1_recent[c].mean() - t2_recent[c].mean())

    # Asymmetric / context features
    feats['Is_Home']       = 1 if t1_is_home else 0
    feats['Team_Is_B2B']   = 0   # default to no fatigue effects
    feats['Opp_Is_B2B']    = 0
    feats['Team_Is_3in4']  = 0
    feats['Opp_Is_3in4']   = 0
    feats['Team_Is_4in5']  = 0
    feats['Opp_Is_4in5']   = 0
    feats['Diff_Days_Rest'] = 0  # tied

    feats['Team_Altitude_Dis'] = int((not t1_is_home) and (t2_id in ALTITUDE_TEAM_IDS))
    feats['Opp_Altitude_Dis']  = int(t1_is_home and (t1_id in ALTITUDE_TEAM_IDS))

    h2h = df[(df['TeamID'] == t1_id) & (df['OppID'] == t2_id)]
    feats['H2H_Win_Pct'] = float(h2h['Target_Win'].mean()) if len(h2h) > 0 else 0.5

    # Recompute the two Elo-derived features from the synthetic Diff_Pregame_Elo
    hca = ELO_HOME_ADV if t1_is_home else -ELO_HOME_ADV
    feats['Elo_Win_Prob'] = float(1 / (1 + 10 ** (-(feats['Diff_Pregame_Elo'] + hca) / 400)))
    feats['Diff_Power_Rating'] = feats['Diff_Net_Rating'] + ELO_TO_NETRATING * feats['Diff_Pregame_Elo']

    return pd.DataFrame([feats])[FEATURE_LIST]


def predict_xgb(model, X_pred):
    return float(np.clip(model.predict(X_pred)[0], 0.001, 0.999))


# ============================================================================
# Logistic regression: T1 vs T2 historical games only
# ============================================================================

def analyze_matchup(df, t1_id, t2_id, actionable_features):
    """Train L2-regularised LR on T1 vs T2 historical games. Pair β with t-test p-values
    comparing actionable features in T1-won vs T1-lost games."""
    matchup = df[(df['TeamID'] == t1_id) & (df['OppID'] == t2_id)].copy()
    matchup = matchup.dropna(subset=actionable_features + ['Target_Win'])

    n_games = len(matchup)
    if n_games < 5:
        return None, n_games, "Need at least 5 historical games."
    if matchup['Target_Win'].nunique() < 2:
        return None, n_games, "Only one outcome in history (T1 always won or always lost)."

    X = matchup[actionable_features].values
    y = matchup['Target_Win'].astype(int).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='lbfgs')
    lr.fit(X_scaled, y)
    lr_predict = lr.predict_proba(X_scaled)[:, 1].mean()  # in-sample mean prob (sanity)

    rows = []
    for i, f in enumerate(actionable_features):
        won  = matchup.loc[matchup['Target_Win'] == 1, f]
        lost = matchup.loc[matchup['Target_Win'] == 0, f]
        won_mean  = float(won.mean())  if len(won)  > 0 else np.nan
        lost_mean = float(lost.mean()) if len(lost) > 0 else np.nan
        if len(won) > 1 and len(lost) > 1:
            t_stat, p_val = ttest_ind(won, lost, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan
        rows.append({
            'feature':     f,
            'description': FEATURE_DESCRIPTIONS.get(f, f),
            'beta':        float(lr.coef_[0][i]),
            'won_mean':    won_mean,
            'lost_mean':   lost_mean,
            't_stat':      float(t_stat) if not np.isnan(t_stat) else np.nan,
            'p_value':     float(p_val)  if not np.isnan(p_val)  else np.nan,
        })

    res = pd.DataFrame(rows)
    res['abs_beta'] = res['beta'].abs()
    res = res.sort_values('abs_beta', ascending=False).reset_index(drop=True)

    info = {
        'n_games':     n_games,
        'n_t1_wins':   int(y.sum()),
        'n_t1_losses': int(len(y) - y.sum()),
        'lr_in_sample_mean_prob': lr_predict,
    }
    return res, info, None

# ============================================================================
# Polished UI helpers
# ============================================================================

TEAM_META = {
    "Atlanta Hawks": {"abbr": "ATL", "color": "#e03a3e", "accent": "#c1d32f"},
    "Boston Celtics": {"abbr": "BOS", "color": "#007a33", "accent": "#ba9653"},
    "Cleveland Cavaliers": {"abbr": "CLE", "color": "#860038", "accent": "#FDBB30"},
    "New Orleans Pelicans": {"abbr": "NOP", "color": "#0c2340", "accent": "#c8102e"},
    "Chicago Bulls": {"abbr": "CHI", "color": "#ce1141", "accent": "#111111"},
    "Dallas Mavericks": {"abbr": "DAL", "color": "#00538c", "accent": "#b8c4ca"},
    "Denver Nuggets": {"abbr": "DEN", "color": "#0e2240", "accent": "#fec524"},
    "Golden State Warriors": {"abbr": "GSW", "color": "#1d428a", "accent": "#ffc72c"},
    "Houston Rockets": {"abbr": "HOU", "color": "#ce1141", "accent": "#c4ced4"},
    "LA Clippers": {"abbr": "LAC", "color": "#c8102e", "accent": "#1d428a"},
    "Los Angeles Lakers": {"abbr": "LAL", "color": "#552583", "accent": "#fdb927"},
    "Miami Heat": {"abbr": "MIA", "color": "#98002e", "accent": "#f9a01b"},
    "Milwaukee Bucks": {"abbr": "MIL", "color": "#00471b", "accent": "#eee1c6"},
    "Minnesota Timberwolves": {"abbr": "MIN", "color": "#0c2340", "accent": "#78be20"},
    "Brooklyn Nets": {"abbr": "BKN", "color": "#111111", "accent": "#ffffff"},
    "New York Knicks": {"abbr": "NYK", "color": "#006bb6", "accent": "#f58426"},
    "Orlando Magic": {"abbr": "ORL", "color": "#0077c0", "accent": "#c4ced4"},
    "Indiana Pacers": {"abbr": "IND", "color": "#002d62", "accent": "#fdbb30"},
    "Philadelphia 76ers": {"abbr": "PHI", "color": "#006bb6", "accent": "#ed174c"},
    "Phoenix Suns": {"abbr": "PHX", "color": "#1d1160", "accent": "#e56020"},
    "Portland Trail Blazers": {"abbr": "POR", "color": "#e03a3e", "accent": "#111111"},
    "Sacramento Kings": {"abbr": "SAC", "color": "#5a2d81", "accent": "#63727a"},
    "San Antonio Spurs": {"abbr": "SAS", "color": "#111111", "accent": "#c4ced4"},
    "Oklahoma City Thunder": {"abbr": "OKC", "color": "#007ac1", "accent": "#ef3b24"},
    "Toronto Raptors": {"abbr": "TOR", "color": "#ce1141", "accent": "#a1a1a4"},
    "Utah Jazz": {"abbr": "UTA", "color": "#002b5c", "accent": "#f9a01b"},
    "Memphis Grizzlies": {"abbr": "MEM", "color": "#5d76a9", "accent": "#12173f"},
    "Washington Wizards": {"abbr": "WAS", "color": "#002b5c", "accent": "#e31837"},
    "Detroit Pistons": {"abbr": "DET", "color": "#c8102e", "accent": "#1d42ba"},
    "Charlotte Hornets": {"abbr": "CHA", "color": "#1d1160", "accent": "#00788c"},
}


def get_team_meta(team_name: str) -> dict:
    return TEAM_META.get(team_name, {"abbr": team_name[:3].upper(), "color": "#f97316", "accent": "#facc15"})


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: #090b12;
            --panel: rgba(17, 24, 39, 0.78);
            --panel-2: rgba(30, 41, 59, 0.74);
            --border: rgba(148, 163, 184, 0.18);
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #f97316;
            --accent-2: #facc15;
            --good: #22c55e;
            --bad: #ef4444;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(249, 115, 22, 0.18), transparent 34rem),
                radial-gradient(circle at top right, rgba(250, 204, 21, 0.12), transparent 34rem),
                linear-gradient(135deg, #070910 0%, #0f172a 52%, #111827 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(2, 6, 23, 0.98));
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { color: #e5e7eb; }
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 4rem;
            max-width: 1320px;
        }
        h1, h2, h3 { letter-spacing: -0.03em; }
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.62);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.18);
        }
        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 32px;
            padding: 2rem;
            margin-bottom: 1.1rem;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.82), rgba(30, 41, 59, 0.70)),
                radial-gradient(circle at 15% 10%, rgba(249, 115, 22, 0.33), transparent 20rem),
                radial-gradient(circle at 90% 5%, rgba(250, 204, 21, 0.23), transparent 20rem);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.26);
        }
        .hero-title {
            font-size: 3.0rem;
            line-height: 1.02;
            font-weight: 900;
            margin: 0;
        }
        .hero-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            margin-top: .75rem;
            max-width: 850px;
        }
        .matchup-strip {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 1rem;
            align-items: stretch;
            margin-top: 1.4rem;
        }
        .team-panel {
            border-radius: 28px;
            padding: 1.15rem;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04), 0 16px 40px rgba(0,0,0,.18);
            min-height: 132px;
        }
        .team-abbr {
            font-size: 2.3rem;
            font-weight: 950;
            letter-spacing: -0.06em;
        }
        .team-name {
            font-size: 1.05rem;
            font-weight: 750;
            margin-top: .2rem;
        }
        .team-loc {
            display: inline-flex;
            gap: .4rem;
            align-items: center;
            margin-top: .7rem;
            padding: .28rem .62rem;
            border-radius: 999px;
            background: rgba(255,255,255,.14);
            font-size: .83rem;
            font-weight: 700;
        }
        .versus {
            align-self: center;
            justify-self: center;
            width: 64px;
            height: 64px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            font-size: 1.1rem;
            font-weight: 900;
            color: #0f172a;
            background: linear-gradient(135deg, #facc15, #fb923c);
            box-shadow: 0 18px 45px rgba(249, 115, 22, .35);
        }
        .glass-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 26px;
            padding: 1.25rem;
            box-shadow: 0 18px 45px rgba(0,0,0,.22);
            margin-bottom: 1rem;
        }
        .mini-card {
            background: rgba(15, 23, 42, 0.68);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem;
            min-height: 112px;
        }
        .mini-label { color: var(--muted); font-size: .82rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; }
        .mini-value { font-size: 1.75rem; font-weight: 900; margin-top: .35rem; }
        .mini-help { color: var(--muted); font-size: .86rem; margin-top: .25rem; }
        .prob-wrap {
            border-radius: 999px;
            height: 34px;
            overflow: hidden;
            display: flex;
            background: rgba(15, 23, 42, .8);
            border: 1px solid var(--border);
            box-shadow: inset 0 2px 8px rgba(0,0,0,.22);
        }
        .prob-left, .prob-right {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: .9rem;
            white-space: nowrap;
        }
        .prob-right { color: #f8fafc; }
        .edge-card {
            border-radius: 22px;
            border: 1px solid var(--border);
            padding: 1rem 1.1rem;
            margin: .65rem 0;
            background: linear-gradient(135deg, rgba(15,23,42,.78), rgba(30,41,59,.65));
            box-shadow: 0 14px 35px rgba(0,0,0,.18);
        }
        .edge-top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
        .edge-feature { font-weight: 900; font-size: 1.05rem; }
        .edge-badge { border-radius: 999px; padding: .28rem .62rem; font-size: .82rem; font-weight: 900; }
        .edge-meta { color: var(--muted); margin-top: .45rem; font-size: .92rem; }
        .soft-note {
            border-radius: 18px;
            border: 1px solid rgba(250, 204, 21, .26);
            background: rgba(250, 204, 21, .08);
            padding: .85rem 1rem;
            color: #fde68a;
        }
        .stTabs [data-baseweb="tab-list"] { gap: .45rem; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: .6rem 1.0rem;
            background: rgba(15, 23, 42, .55);
            border: 1px solid var(--border);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(249,115,22,.95), rgba(250,204,21,.85)) !important;
            color: #0f172a !important;
            font-weight: 900;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        @media (max-width: 900px) {
            .hero-title { font-size: 2.1rem; }
            .matchup-strip { grid-template-columns: 1fr; }
            .versus { width: 52px; height: 52px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def team_panel_html(team_name: str, is_home: bool, align: str = "left") -> str:
    meta = get_team_meta(team_name)
    loc = "Home court" if is_home else "Away"
    icon = "🏠" if is_home else "✈️"
    text_align = "right" if align == "right" else "left"
    return f"""
    <div class="team-panel" style="background: linear-gradient(135deg, {meta['color']}, rgba(15,23,42,.82)); text-align:{text_align};">
        <div class="team-abbr">{meta['abbr']}</div>
        <div class="team-name">{team_name}</div>
        <div class="team-loc">{icon} {loc}</div>
    </div>
    """


def hero_html(team1_name: str, team2_name: str, t1_is_home: bool) -> str:
    return f"""
    <div class="hero">
        <p style="margin:0; color:#facc15; font-weight:900; text-transform:uppercase; letter-spacing:.12em; font-size:.78rem;">NBA win probability lab</p>
        <h1 class="hero-title">Matchup Intelligence Dashboard</h1>
        <div class="hero-subtitle">
            Simulate a head-to-head game, inspect model confidence, and turn historical matchup patterns into coachable edges.
        </div>
        <div class="matchup-strip">
            {team_panel_html(team1_name, t1_is_home, 'left')}
            <div class="versus">VS</div>
            {team_panel_html(team2_name, not t1_is_home, 'right')}
        </div>
    </div>
    """


def prob_bar_html(team1_name: str, team2_name: str, p_t1: float) -> str:
    meta1 = get_team_meta(team1_name)
    meta2 = get_team_meta(team2_name)
    left = max(0.0, min(100.0, p_t1 * 100))
    right = 100 - left
    return f"""
    <div class="prob-wrap">
        <div class="prob-left" style="width:{left:.2f}%; background: linear-gradient(90deg, {meta1['color']}, {meta1['accent']}); color:#f8fafc;">
            {left:.1f}%
        </div>
        <div class="prob-right" style="width:{right:.2f}%; background: linear-gradient(90deg, {meta2['accent']}, {meta2['color']});">
            {right:.1f}%
        </div>
    </div>
    <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:.85rem; margin-top:.45rem;">
        <span>{team1_name}</span><span>{team2_name}</span>
    </div>
    """


def confidence_label(p_t1: float) -> tuple[str, str]:
    edge = abs(p_t1 - 0.5)
    if edge >= 0.22:
        return "Strong lean", "Model sees a major team/context gap."
    if edge >= 0.12:
        return "Moderate lean", "There is a clear favorite, but not a lock."
    if edge >= 0.06:
        return "Thin edge", "The matchup is close; small assumptions can flip it."
    return "Coin flip", "The model sees this as almost evenly matched."


def mini_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{label}</div>
            <div class="mini-value">{value}</div>
            <div class="mini-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_pvalue(p):
    if pd.isna(p):
        return "n/a"
    if p < 0.01:
        return "p < 0.01"
    if p < 0.05:
        return "p < 0.05"
    if p < 0.10:
        return "p < 0.10"
    return f"p = {p:.2f}"


def render_edge_card(row, team2_name: str):
    beta = row['beta']
    desc = row['description']
    won_m, lost_m, p = row['won_mean'], row['lost_mean'], row['p_value']
    direction = "raise" if beta > 0 else "lower"
    badge_bg = "rgba(34,197,94,.17)" if beta > 0 else "rgba(239,68,68,.17)"
    badge_color = "#86efac" if beta > 0 else "#fca5a5"
    badge_text = "Push higher" if beta > 0 else "Keep lower"
    st.markdown(
        f"""
        <div class="edge-card">
            <div class="edge-top">
                <div class="edge-feature">{desc}</div>
                <div class="edge-badge" style="background:{badge_bg}; color:{badge_color};">{badge_text}</div>
            </div>
            <div class="edge-meta">
                To exploit {team2_name}, try to <b>{direction}</b> this matchup differential. &nbsp;
                β = <b>{beta:+.3f}</b> · wins Δ = <b>{won_m:+.2f}</b> · losses Δ = <b>{lost_m:+.2f}</b> · {format_pvalue(p)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_dataframe(df_display: pd.DataFrame):
    return (
        df_display.style
        .format(precision=3)
        .background_gradient(subset=['β (standardised)'], cmap='RdYlGn')
        .bar(subset=['|β|'], align='left')
    )


# ============================================================================
# UI — terminal one-page visual redesign
# ============================================================================

# Override the earlier visual helpers with a darker terminal-style aesthetic.
def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: #060912;
            --bg2: #0b1020;
            --panel: rgba(12, 18, 32, 0.82);
            --panel-solid: #0e1524;
            --panel-soft: rgba(18, 27, 46, 0.72);
            --grid: rgba(95, 220, 255, 0.12);
            --border: rgba(125, 211, 252, 0.20);
            --border-strong: rgba(0, 229, 255, 0.42);
            --text: #E6EDF3;
            --muted: #8B9BB0;
            --cyan: #00E5FF;
            --cyan-soft: rgba(0, 229, 255, .12);
            --magenta: #FF2E9A;
            --lime: #39FF14;
            --red: #FF3B5C;
            --amber: #FFB020;
            --violet: #A78BFA;
        }

        html, body, .stApp, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 12% 3%, rgba(0,229,255,.18), transparent 28rem),
                radial-gradient(circle at 86% 2%, rgba(255,46,154,.14), transparent 28rem),
                radial-gradient(circle at 50% 100%, rgba(57,255,20,.08), transparent 40rem),
                linear-gradient(135deg, #05070d 0%, #09101f 45%, #070b14 100%) !important;
            color: var(--text) !important;
            font-family: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace !important;
        }

        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            opacity: .42;
            background-image:
                linear-gradient(rgba(0,229,255,.055) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,229,255,.045) 1px, transparent 1px);
            background-size: 44px 44px;
            mask-image: radial-gradient(circle at center, black 0%, transparent 78%);
        }

        [data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
        .block-container { max-width: 1400px; padding-top: 1.15rem; padding-bottom: 3rem; }

        #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent !important; }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top left, rgba(0,229,255,.14), transparent 16rem),
                linear-gradient(180deg, rgba(10,14,20,.98), rgba(6,9,18,.98)) !important;
            border-right: 1px solid var(--border) !important;
            box-shadow: 18px 0 45px rgba(0,0,0,.24);
        }
        [data-testid="stSidebar"] * { color: var(--text) !important; }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            font-family: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
        }
        h1, h2, h3 { letter-spacing: -0.035em; }
        h2 { margin-top: .4rem; }

        .terminal-brand {
            border: 1px solid var(--border-strong);
            border-left: 3px solid var(--cyan);
            background:
                linear-gradient(135deg, rgba(14,21,36,.90), rgba(9,15,28,.78)),
                radial-gradient(circle at 10% 0%, rgba(0,229,255,.24), transparent 24rem),
                radial-gradient(circle at 90% 15%, rgba(255,46,154,.18), transparent 26rem);
            box-shadow: 0 0 0 1px rgba(255,255,255,.03) inset, 0 24px 75px rgba(0,0,0,.35), 0 0 40px rgba(0,229,255,.08);
            border-radius: 8px;
            padding: 1.4rem 1.55rem;
            margin-bottom: 1rem;
            overflow: hidden;
            position: relative;
        }
        .terminal-brand::after {
            content: "";
            position: absolute;
            left: 0; right: 0; bottom: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--cyan), var(--magenta), transparent);
            opacity: .8;
        }
        .eyebrow {
            color: var(--cyan);
            letter-spacing: .34em;
            text-transform: uppercase;
            font-size: .72rem;
            font-weight: 900;
            text-shadow: 0 0 18px rgba(0,229,255,.35);
        }
        .terminal-title {
            font-size: clamp(2rem, 4.4vw, 4.6rem);
            line-height: .95;
            margin: .45rem 0 .35rem 0;
            font-weight: 950;
            color: var(--text);
        }
        .terminal-subtitle {
            color: var(--muted);
            max-width: 950px;
            font-size: .95rem;
            line-height: 1.55;
        }
        .status-row {
            display: flex;
            gap: .6rem;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .status-pill {
            border: 1px solid rgba(0,229,255,.34);
            color: var(--cyan);
            background: rgba(0,229,255,.07);
            border-radius: 999px;
            padding: .28rem .62rem;
            font-size: .68rem;
            letter-spacing: .16em;
            text-transform: uppercase;
            font-weight: 800;
        }
        .status-pill.alt {
            border-color: rgba(255,46,154,.34);
            color: #ff8cc7;
            background: rgba(255,46,154,.08);
        }

        .matchup-strip {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 74px minmax(0, 1fr);
            gap: .85rem;
            align-items: stretch;
            margin-top: 1.15rem;
        }
        .team-panel {
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            padding: 1.05rem 1.1rem;
            border: 1px solid rgba(255,255,255,.15);
            min-height: 138px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.04), 0 16px 44px rgba(0,0,0,.24);
        }
        .team-panel::after {
            content: "";
            position:absolute; inset:0;
            background: repeating-linear-gradient(135deg, rgba(255,255,255,.045) 0 1px, transparent 1px 12px);
            opacity: .36;
            pointer-events:none;
        }
        .team-abbr {
            position: relative; z-index: 1;
            font-size: clamp(2.2rem, 5vw, 4.25rem);
            font-weight: 950;
            letter-spacing: -.08em;
            line-height: .92;
            text-shadow: 0 18px 45px rgba(0,0,0,.35);
        }
        .team-name {
            position: relative; z-index: 1;
            font-size: .96rem;
            font-weight: 850;
            margin-top: .35rem;
            color: rgba(255,255,255,.92);
        }
        .team-loc {
            position: relative; z-index: 1;
            display: inline-flex;
            gap: .4rem;
            align-items: center;
            margin-top: .7rem;
            padding: .27rem .6rem;
            border-radius: 4px;
            background: rgba(0,0,0,.28);
            border: 1px solid rgba(255,255,255,.16);
            font-size: .74rem;
            font-weight: 850;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .versus {
            align-self: center;
            justify-self: center;
            width: 66px;
            height: 66px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            font-size: 1.08rem;
            font-weight: 950;
            color: #06111b;
            background: linear-gradient(135deg, var(--cyan), var(--lime));
            border: 1px solid rgba(255,255,255,.35);
            box-shadow: 0 0 36px rgba(0,229,255,.32), 0 18px 50px rgba(0,0,0,.30);
        }

        .section-card {
            background:
                linear-gradient(180deg, rgba(14,21,36,.86), rgba(9,14,25,.76));
            border: 1px solid var(--border);
            border-left: 2px solid var(--cyan);
            border-radius: 8px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,.24), inset 0 0 0 1px rgba(255,255,255,.025);
            margin: 1rem 0;
        }
        .section-label {
            display: flex;
            align-items: center;
            gap: .6rem;
            margin-bottom: .8rem;
            color: var(--muted);
            letter-spacing: .22em;
            text-transform: uppercase;
            font-size: .70rem;
            font-weight: 900;
        }
        .section-label::before {
            content:"";
            width: 7px; height: 7px;
            border-radius: 999px;
            background: var(--cyan);
            box-shadow: 0 0 16px var(--cyan);
        }
        .section-heading {
            color: var(--text);
            font-size: 1.38rem;
            font-weight: 950;
            margin: 0 0 .75rem 0;
            letter-spacing: -.03em;
        }
        .section-copy {
            color: var(--muted);
            font-size: .88rem;
            line-height: 1.55;
            margin-top: -.35rem;
            margin-bottom: .85rem;
        }

        .metric-tile {
            background:
                radial-gradient(circle at 0% 0%, rgba(0,229,255,.10), transparent 15rem),
                linear-gradient(180deg, rgba(16,24,40,.86), rgba(8,13,24,.78));
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: .9rem 1rem;
            min-height: 112px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.025), 0 14px 40px rgba(0,0,0,.18);
        }
        .metric-label { color: var(--muted); font-size: .68rem; font-weight: 900; text-transform: uppercase; letter-spacing: .18em; }
        .metric-value { color: var(--text); font-size: 1.65rem; font-weight: 950; margin-top: .35rem; font-variant-numeric: tabular-nums; }
        .metric-help { color: var(--muted); font-size: .78rem; margin-top: .25rem; line-height: 1.4; }

        .prob-wrap {
            border-radius: 6px;
            height: 42px;
            overflow: hidden;
            display: flex;
            background: rgba(6, 10, 18, .92);
            border: 1px solid var(--border);
            box-shadow: inset 0 2px 10px rgba(0,0,0,.38), 0 0 28px rgba(0,229,255,.08);
        }
        .prob-left, .prob-right {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 950;
            font-size: .93rem;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }
        .prob-caption {
            display:flex; justify-content:space-between; color:var(--muted); font-size:.78rem; margin-top:.45rem;
        }
        .terminal-note {
            border-radius: 8px;
            border: 1px solid rgba(255,176,32,.32);
            background: linear-gradient(135deg, rgba(255,176,32,.10), rgba(255,176,32,.035));
            padding: .85rem .95rem;
            color: #ffd98a;
            font-size: .82rem;
            line-height: 1.45;
            margin-top: .8rem;
        }

        .edge-card {
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: .9rem 1rem;
            margin: .62rem 0;
            background:
                radial-gradient(circle at 0% 0%, rgba(0,229,255,.09), transparent 12rem),
                linear-gradient(135deg, rgba(12,18,32,.92), rgba(17,25,42,.76));
            box-shadow: 0 12px 34px rgba(0,0,0,.20);
        }
        .edge-top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
        .edge-feature { font-weight: 950; font-size: .98rem; color: var(--text); }
        .edge-badge { border-radius: 4px; padding: .24rem .55rem; font-size: .70rem; font-weight: 950; letter-spacing: .08em; text-transform: uppercase; }
        .edge-meta { color: var(--muted); margin-top: .45rem; font-size: .80rem; line-height: 1.5; }

        div[data-testid="stMetric"] {
            background: rgba(12,18,32,.78);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: .75rem .85rem;
            box-shadow: 0 12px 34px rgba(0,0,0,.18);
        }
        div[data-testid="stMetric"] label { color: var(--muted) !important; letter-spacing: .08em; text-transform: uppercase; }
        div[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 950; }
        div[data-testid="stMetricDelta"] { color: var(--cyan) !important; }

        [data-baseweb="select"] > div, [data-baseweb="input"] > div, textarea {
            background: rgba(6,10,18,.88) !important;
            border-color: var(--border) !important;
            border-radius: 6px !important;
            color: var(--text) !important;
        }
        [data-baseweb="select"] span, input, textarea { color: var(--text) !important; }
        [data-testid="stSlider"] [role="slider"] { background-color: var(--cyan) !important; box-shadow: 0 0 0 4px rgba(0,229,255,.16) !important; }
        .stButton > button {
            background: linear-gradient(135deg, rgba(0,229,255,.12), rgba(255,46,154,.08)) !important;
            border: 1px solid var(--border-strong) !important;
            color: var(--cyan) !important;
            border-radius: 6px !important;
            font-family: 'JetBrains Mono', monospace !important;
            letter-spacing: .14em;
            text-transform: uppercase;
            font-size: .76rem;
            font-weight: 900;
        }
        .stButton > button:hover { border-color: var(--cyan) !important; box-shadow: 0 0 22px rgba(0,229,255,.20); }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: 0 14px 42px rgba(0,0,0,.16);
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            background: rgba(12,18,32,.58) !important;
        }
        .stAlert { border-radius: 8px; }

        /* ──────────────────────────────────────────────────────────────────
           st.dataframe (Glide Data Grid) — match dark terminal theme.
           Glide exposes CSS custom properties prefixed --gdg-*; overriding
           them on the wrapper themes the canvas-rendered grid.
           ────────────────────────────────────────────────────────────────── */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrameResizable"] {
            background: rgba(12, 18, 32, 0.62) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            overflow: hidden;
            padding: 0 !important;
        }
        [data-testid="stDataFrame"] > div,
        [data-testid="stDataFrameResizable"] > div {
            background: transparent !important;
        }
        /* The Glide grid container — set the CSS vars Glide reads */
        [data-testid="stDataFrame"] [class*="dvn-scroller"],
        [data-testid="stDataFrame"] .glideDataEditor,
        [data-testid="stDataFrame"] [data-testid="glide-cell-0-0"],
        [data-testid="stDataFrame"] {
            --gdg-bg-cell:                 rgba(12, 18, 32, 0.65) !important;
            --gdg-bg-cell-medium:          rgba(18, 27, 46, 0.78) !important;
            --gdg-bg-header:               rgba(2, 6, 23, 0.92) !important;
            --gdg-bg-header-has-focus:     rgba(0, 229, 255, 0.18) !important;
            --gdg-bg-header-hovered:       rgba(0, 229, 255, 0.12) !important;
            --gdg-text-dark:               #E6EDF3 !important;
            --gdg-text-medium:             #C5D1DE !important;
            --gdg-text-light:              #8B9BB0 !important;
            --gdg-text-bubble:             #E6EDF3 !important;
            --gdg-bg-icon-header:          #8B9BB0 !important;
            --gdg-fg-icon-header:          #E6EDF3 !important;
            --gdg-text-header:             #E6EDF3 !important;
            --gdg-text-header-selected:    #00E5FF !important;
            --gdg-text-group-header:       #E6EDF3 !important;
            --gdg-bg-bubble:               rgba(0, 229, 255, 0.18) !important;
            --gdg-bg-bubble-selected:      rgba(0, 229, 255, 0.32) !important;
            --gdg-border-color:            rgba(125, 211, 252, 0.18) !important;
            --gdg-drilldown-border:        rgba(125, 211, 252, 0.30) !important;
            --gdg-link-color:              #00E5FF !important;
            --gdg-cell-horizontal-padding: 12px !important;
            --gdg-cell-vertical-padding:   8px !important;
            --gdg-header-bottom-border-color: rgba(0, 229, 255, 0.32) !important;
            --gdg-horizontal-border-color: rgba(125, 211, 252, 0.10) !important;
            --gdg-accent-color:            #00E5FF !important;
            --gdg-accent-light:            rgba(0, 229, 255, 0.18) !important;
            --gdg-text-input-bg:           rgba(2, 6, 23, 0.95) !important;
            --gdg-bg-search-result:        rgba(255, 176, 32, 0.20) !important;
        }
        [data-testid="stDataFrame"] canvas {
            background: transparent !important;
        }
        /* The thin scroll/info bar Streamlit puts under the grid */
        [data-testid="stDataFrame"] [data-testid="stDataFrameToolbar"],
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"] {
            background: rgba(2, 6, 23, 0.85) !important;
            color: #8B9BB0 !important;
        }
        [data-testid="stDataFrame"] [data-testid="stDataFrameToolbar"] button,
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button {
            color: #C5D1DE !important;
        }

        /* ──────────────────────────────────────────────────────────────────
           Plotly charts — make any st.plotly_chart container blend in.
           ────────────────────────────────────────────────────────────────── */
        [data-testid="stPlotlyChart"] {
            background: transparent !important;
        }
        [data-testid="stPlotlyChart"] > div {
            background: transparent !important;
        }
        /* Modebar (we hide it via config, but just in case) */
        .modebar {
            background: rgba(2, 6, 23, 0.85) !important;
        }

        @media (max-width: 900px) {
            .matchup-strip { grid-template-columns: 1fr; }
            .versus { width: 54px; height: 54px; }
            .terminal-brand { padding: 1.1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def team_panel_html(team_name: str, is_home: bool, align: str = "left") -> str:
    meta = get_team_meta(team_name)
    loc = "Home court" if is_home else "Away"
    icon = "HOME" if is_home else "AWAY"
    text_align = "right" if align == "right" else "left"
    gradient = (
        f"linear-gradient(135deg, {meta['color']} 0%, rgba(8,13,24,.94) 72%), "
        f"radial-gradient(circle at 10% 0%, {meta['accent']}55, transparent 18rem)"
    )
    # NOTE: kept as a single concatenated string with no leading whitespace so Streamlit's
    # Markdown renderer doesn't treat the indented HTML as a code block (which is what was
    # causing the literal "</div>" to appear in the hero).
    return (
        f'<div class="team-panel" style="background:{gradient}; text-align:{text_align};">'
        f'<div class="team-abbr">{meta["abbr"]}</div>'
        f'<div class="team-name">{team_name}</div>'
        f'<div class="team-loc">{icon} · {loc}</div>'
        f'</div>'
    )


def hero_html(team1_name: str, team2_name: str, t1_is_home: bool) -> str:
    # Single-line concatenated string to avoid Markdown indent-as-code-block.
    return (
        '<div class="terminal-brand">'
            '<div class="eyebrow">◉ NBA MODEL TERMINAL · MATCHUP LAB</div>'
            '<div class="terminal-title">Win Probability Console</div>'
            '<div class="terminal-subtitle">'
                'One-page command center for XGBoost matchup probability, head-to-head logistic '
                'regression edges, recent history, and synthetic model inputs.'
            '</div>'
            '<div class="status-row">'
                '<span class="status-pill">XGBoost · cached inference</span>'
                '<span class="status-pill alt">LR coefficients · actionable features</span>'
                '<span class="status-pill">Scenario controls live</span>'
            '</div>'
            '<div class="matchup-strip">'
                f'{team_panel_html(team1_name, t1_is_home, "left")}'
                '<div class="versus">VS</div>'
                f'{team_panel_html(team2_name, not t1_is_home, "right")}'
            '</div>'
        '</div>'
    )


def prob_bar_html(team1_name: str, team2_name: str, p_t1: float) -> str:
    meta1 = get_team_meta(team1_name)
    meta2 = get_team_meta(team2_name)
    left = max(1.0, min(99.0, p_t1 * 100))
    right = 100 - left
    return f"""
    <div class="prob-wrap">
        <div class="prob-left" style="width:{left:.2f}%; background: linear-gradient(90deg, {meta1['color']}, {meta1['accent']}); color:#f8fafc;">
            {p_t1*100:.1f}%
        </div>
        <div class="prob-right" style="width:{right:.2f}%; background: linear-gradient(90deg, {meta2['accent']}, {meta2['color']}); color:#f8fafc;">
            {(1-p_t1)*100:.1f}%
        </div>
    </div>
    <div class="prob-caption"><span>{team1_name}</span><span>{team2_name}</span></div>
    """


def mini_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(label: str, title: str, copy: str = ""):
    st.markdown(
        f"""
        <div class="section-label">{label}</div>
        <div class="section-heading">{title}</div>
        {f'<div class="section-copy">{copy}</div>' if copy else ''}
        """,
        unsafe_allow_html=True,
    )


def render_edge_card(row, team2_name: str):
    beta = row['beta']
    desc = row['description']
    won_m, lost_m, p = row['won_mean'], row['lost_mean'], row['p_value']
    direction = "raise" if beta > 0 else "lower"
    badge_bg = "rgba(57,255,20,.13)" if beta > 0 else "rgba(255,59,92,.13)"
    badge_color = "#88ff72" if beta > 0 else "#ff91a4"
    badge_text = "PUSH HIGHER" if beta > 0 else "KEEP LOWER"
    st.markdown(
        f"""
        <div class="edge-card">
            <div class="edge-top">
                <div class="edge-feature">{desc}</div>
                <div class="edge-badge" style="background:{badge_bg}; color:{badge_color}; border:1px solid {badge_color}55;">{badge_text}</div>
            </div>
            <div class="edge-meta">
                vs {team2_name}: <b>{direction}</b> this differential ·
                β <b>{beta:+.3f}</b> · wins Δ <b>{won_m:+.2f}</b> · losses Δ <b>{lost_m:+.2f}</b> · {format_pvalue(p)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_beta_chart_df(result: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    chart_df = result.head(n)[['description', 'beta']].copy()
    chart_df = chart_df.iloc[::-1]
    return chart_df.set_index('description')


st.set_page_config(
    page_title="NBA Model Terminal",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="expanded",
)
inject_css()

sorted_names = sorted(NAME_TO_ID.keys())
if "team1_select" not in st.session_state:
    st.session_state.team1_select = "Boston Celtics"
if "team2_select" not in st.session_state:
    st.session_state.team2_select = "Los Angeles Lakers"
if "t1_home_toggle" not in st.session_state:
    st.session_state.t1_home_toggle = True


def swap_teams():
    st.session_state.team1_select, st.session_state.team2_select = (
        st.session_state.team2_select,
        st.session_state.team1_select,
    )
    st.session_state.t1_home_toggle = not st.session_state.t1_home_toggle


with st.sidebar:
    st.markdown("<div class='eyebrow'>CONTROL PANEL</div>", unsafe_allow_html=True)
    st.markdown("### Matchup")
    st.button("⇄ Swap teams", use_container_width=True, on_click=swap_teams)
    team1_name = st.selectbox("Team 1", sorted_names, key="team1_select")
    team2_name = st.selectbox("Team 2", sorted_names, key="team2_select")
    t1_is_home = st.toggle("Team 1 has home court", key="t1_home_toggle")

    st.markdown("---")
    st.markdown("### Model controls")
    recent_n = st.slider("Recent games used", 5, 60, 30, step=5)
    diff_days_rest = st.slider("Rest edge: Team 1 minus Team 2", -3, 3, 0)

    with st.expander("Fatigue flags", expanded=False):
        c_fat1, c_fat2 = st.columns(2)
        with c_fat1:
            st.caption("Team 1")
            team_is_b2b = st.checkbox("B2B", key="team_b2b")
            team_is_3in4 = st.checkbox("3-in-4", key="team_3in4")
            team_is_4in5 = st.checkbox("4-in-5", key="team_4in5")
        with c_fat2:
            st.caption("Team 2")
            opp_is_b2b = st.checkbox("B2B", key="opp_b2b")
            opp_is_3in4 = st.checkbox("3-in-4", key="opp_3in4")
            opp_is_4in5 = st.checkbox("4-in-5", key="opp_4in5")

    st.markdown("---")
    st.caption("One-page layout: prediction → coach edge → history → model inputs.")


t1_id = NAME_TO_ID[team1_name]
t2_id = NAME_TO_ID[team2_name]

if t1_id == t2_id:
    st.warning("Pick two different teams.")
    st.stop()

with st.spinner("Loading matchup data and training XGBoost. First run is cached afterward..."):
    df = load_data()
    xgb_model = train_xgboost(df)

with st.sidebar:
    if 'DayDate' in df.columns:
        latest_date = pd.to_datetime(df['DayDate']).max().strftime('%b %d, %Y')
        earliest_date = pd.to_datetime(df['DayDate']).min().strftime('%b %d, %Y')
        st.markdown("### Data status")
        st.caption(f"{len(df):,} matchup rows")
        st.caption(f"{earliest_date} → {latest_date}")

st.markdown(hero_html(team1_name, team2_name, t1_is_home), unsafe_allow_html=True)

X_pred = build_synthetic_row(df, t1_id, t2_id, t1_is_home=t1_is_home, recent_n=recent_n)
if X_pred is not None:
    X_pred = X_pred.copy()
    X_pred.loc[:, 'Diff_Days_Rest'] = diff_days_rest
    X_pred.loc[:, 'Team_Is_B2B'] = int(team_is_b2b)
    X_pred.loc[:, 'Opp_Is_B2B'] = int(opp_is_b2b)
    X_pred.loc[:, 'Team_Is_3in4'] = int(team_is_3in4)
    X_pred.loc[:, 'Opp_Is_3in4'] = int(opp_is_3in4)
    X_pred.loc[:, 'Team_Is_4in5'] = int(team_is_4in5)
    X_pred.loc[:, 'Opp_Is_4in5'] = int(opp_is_4in5)

result, info, msg = analyze_matchup(df, t1_id, t2_id, ACTIONABLE_FEATURES)

# -----------------------------------------------------------------------------
# Section 1 — Prediction
# -----------------------------------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
render_section_header(
    "prediction",
    "XGBoost win probability",
    "Synthetic current-strength row generated from recent team performance, location, rest, fatigue, altitude and head-to-head context.",
)

if X_pred is None:
    st.error("Not enough recent data for one of these teams.")
else:
    p_t1 = predict_xgb(xgb_model, X_pred)
    p_t2 = 1 - p_t1
    favorite = team1_name if p_t1 >= p_t2 else team2_name
    label, help_text = confidence_label(p_t1)

    left, right = st.columns([1.35, 1.0], gap="large")
    with left:
        st.markdown(prob_bar_html(team1_name, team2_name, p_t1), unsafe_allow_html=True)
        st.progress(p_t1, text=f"{team1_name} win probability")
        st.markdown(
            "<div class='terminal-note'>Scenario estimate only — this is a modeling dashboard, not a betting guarantee. "
            "Use the controls to stress-test assumptions instead of treating the output as fixed truth.</div>",
            unsafe_allow_html=True,
        )
    with right:
        c1, c2 = st.columns(2)
        with c1:
            mini_card("Favorite", favorite, label)
        with c2:
            mini_card("Confidence", label, help_text)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(team1_name, f"{p_t1 * 100:.1f}%", "HOME" if t1_is_home else "AWAY")
    with m2:
        st.metric(team2_name, f"{p_t2 * 100:.1f}%", "AWAY" if t1_is_home else "HOME")
    with m3:
        st.metric("Recent window", f"{recent_n} games")
    with m4:
        st.metric("Rest diff", f"{diff_days_rest:+d} days")
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 2 — Coach's edge
# -----------------------------------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
render_section_header(
    "coach edge",
    f"Actionable head-to-head patterns: {team1_name} vs {team2_name}",
    "The logistic regression is trained only on historical games between the selected teams and ranks coachable features by standardized β.",
)

if result is None:
    st.warning(msg)
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        mini_card("Historical sample", f"{info['n_games']} games", "T1 vs T2 only")
    with c2:
        mini_card(f"{get_team_meta(team1_name)['abbr']} wins", str(info['n_t1_wins']), "in the sample")
    with c3:
        win_rate = info['n_t1_wins'] / max(info['n_games'], 1)
        mini_card("H2H win rate", f"{win_rate * 100:.1f}%", "Team 1 win share")

    if info['n_games'] < 25:
        st.warning(
            f"Sample size is only {info['n_games']} games for {len(ACTIONABLE_FEATURES)} coefficients. "
            "Treat individual β values as suggestive, not definitive."
        )

    ctrl1, ctrl2 = st.columns([1.2, 1.2])
    with ctrl1:
        top_n = st.slider("Signals displayed", 3, 12, 6)
    with ctrl2:
        p_filter = st.select_slider(
            "Significance filter",
            options=["All", "p < 0.10", "p < 0.05", "p < 0.01"],
            value="All",
        )

    filtered = result.copy()
    if p_filter != "All":
        cutoff = float(p_filter.split("<")[-1].strip())
        filtered = filtered[filtered['p_value'] < cutoff]
    if len(filtered) == 0:
        st.caption("No features passed the selected p-value filter. Showing top features by |β| instead.")
        filtered = result.copy()

    edge_col, chart_col = st.columns([1.12, 1.0], gap="large")
    with edge_col:
        for _, row in filtered.head(top_n).iterrows():
            render_edge_card(row, team2_name)
    with chart_col:
        st.caption("Top standardized β coefficients")
        chart_df = build_beta_chart_df(filtered, n=min(10, len(filtered)))
        # Horizontal bar chart with theme-matching colors
        # Positive β  (helps Team 1) -> lime  ;  Negative β (helps Team 2) -> red
        bar_colors = ['#39FF14' if v > 0 else '#FF3B5C' for v in chart_df['beta']]
        fig = go.Figure(go.Bar(
            x=chart_df['beta'],
            y=chart_df.index,
            orientation='h',
            marker=dict(
                color=bar_colors,
                line=dict(color='rgba(255,255,255,0.18)', width=0.8),
            ),
            hovertemplate='<b>%{y}</b><br>β = %{x:+.3f}<extra></extra>',
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                color='#E6EDF3',
                family='JetBrains Mono, SF Mono, Menlo, Consolas, monospace',
                size=12,
            ),
            xaxis=dict(
                gridcolor='rgba(125,211,252,0.12)',
                zerolinecolor='rgba(0,229,255,0.45)',
                zerolinewidth=1.5,
                showline=False,
                tickfont=dict(size=11, color='#8B9BB0'),
            ),
            yaxis=dict(
                gridcolor='rgba(0,0,0,0)',
                tickfont=dict(size=11, color='#E6EDF3'),
                automargin=True,
            ),
            margin=dict(l=8, r=12, t=8, b=8),
            height=410,
            showlegend=False,
            hoverlabel=dict(
                bgcolor='rgba(2, 6, 23, 0.92)',
                bordercolor='rgba(0,229,255,0.45)',
                font=dict(color='#E6EDF3', family='JetBrains Mono, monospace', size=12),
            ),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'displayModeBar': False},
        )

    with st.expander("Full coefficient table", expanded=False):
        display = result[['description', 'beta', 'abs_beta', 'won_mean', 'lost_mean', 't_stat', 'p_value']].copy()
        display.columns = [
            'Feature', 'β (standardised)', '|β|',
            f'Δ when {team1_name} won', f'Δ when {team1_name} lost',
            't-stat', 'p-value'
        ]
        numeric_cols = display.select_dtypes(include=[np.number]).columns
        display[numeric_cols] = display[numeric_cols].round(3)
        st.dataframe(style_dataframe(display), use_container_width=True, hide_index=True)

    st.caption(
        "Reading guide: Δ is Team 1 minus Team 2 in historical games. β is standardized, "
        "so use it to rank features; use the Δ columns for basketball-unit magnitudes."
    )
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 3 — Recent history
# -----------------------------------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
render_section_header(
    "history",
    f"Recent {team1_name} vs {team2_name} games",
    "Quick sanity check for what the model has seen in this particular head-to-head matchup.",
)

hist_n = st.slider("Games to show", 5, 25, 10, key="hist_n")
hist = (df[(df['TeamID'] == t1_id) & (df['OppID'] == t2_id)]
        .sort_values('DayDate', ascending=False).head(hist_n)).copy()
if len(hist) > 0:
    show_cols = ['DayDate', 'Season', 'Is_Home', 'Target_Win',
                 'Diff_Pregame_Elo', 'Diff_Avg_Score', 'Diff_Avg_Opp_Score', 'Elo_Win_Prob']
    show_cols = [c for c in show_cols if c in hist.columns]
    hist_disp = hist[show_cols].copy()
    hist_disp['DayDate'] = hist_disp['DayDate'].dt.strftime('%Y-%m-%d')
    hist_disp['Winner'] = hist_disp['Target_Win'].map({1: team1_name, 0: team2_name})
    hist_disp['Result'] = hist_disp['Target_Win'].map({1: 'W', 0: 'L'})
    hist_disp['Location'] = hist_disp['Is_Home'].map({1: 'Home', 0: 'Away'})
    rename_map = {
        'DayDate': 'Date',
        'Season': 'Season',
        'Diff_Pregame_Elo': 'Pregame Elo diff',
        'Diff_Avg_Score': 'Avg score diff',
        'Diff_Avg_Opp_Score': 'Avg allowed diff',
        'Elo_Win_Prob': 'Elo win prob',
    }
    hist_disp = hist_disp.drop(columns=[c for c in ['Target_Win', 'Is_Home'] if c in hist_disp.columns])
    hist_disp = hist_disp.rename(columns=rename_map)
    for c in ['Pregame Elo diff', 'Avg score diff', 'Avg allowed diff', 'Elo win prob']:
        if c in hist_disp.columns:
            hist_disp[c] = hist_disp[c].round(2)
    ordered_cols = [c for c in ['Date', 'Season', 'Location', 'Result', 'Winner', 'Pregame Elo diff', 'Avg score diff', 'Avg allowed diff', 'Elo win prob'] if c in hist_disp.columns]
    st.dataframe(hist_disp[ordered_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
else:
    st.info(f"No historical {team1_name} vs {team2_name} games in the data.")
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 4 — Model inputs
# -----------------------------------------------------------------------------
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
render_section_header(
    "model inputs",
    "Synthetic XGBoost feature vector",
    "Search the generated 50-feature row to debug exactly what the model consumed for this scenario.",
)

if X_pred is None:
    st.error("No synthetic row available.")
else:
    search = st.text_input("Filter features", placeholder="type Elo, rest, score, rebound, altitude...")
    df_show = X_pred.T.rename(columns={X_pred.index[0]: 'Value'}).copy()
    df_show['Value'] = df_show['Value'].round(4)
    df_show = df_show.reset_index().rename(columns={'index': 'Feature'})
    if search:
        df_show = df_show[df_show['Feature'].str.contains(search, case=False, na=False)]
    st.dataframe(df_show, use_container_width=True, hide_index=True, height=520)
st.markdown("</div>", unsafe_allow_html=True)