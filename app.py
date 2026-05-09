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
# UI
# ============================================================================

st.set_page_config(page_title="NBA Matchup Predictor", layout="wide", page_icon="🏀")
st.title("🏀 NBA Matchup Predictor")

# Sidebar — team selection
with st.sidebar:
    st.header("Matchup")
    sorted_names = sorted(NAME_TO_ID.keys())
    default_t1 = sorted_names.index("Boston Celtics")
    default_t2 = sorted_names.index("Los Angeles Lakers")
    team1_name = st.selectbox("Team 1", sorted_names, index=default_t1)
    team2_name = st.selectbox("Team 2", sorted_names, index=default_t2)
    t1_is_home = st.toggle("Team 1 is home", value=True)
    st.markdown("---")
    st.caption(
        "**XGBoost** predicts using each team's last 30 games to estimate current strength. "
        "**Logistic regression** trains only on historical games where Team 1 played Team 2 "
        "and reports β coefficients on the 22 actionable features (the ones a coach can "
        "actually game-plan around)."
    )

t1_id = NAME_TO_ID[team1_name]
t2_id = NAME_TO_ID[team2_name]

if t1_id == t2_id:
    st.warning("Pick two different teams.")
    st.stop()

with st.spinner("Loading data and training XGBoost (first time only — cached afterward)..."):
    df = load_data()
    xgb_model = train_xgboost(df)


# ----------------------------------------------------------------------------
# Section 1 — XGBoost win probability
# ----------------------------------------------------------------------------
st.header("📊 XGBoost Win Probability")

X_pred = build_synthetic_row(df, t1_id, t2_id, t1_is_home=t1_is_home)
if X_pred is None:
    st.error("Not enough recent data for one of these teams.")
else:
    p_t1 = predict_xgb(xgb_model, X_pred)
    p_t2 = 1 - p_t1

    c1, c2 = st.columns(2)
    c1.metric(f"{team1_name}{' 🏠' if t1_is_home else ' ✈️'}", f"{p_t1*100:.1f}%")
    c2.metric(f"{team2_name}{' ✈️' if t1_is_home else ' 🏠'}", f"{p_t2*100:.1f}%")
    st.progress(p_t1)

    with st.expander("Show synthesised feature vector (the 50 inputs to XGBoost)"):
        df_show = X_pred.T.rename(columns={X_pred.index[0]: 'Value'}).copy()
        df_show['Value'] = df_show['Value'].round(4)
        st.dataframe(df_show, use_container_width=True)


# ----------------------------------------------------------------------------
# Section 2 — Logistic regression on T1 vs T2 historical games
# ----------------------------------------------------------------------------
st.header(f"⚙️ Logistic Regression — {team1_name} vs {team2_name} historical patterns")

result, info, msg = analyze_matchup(df, t1_id, t2_id, ACTIONABLE_FEATURES)

if result is None:
    st.warning(msg)
else:
    st.caption(
        f"Trained on **{info['n_games']}** historical {team1_name} vs {team2_name} matchups "
        f"({info['n_t1_wins']} {team1_name} wins, {info['n_t1_losses']} losses). "
        f"Excluded from this model: Elo, Power Rating, Net Rating, Off/Def Eff, "
        f"Pythagorean, all Win Pct variants, win streaks, form gaps, H2H, schedule "
        f"density, altitude, home/away — none of which a team can change game-to-game."
    )

    if info['n_games'] < 25:
        st.warning(
            f"⚠️ Sample size of {info['n_games']} games is small for fitting "
            f"{len(ACTIONABLE_FEATURES)} coefficients. The L2 regularisation stabilises "
            "things, but treat individual β values as suggestive, not definitive."
        )

    # Full coefficient table
    st.subheader("All actionable β coefficients (sorted by |β|)")
    display = result[['description', 'beta', 'won_mean', 'lost_mean', 't_stat', 'p_value']].copy()
    display.columns = ['Feature', 'β (standardised)', f'Δ when {team1_name} won',
                       f'Δ when {team1_name} lost', 't-stat', 'p-value']
    display['β (standardised)'] = display['β (standardised)'].round(3)
    display[f'Δ when {team1_name} won']  = display[f'Δ when {team1_name} won'].round(2)
    display[f'Δ when {team1_name} lost'] = display[f'Δ when {team1_name} lost'].round(2)
    display['t-stat']  = display['t-stat'].round(2)
    display['p-value'] = display['p-value'].round(3)
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Coach's edge — actionable insights
    st.subheader(f"🎯 Coach's edge — what should help {team1_name} beat {team2_name}")

    significant = result[result['p_value'] < 0.10].head(8).copy()
    if len(significant) == 0:
        significant = result.head(5).copy()
        st.caption("No features are even borderline-significant (p < 0.10). "
                   "Showing top 5 by |β| anyway — but interpret with extreme caution.")

    for _, row in significant.iterrows():
        beta = row['beta']
        desc = row['description']
        won_m, lost_m, p = row['won_mean'], row['lost_mean'], row['p_value']

        direction = "HIGHER" if beta > 0 else "LOWER"
        sign_emoji = "🟢" if beta > 0 else "🔴"

        if not pd.isna(p):
            if p < 0.01:    sig = "**highly significant** (p < 0.01)"
            elif p < 0.05:  sig = "**significant** (p < 0.05)"
            elif p < 0.10:  sig = "borderline (p < 0.10)"
            else:           sig = f"weak (p = {p:.2f})"
        else:
            sig = "p-value unavailable"

        st.markdown(
            f"- {sign_emoji} **{desc}** — push it **{direction}** than {team2_name}.  \n"
            f"  &nbsp;&nbsp;&nbsp;&nbsp;β = `{beta:+.3f}` &nbsp;|&nbsp; "
            f"avg Δ in wins: `{won_m:+.2f}` &nbsp;|&nbsp; "
            f"avg Δ in losses: `{lost_m:+.2f}` &nbsp;|&nbsp; {sig}"
        )

    st.caption(
        "**Reading guide.** Each `Δ` is `Team 1 − Team 2` averaged over historical games of "
        "that outcome. β is the standardised LR coefficient — it's not in raw stat units, so "
        "use β for *ranking* features and use the `Δ in wins` / `Δ in losses` columns for "
        "*magnitudes* in real basketball units."
    )


# ----------------------------------------------------------------------------
# Section 3 — Recent historical games
# ----------------------------------------------------------------------------
st.header(f"📜 Last 10 {team1_name} vs {team2_name} games")
hist = (df[(df['TeamID'] == t1_id) & (df['OppID'] == t2_id)]
        .sort_values('DayDate', ascending=False).head(10)).copy()
if len(hist) > 0:
    show_cols = ['DayDate', 'Season', 'Is_Home', 'Target_Win',
                 'Diff_Pregame_Elo', 'Diff_Avg_Score', 'Diff_Avg_Opp_Score', 'Elo_Win_Prob']
    show_cols = [c for c in show_cols if c in hist.columns]
    hist_disp = hist[show_cols].copy()
    hist_disp['DayDate']  = hist_disp['DayDate'].dt.strftime('%Y-%m-%d')
    hist_disp['Target_Win'] = hist_disp['Target_Win'].map({1: f"✅ {team1_name}", 0: f"❌ {team2_name}"})
    hist_disp['Is_Home']  = hist_disp['Is_Home'].map({1: 'Home', 0: 'Away'})
    for c in ['Diff_Pregame_Elo', 'Diff_Avg_Score', 'Diff_Avg_Opp_Score', 'Elo_Win_Prob']:
        if c in hist_disp.columns:
            hist_disp[c] = hist_disp[c].round(2)
    st.dataframe(hist_disp.reset_index(drop=True), use_container_width=True, hide_index=True)
else:
    st.info(f"No historical {team1_name} vs {team2_name} games in the data.")
