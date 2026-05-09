# NBA Matchup Predictor Dashboard

A Streamlit dashboard that combines an XGBoost win-probability model with a per-matchup logistic regression analysis, built on top of `matchups_v2.csv` from the `build_features_nba_final.ipynb` pipeline.

## What it does

**1. XGBoost win probability (top section).**
The trained model uses all 50 features from `FEATURE_LIST`. To predict a hypothetical *Team 1 vs Team 2 right now*, the dashboard pulls each team's last 30 historical games and combines them: the synthesised `Diff_X` for each feature is `mean(T1's Diff_X over recent games) − mean(T2's Diff_X over recent games)`, which works out to `T1_X − T2_X` since each team's `Diff_X` averages to roughly `team_X − league_avg_X` over many opponents.

**2. Logistic regression — actionable coefficients (middle section).**
Filters historical data to *only games where Team 1 played Team 2* (typically 18–40 games), fits an L2-regularised LR (C=1.0) on the 22 **actionable features** — the box-score categories a coach can game-plan around — and pairs each β with a Welch's t-test p-value comparing T1-won vs T1-lost games.

The 22 actionable features are: `Diff_Avg_Score`, `Diff_Avg_FGA`, `Diff_Avg_FGM3`, `Diff_Avg_FGA3`, `Diff_Avg_FTA`, `Diff_Avg_OR`, `Diff_Avg_DR`, `Diff_Avg_Ast`, `Diff_Avg_TO`, `Diff_Avg_Stl`, `Diff_Avg_Blk`, plus the 11 `Diff_Avg_Opp_*` defensive equivalents.

**Excluded** (because they're cumulative state or fixed context, not things a team can change game-to-game): Elo, Power Rating, Net Rating, Off/Def Eff, Pythagorean win pct + luck gap, all win-pct variants, win streaks, form gaps, head-to-head, schedule density (B2B / 3in4 / 4in5), Days_Rest, altitude flags, home/away.

**3. Coach's edge.**
For each significant β (p < 0.10), the dashboard shows the direction (push higher or lower than the opponent) and the historic mean Δ in T1-won vs T1-lost games — so you have both a ranking signal (β) and a magnitude in real basketball units (raw avg differences).

**4. Last 10 head-to-head games** at the bottom for context.

## Setup

```bash
pip install streamlit xgboost scikit-learn pandas numpy scipy
```

Place `app.py` and `matchups_v2.csv` in the same directory.

## Run

```bash
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. First load takes ~10 seconds (XGBoost trains on ~24K rows); after that, switching teams is instant — XGBoost is cached via `@st.cache_resource`, the data via `@st.cache_data`.

## Interpreting the LR section

- **β** is *standardised* — each feature is z-scored before fitting, so β is the change in log-odds of T1 winning per 1-standard-deviation change in that `Diff_*` feature. Comparable across features, **not** in raw stat units.
- **β > 0** → T1 wins more when the `Diff_*` is **higher** (Team 1 has more of that stat than Team 2).
- **β < 0** → T1 wins more when the `Diff_*` is **lower** (Team 1 has less of that stat than Team 2). For things like `Diff_Avg_TO`, this is good — fewer turnovers helps.
- **p-value** comes from Welch's t-test on the raw feature values comparing T1-won vs T1-lost games. With ~20-game samples per matchup, take p-values as suggestive — anything below 0.05 is genuinely worth attention; p > 0.20 is essentially noise.

## Sample-size caveats

NBA team pairs play each other 2-4 times per season. Across 10 seasons of data that's typically **18–40 games per matchup**. For 22 actionable features, that's well below the rule-of-thumb 10 samples per coefficient, which is why:

- We use **L2 regularisation** (C=1.0) to stabilise coefficients.
- We complement β with a **t-test** that doesn't depend on a multivariable fit.
- We **rank features by |β|** and surface the top with significance flags rather than reporting confidence intervals on individual coefficients.

The pattern signal is real, but treat individual coefficient values as suggestive, not as published estimates.
