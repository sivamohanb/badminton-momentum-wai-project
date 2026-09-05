import pandas as pd
import numpy as np
import json

games = pd.read_pickle('/home/claude/data/games_parsed.pkl')
runs_df = pd.read_pickle('/home/claude/data/runs_df.pkl')
lag_df = pd.read_pickle('/home/claude/data/lag_df.pkl')

# 1. Games by discipline
games_by_disc = games['Type'].value_counts().rename_axis('Type').reset_index(name='Games')

# 2. Games by year
games_by_year = games['Year'].value_counts().rename_axis('Year').reset_index(name='Games').sort_values('Year')

# 3. Conditional probability by discipline (long format: Type, Condition, Probability)
rows = []
for t, g in lag_df.groupby('Type'):
    p1 = g.loc[g['prev_side0_won_prev'] == 1, 'side0_wins_curr'].mean()
    p0 = g.loc[g['prev_side0_won_prev'] == 0, 'side0_wins_curr'].mean()
    rows.append({'Type': t, 'Condition': 'Won Previous Point', 'Probability': float(p1)})
    rows.append({'Type': t, 'Condition': 'Lost Previous Point', 'Probability': float(p0)})
# overall (Type = 'ALL')
p1_all = lag_df.loc[lag_df['prev_side0_won_prev'] == 1, 'side0_wins_curr'].mean()
p0_all = lag_df.loc[lag_df['prev_side0_won_prev'] == 0, 'side0_wins_curr'].mean()
rows.append({'Type': 'ALL', 'Condition': 'Won Previous Point', 'Probability': float(p1_all)})
rows.append({'Type': 'ALL', 'Condition': 'Lost Previous Point', 'Probability': float(p0_all)})
cond_prob = pd.DataFrame(rows)

# 4. Run length distribution: observed vs iid benchmark (<=8)
obs = runs_df['length'].value_counts(normalize=True).sort_index()
obs = obs[obs.index <= 8]
p_bar = games['winners'].apply(lambda w: np.mean(np.array(w) == 0)).mean()
ks = obs.index.values
pmf = []
for k in ks:
    prob = p_bar * (1 - p_bar) ** (k - 1) * p_bar + (1 - p_bar) * p_bar ** (k - 1) * (1 - p_bar)
    pmf.append(prob)
pmf = np.array(pmf)
pmf = pmf / pmf.sum() * obs.sum()
run_len = pd.DataFrame({'Length': ks.astype(int), 'Observed': obs.values.astype(float), 'IIDBenchmark': pmf.astype(float)})

# 5. Odds ratio by discipline (from Model 4, hardcoded from earlier fitted results)
odds_ratio = pd.DataFrame({
    'Type': ['MD', 'XD', 'WD', 'WS', 'MS'],
    'OddsRatio': [0.5886, 0.6715, 0.7493, 0.8674, 0.9231]
})

# 6. Model AUC comparison (next-point + win-probability tasks)
model_auc = pd.DataFrame([
    {'Task': 'Next-Point Prediction', 'Model': 'Logistic Regression', 'TestAUC': 0.539},
    {'Task': 'Next-Point Prediction', 'Model': 'Random Forest', 'TestAUC': 0.550},
    {'Task': 'Next-Point Prediction', 'Model': 'XGBoost', 'TestAUC': 0.551},
    {'Task': 'Win-Probability (11-pt checkpoint)', 'Model': 'Logistic Regression', 'TestAUC': 0.716},
    {'Task': 'Win-Probability (11-pt checkpoint)', 'Model': 'Random Forest', 'TestAUC': 0.713},
    {'Task': 'Win-Probability (11-pt checkpoint)', 'Model': 'XGBoost', 'TestAUC': 0.710},
])

# 7. Margin distribution
margin_dist = games['margin'].value_counts().rename_axis('Margin').reset_index(name='Games').sort_values('Margin')

# 8. Game-level flat table (for filters/drill-down) - keep it reasonably small (no point-level detail)
game_level = games[['game_id', 'Year', 'Tournament', 'Round', 'Type', 'final_a', 'final_b', 'winner', 'margin', 'n_points']].copy()
game_level['Year'] = game_level['Year'].astype(int)
game_level['final_a'] = game_level['final_a'].astype(int)
game_level['final_b'] = game_level['final_b'].astype(int)
game_level['winner'] = game_level['winner'].astype(int)
game_level['margin'] = game_level['margin'].astype(int)
game_level['n_points'] = game_level['n_points'].astype(int)
game_level['game_id'] = game_level['game_id'].astype(int)

out = {
    'games_by_disc': games_by_disc,
    'games_by_year': games_by_year,
    'cond_prob': cond_prob,
    'run_len': run_len,
    'odds_ratio': odds_ratio,
    'model_auc': model_auc,
    'margin_dist': margin_dist,
    'game_level': game_level,
}
for k, v in out.items():
    v.to_pickle(f'/home/claude/data/tableau_{k}.pkl')
    print(k, v.shape)
    print(v.head(3))
    print()
