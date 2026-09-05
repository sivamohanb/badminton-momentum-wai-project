import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

games = pd.read_pickle('games_parsed.pkl')

rows = []
for _, r in games.iterrows():
    w = np.array(r['winners'])
    n = len(w)
    a_before = np.concatenate(([0], np.cumsum(w == 0)))[:-1]  # score of side0 before point t
    b_before = np.concatenate(([0], np.cumsum(w == 1)))[:-1]
    for t in range(1, n):  # start at 1 so lag1 exists
        rows.append({
            'game_id': r['game_id'],
            'Type': r['Type'],
            'Round': r['Round'],
            'Year': r['Year'],
            't': t,
            'a_before': a_before[t],
            'b_before': b_before[t],
            'lag1': 1 if w[t-1] == 0 else 0,   # 1 if side0 won previous point
            'lag2': (1 if w[t-2] == 0 else 0) if t >= 2 else np.nan,
            'y': 1 if w[t] == 0 else 0,        # side0 wins current point
        })

pdf = pd.DataFrame(rows)
pdf['margin_before'] = pdf['a_before'] - pdf['b_before']          # signed, +ve favors side0
pdf['abs_margin_before'] = pdf['margin_before'].abs()
pdf['max_score_before'] = pdf[['a_before', 'b_before']].max(axis=1)
pdf['near_game_point'] = (pdf['max_score_before'] >= 20).astype(int)
pdf['post_interval'] = (pdf['max_score_before'] == 11).astype(int)  # point right after the 11-pt break
pdf['points_played'] = pdf['a_before'] + pdf['b_before']
pdf.to_pickle('point_level.pkl')
print(f"Point-level observations: {len(pdf)}")

# ---- Model 1: lag1 only ----
m1 = smf.logit('y ~ lag1', data=pdf).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pdf['game_id']})
print("\n=== Model 1: y ~ lag1 (cluster-robust SE by game) ===")
print(m1.summary().tables[1])

# ---- Model 2: lag1 + lag2 ----
pdf2 = pdf.dropna(subset=['lag2']).copy()
m2 = smf.logit('y ~ lag1 + lag2', data=pdf2).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pdf2['game_id']})
print("\n=== Model 2: y ~ lag1 + lag2 ===")
print(m2.summary().tables[1])

# ---- Model 3: lag1 controlling for game-state (margin, near game point, interval, points played) ----
m3 = smf.logit('y ~ lag1 + margin_before + near_game_point + post_interval + points_played',
               data=pdf).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pdf['game_id']})
print("\n=== Model 3: y ~ lag1 + margin_before + near_game_point + post_interval + points_played ===")
print(m3.summary().tables[1])
print(m3.summary().tables[0])

# ---- Model 4: add discipline (Type) fixed effects + lag1 x Type interaction ----
m4 = smf.logit('y ~ lag1 * C(Type) + margin_before + near_game_point + post_interval + points_played',
               data=pdf).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pdf['game_id']})
print("\n=== Model 4: + discipline fixed effects & lag1 x discipline interaction ===")
print(m4.summary().tables[1])

m1.save('m1.pickle')
m3.save('m3.pickle')
m4.save('m4.pickle')

# Odds ratio & implied probability swing for headline model (Model 3)
import numpy as np
coef = m3.params['lag1']
orat = np.exp(coef)
base_p = pdf['y'].mean()
print(f"\nModel 3 lag1 coefficient = {coef:.4f}, Odds Ratio = {orat:.4f}")
print(f"Baseline P(side0 wins point) = {base_p:.4f}")
