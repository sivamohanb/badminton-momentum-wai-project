import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

pdf = pd.read_pickle('point_level.pkl')
games = pd.read_pickle('games_parsed.pkl')[['game_id', 'margin']]
pdf = pdf.merge(games, on='game_id', how='left')

pdf['close_game'] = (pdf['margin'] <= 3).astype(int)

print("=== Momentum coefficient (lag1) by game closeness ===")
for label, sub in [('Close games (final margin <=3)', pdf[pdf['close_game'] == 1]),
                    ('Lopsided games (final margin >3)', pdf[pdf['close_game'] == 0])]:
    m = smf.logit('y ~ lag1 + margin_before + points_played', data=sub).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': sub['game_id']})
    print(f"{label}: n={len(sub)}  lag1 coef={m.params['lag1']:.4f}  p={m.pvalues['lag1']:.2g}  OR={np.exp(m.params['lag1']):.4f}")

print("\n=== Momentum coefficient (lag1) by round stage ===")
for rnd, sub in pdf.groupby('Round'):
    if len(sub) < 2000:
        continue
    m = smf.logit('y ~ lag1 + margin_before + points_played', data=sub).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': sub['game_id']})
    print(f"Round {rnd}: n={len(sub)}  lag1 coef={m.params['lag1']:.4f}  p={m.pvalues['lag1']:.2g}  OR={np.exp(m.params['lag1']):.4f}")

print("\n=== Momentum coefficient (lag1) by year ===")
for yr, sub in pdf.groupby('Year'):
    m = smf.logit('y ~ lag1 + margin_before + points_played', data=sub).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': sub['game_id']})
    print(f"Year {yr}: n={len(sub)}  lag1 coef={m.params['lag1']:.4f}  p={m.pvalues['lag1']:.2g}  OR={np.exp(m.params['lag1']):.4f}")

# also recompute per-discipline net effect from Model 4 interaction terms for reporting table
import statsmodels.api as sm
m4 = sm.load('m4.pickle')
base = m4.params['lag1']
disc_effects = {'MD': base}
for t in ['MS', 'WD', 'WS', 'XD']:
    disc_effects[t] = base + m4.params[f'lag1:C(Type)[T.{t}]']
print("\n=== Net lag1 effect by discipline (from Model 4 interaction) ===")
for k, v in disc_effects.items():
    print(f"{k}: {v:.4f}  (OR={np.exp(v):.4f})")
