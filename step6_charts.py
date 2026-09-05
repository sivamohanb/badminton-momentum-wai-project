import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})

runs_df = pd.read_pickle('/home/claude/data/runs_df.pkl')
games = pd.read_pickle('/home/claude/data/games_parsed.pkl')

# ---- Chart 1: observed vs i.i.d.-expected run-length distribution ----
obs = runs_df['length'].value_counts(normalize=True).sort_index()
obs = obs[obs.index <= 8]
p_bar = games['winners'].apply(lambda w: np.mean(np.array(w) == 0)).mean()
# average of both symbols' geometric-ish mixture: approx via mean p of the whole corpus
ks = obs.index.values
# theoretical run-length pmf for a 2-state i.i.d. process (mixture over which symbol runs)
p = p_bar
pmf = []
for k in ks:
    prob = p * (1 - p) ** (k - 1) * p + (1 - p) * p ** (k - 1) * (1 - p)
    pmf.append(prob)
pmf = np.array(pmf)
pmf = pmf / pmf.sum() * obs.sum()  # rescale to same truncated mass for fair visual comparison

fig, ax = plt.subplots(figsize=(6.5, 4))
width = 0.38
ax.bar(ks - width/2, obs.values, width, label='Observed', color='#2b6cb0')
ax.bar(ks + width/2, pmf, width, label='i.i.d. benchmark', color='#cbd5e0')
ax.set_xlabel('Run length (consecutive points by same side)')
ax.set_ylabel('Proportion of runs')
ax.set_title('Observed vs. i.i.d.-Benchmark Run-Length Distribution')
ax.set_xticks(ks)
ax.legend()
fig.tight_layout()
fig.savefig('/home/claude/data/chart1_runlength.png', dpi=170)
plt.close(fig)

# ---- Chart 2: P(win|prev win) vs P(win|prev loss) by discipline ----
lag_df = pd.read_pickle('/home/claude/data/lag_df.pkl')
disc_stats = []
for t, g in lag_df.groupby('Type'):
    p1 = g.loc[g['prev_side0_won_prev']==1,'side0_wins_curr'].mean()
    p0 = g.loc[g['prev_side0_won_prev']==0,'side0_wins_curr'].mean()
    disc_stats.append((t, p1, p0))
disc_df = pd.DataFrame(disc_stats, columns=['Type', 'P_win_after_win', 'P_win_after_loss']).set_index('Type')
disc_df = disc_df.loc[['MS', 'WS', 'MD', 'WD', 'XD']]

fig, ax = plt.subplots(figsize=(6.5, 4))
x = np.arange(len(disc_df))
width = 0.35
ax.bar(x - width/2, disc_df['P_win_after_win'], width, label='P(win | won prev pt)', color='#2b6cb0')
ax.bar(x + width/2, disc_df['P_win_after_loss'], width, label='P(win | lost prev pt)', color='#e07a5f')
ax.set_xticks(x)
ax.set_xticklabels(disc_df.index)
ax.set_ylabel('Probability of winning current point')
ax.set_title('Conditional Point-Win Probability by Discipline')
ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--')
ax.legend()
fig.tight_layout()
fig.savefig('/home/claude/data/chart2_discipline.png', dpi=170)
plt.close(fig)

# ---- Chart 3: permutation null distribution vs observed ----
null_stats = np.load('/home/claude/data/null_stats.npy')
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(null_stats, bins=30, color='#cbd5e0', label='Permutation null distribution')
ax.axvline(-0.06782, color='#c0392b', linewidth=2, label='Observed statistic')
ax.set_xlabel('P(win | prev win) − P(win | prev loss)')
ax.set_ylabel('Frequency (out of 300 permutations)')
ax.set_title('Permutation Test: Observed vs. i.i.d.-Shuffled Null')
ax.legend()
fig.tight_layout()
fig.savefig('/home/claude/data/chart3_permutation.png', dpi=170)
plt.close(fig)

print("Charts saved.")
print(disc_df)
