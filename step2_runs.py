import pandas as pd
import numpy as np
from scipy import stats

games = pd.read_pickle('games_parsed.pkl')

def run_lengths(seq):
    seq = np.array(seq)
    runs = []
    cur = seq[0]
    length = 1
    for x in seq[1:]:
        if x == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = x
            length = 1
    runs.append((cur, length))
    return runs

all_runs = []
for _, r in games.iterrows():
    for side, length in run_lengths(r['winners']):
        all_runs.append({'game_id': r['game_id'], 'Type': r['Type'], 'side': side, 'length': length})
runs_df = pd.DataFrame(all_runs)
runs_df.to_pickle('runs_df.pkl')

print("=== Observed run-length distribution (all games, both sides pooled) ===")
obs_dist = runs_df['length'].value_counts(normalize=True).sort_index()
print(obs_dist.head(12))
print(f"Mean observed run length: {runs_df['length'].mean():.4f}")
print(f"Total runs: {len(runs_df)}  across {games.shape[0]} games")

# Expected geometric distribution under i.i.d. null, using each game's own empirical
# point-win probability (p = points won by side / total points), aggregated by simulation-free
# theoretical mixture: for i.i.d. Bernoulli(p), P(run length = k for the side with prob p... )
# We approximate expected mean run length under i.i.d. per game and compare to observed.
def expected_mean_run_length(p):
    # For a 2-state i.i.d. sequence, expected run length (averaged over both symbols,
    # weighted by their frequency) = 1/p*p + 1/(1-p)*(1-p) is NOT quite right;
    # correct: E[run length | in a run of symbol A] = 1/(1-p) is wrong too.
    # Use direct formula: expected run length overall = 1 / (p^2 + (1-p)^2) is the
    # well-known result for a two-state i.i.d. sequence (average run length across all runs).
    return 1.0 / (p**2 + (1 - p)**2)

games['p_a'] = games['winners'].apply(lambda w: np.mean(np.array(w) == 0))
games['exp_mean_run'] = games['p_a'].apply(expected_mean_run_length)

obs_mean_by_game = games['game_id'].map(runs_df.groupby('game_id')['length'].mean())
games['obs_mean_run'] = obs_mean_by_game

print("\n=== Mean run length: observed vs i.i.d.-expected (per game, then averaged) ===")
print(f"Observed mean run length (avg across games): {games['obs_mean_run'].mean():.4f}")
print(f"i.i.d.-expected mean run length (avg across games): {games['exp_mean_run'].mean():.4f}")
diff = games['obs_mean_run'] - games['exp_mean_run']
t, p = stats.ttest_1samp(diff, 0)
print(f"Paired t-test (obs - expected), mean diff = {diff.mean():.4f}, t = {t:.3f}, p = {p:.4g}")
print(f"Wilcoxon signed-rank test:", stats.wilcoxon(diff))

# ---- Conditional probability of winning point t given outcome of point t-1 ----
recs = []
for _, r in games.iterrows():
    w = np.array(r['winners'])
    if len(w) < 3:
        continue
    prev = w[:-1]
    curr = w[1:]
    for pv, cv in zip(prev, curr):
        recs.append((r['game_id'], r['Type'], pv, cv))
lag_df = pd.DataFrame(recs, columns=['game_id', 'Type', 'prev', 'curr'])
lag_df['prev_side0_won_prev'] = (lag_df['prev'] == 0).astype(int)
lag_df['side0_wins_curr'] = (lag_df['curr'] == 0).astype(int)
lag_df.to_pickle('lag_df.pkl')

p_win_after_win = lag_df.loc[lag_df['prev_side0_won_prev'] == 1, 'side0_wins_curr'].mean()
p_win_after_loss = lag_df.loc[lag_df['prev_side0_won_prev'] == 0, 'side0_wins_curr'].mean()
n1 = (lag_df['prev_side0_won_prev'] == 1).sum()
n0 = (lag_df['prev_side0_won_prev'] == 0).sum()
print(f"\n=== Pooled conditional win probability (side A / side 0) ===")
print(f"P(A wins pt t | A won pt t-1)  = {p_win_after_win:.4f}  (n={n1})")
print(f"P(A wins pt t | A lost pt t-1) = {p_win_after_loss:.4f}  (n={n0})")
count = np.array([lag_df.loc[lag_df['prev_side0_won_prev']==1,'side0_wins_curr'].sum(),
                   lag_df.loc[lag_df['prev_side0_won_prev']==0,'side0_wins_curr'].sum()])
nobs = np.array([n1, n0])
from statsmodels.stats.proportion import proportions_ztest
zstat, pval = proportions_ztest(count, nobs)
print(f"Two-proportion z-test: z = {zstat:.3f}, p = {pval:.4g}")
print(f"Difference (win-after-win minus win-after-loss): {p_win_after_win - p_win_after_loss:+.4f}")

# By discipline
print("\n=== By discipline ===")
for t, g in lag_df.groupby('Type'):
    p1 = g.loc[g['prev_side0_won_prev']==1,'side0_wins_curr'].mean()
    p0 = g.loc[g['prev_side0_won_prev']==0,'side0_wins_curr'].mean()
    print(f"{t}: P(win|prev win)={p1:.4f}  P(win|prev loss)={p0:.4f}  diff={p1-p0:+.4f}  n={len(g)}")
