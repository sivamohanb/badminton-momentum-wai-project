import pandas as pd
import numpy as np
import time

games = pd.read_pickle('games_parsed.pkl')
seqs = [np.array(w) for w in games['winners']]

def pooled_stat(seq_list):
    """Pooled P(win|prev win) - P(win|prev loss) for side 0, across all games."""
    n1_win = n1_tot = n0_win = n0_tot = 0
    for w in seq_list:
        prev = w[:-1]
        curr = w[1:]
        m1 = prev == 0
        n1_tot += m1.sum()
        n1_win += (curr[m1] == 0).sum()
        m0 = ~m1
        n0_tot += m0.sum()
        n0_win += (curr[m0] == 0).sum()
    return (n1_win / n1_tot) - (n0_win / n0_tot)

t0 = time.time()
observed = pooled_stat(seqs)
print(f"Observed pooled diff P(win|prevwin)-P(win|prevloss) = {observed:.5f}  (took {time.time()-t0:.2f}s)")

rng = np.random.default_rng(42)
N_PERM = 300
null_stats = np.empty(N_PERM)
t0 = time.time()
for i in range(N_PERM):
    perm_seqs = [rng.permutation(w) for w in seqs]
    null_stats[i] = pooled_stat(perm_seqs)
    if (i + 1) % 50 == 0:
        print(f"  permutation {i+1}/{N_PERM}  elapsed {time.time()-t0:.1f}s")

print(f"\nNull distribution (i.i.d. within-game shuffle): mean={null_stats.mean():.5f}, sd={null_stats.std():.5f}")
print(f"Observed statistic: {observed:.5f}")
p_value = (np.sum(np.abs(null_stats) >= np.abs(observed)) + 1) / (N_PERM + 1)
print(f"Permutation p-value (two-sided): {p_value:.5f}")
np.save('null_stats.npy', null_stats)
with open('perm_result.txt', 'w') as f:
    f.write(f"observed={observed}\nnull_mean={null_stats.mean()}\nnull_sd={null_stats.std()}\np_value={p_value}\n")
