import pandas as pd
import numpy as np

df = pd.read_csv('bwf-ss-gamedata-2015-2017-new.csv')
df['game_id'] = np.arange(len(df))

def parse_scores(s):
    states = []
    for tok in s.split(';'):
        a, b = tok.split('-')
        states.append((int(a), int(b)))
    return states

records = []
bad = 0
for _, row in df.iterrows():
    try:
        states = parse_scores(row['Scores'])
    except Exception:
        bad += 1
        continue
    # dedupe consecutive identical states (e.g. leading 0-0;0-0)
    dedup = [states[0]]
    for s in states[1:]:
        if s != dedup[-1]:
            dedup.append(s)
    # build point-winner sequence from deltas
    winners = []
    valid = True
    for (a0, b0), (a1, b1) in zip(dedup[:-1], dedup[1:]):
        da, db = a1 - a0, b1 - b0
        if da == 1 and db == 0:
            winners.append(0)
        elif db == 1 and da == 0:
            winners.append(1)
        else:
            valid = False
            break
    if not valid or len(winners) < 5:
        bad += 1
        continue
    final_a, final_b = dedup[-1]
    # sanity: badminton game validity (win by 2, cap 30, min 21 unless retired)
    winner = 0 if final_a > final_b else 1
    margin = abs(final_a - final_b)
    records.append({
        'game_id': row['game_id'], 'Year': row['Year'], 'Tournament': row['Tournament'],
        'Round': row['Round'], 'Match': row['Match'], 'Type': row['Type'],
        'final_a': final_a, 'final_b': final_b, 'winner': winner, 'margin': margin,
        'n_points': len(winners), 'winners': winners
    })

print(f"Parsed OK: {len(records)}  |  Dropped/invalid: {bad}  |  Total rows: {len(df)}")

games = pd.DataFrame(records)
# additional validity filter: standard game must end 21+ (win by 2) or capped at 30, or 21-x (x<=19)
def is_valid_final(a, b):
    hi, lo = max(a, b), min(a, b)
    if hi == 30:
        return True
    if hi >= 21 and hi - lo >= 2:
        return True
    return False

games['valid_final'] = games.apply(lambda r: is_valid_final(r['final_a'], r['final_b']), axis=1)
print(games['valid_final'].value_counts())
games_valid = games[games['valid_final']].reset_index(drop=True)
print(f"Final analysis sample: {len(games_valid)} games")

games_valid.to_pickle('games_parsed.pkl')

print(games_valid[['Type']].value_counts())
print(games_valid['n_points'].describe())
print(games_valid['margin'].describe())
