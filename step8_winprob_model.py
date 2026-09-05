import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
                              accuracy_score, precision_score, recall_score, f1_score)
from xgboost import XGBClassifier

plt.rcParams.update({'font.size': 11})

games = pd.read_pickle('games_parsed.pkl')

rows = []
for _, r in games.iterrows():
    w = np.array(r['winners'])
    a = np.cumsum(w == 0)
    b = np.cumsum(w == 1)
    mx = np.maximum(a, b)
    idx = np.argmax(mx >= 11)  # first index where either side reaches 11
    if mx[idx] < 11:
        continue  # game ended before 11 (shouldn't happen given min 21-pt games, but guard anyway)
    a11, b11 = a[idx], b[idx]
    leader = 0 if a11 > b11 else (1 if b11 > a11 else -1)
    if leader == -1:
        continue  # tied at the checkpoint (rare edge case) - drop for a clean binary target
    leader_wins_game = 1 if leader == r['winner'] else 0
    rows.append({
        'game_id': r['game_id'], 'Type': r['Type'], 'Round': r['Round'],
        'margin_at_11': abs(int(a11) - int(b11)),
        'points_played_at_11': int(idx) + 1,
        'leader_wins_game': leader_wins_game,
    })

chk = pd.DataFrame(rows)
print(f"Checkpoint dataset: {len(chk)} games (dropped {len(games) - len(chk)} ties/edge cases)")
print(f"Base rate — leader at 11 wins the game: {chk['leader_wins_game'].mean():.4f}")

type_dummies = pd.get_dummies(chk['Type'], prefix='Type', drop_first=True)
round_dummies = pd.get_dummies(chk['Round'], prefix='Round', drop_first=True)
X = pd.concat([chk[['margin_at_11', 'points_played_at_11']], type_dummies, round_dummies], axis=1).astype(float)
y = chk['leader_wins_game'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train games: {len(X_train)}  Test games: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, eval_metric='logloss', random_state=42, n_jobs=-1),
}

results = {}
fig_roc, ax_roc = plt.subplots(figsize=(6.5, 5.5))

for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_s, y_train)
        proba_train = model.predict_proba(X_train_s)[:, 1]
        proba_test = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        proba_train = model.predict_proba(X_train)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]

    fpr_tr, tpr_tr, thr_tr = roc_curve(y_train, proba_train)
    fpr_te, tpr_te, _ = roc_curve(y_test, proba_test)
    auc_tr, auc_te = auc(fpr_tr, tpr_tr), auc(fpr_te, tpr_te)

    # Optimal decision threshold via Youden's J on the TRAIN fold (avoids degenerate
    # all-one-class confusion matrices caused by the 81% base rate at the default 0.5 cut)
    j_scores = tpr_tr - fpr_tr
    best_thr = thr_tr[np.argmax(j_scores)]
    pred_train = (proba_train >= best_thr).astype(int)
    pred_test = (proba_test >= best_thr).astype(int)

    acc_tr, acc_te = accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test)
    prec_te = precision_score(y_test, pred_test)
    rec_te = recall_score(y_test, pred_test)
    f1_te = f1_score(y_test, pred_test)

    results[name] = dict(acc_train=acc_tr, acc_test=acc_te, precision_test=prec_te,
                          recall_test=rec_te, f1_test=f1_te, auc_train=auc_tr, auc_test=auc_te,
                          threshold=float(best_thr))
    print(f"\n{name}: thr={best_thr:.3f}  Train Acc={acc_tr:.4f} AUC={auc_tr:.4f} | Test Acc={acc_te:.4f} AUC={auc_te:.4f} "
          f"Prec={prec_te:.4f} Rec={rec_te:.4f} F1={f1_te:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, yt, pt, split in [(axes[0], y_train, pred_train, 'Train'), (axes[1], y_test, pred_test, 'Test')]:
        cm = confusion_matrix(yt, pt)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Leader loses', 'Leader wins'])
        disp.plot(ax=ax, colorbar=False, cmap='Greens', values_format='d')
        ax.set_title(f'{split} (n={len(yt)})')
    fig.suptitle(f'{name} — Win-Probability Model Confusion Matrix')
    fig.tight_layout()
    safe = name.lower().replace(' ', '_')
    fig.savefig(f'wp_cm_{safe}.png', dpi=170)
    plt.close(fig)

    ax_roc.plot(fpr_te, tpr_te, label=f'{name} (test AUC={auc_te:.3f})')

ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Chance')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curves — Test Set (In-Play Win-Probability Model)')
ax_roc.legend(loc='lower right')
fig_roc.tight_layout()
fig_roc.savefig('wp_roc_all_models.png', dpi=170)
plt.close(fig_roc)

import json
with open('wp_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
with open('wp_base_rate.txt', 'w') as f:
    f.write(str(chk['leader_wins_game'].mean()))

print("\nSaved win-probability model artifacts.")
print(pd.DataFrame(results).T.round(4))
