import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
                              accuracy_score, precision_score, recall_score, f1_score)
from xgboost import XGBClassifier

plt.rcParams.update({'font.size': 11})

pdf = pd.read_pickle('/home/claude/data/point_level.pkl').dropna(subset=['lag2']).reset_index(drop=True)

feature_cols = ['lag1', 'lag2', 'margin_before', 'near_game_point', 'post_interval', 'points_played']
type_dummies = pd.get_dummies(pdf['Type'], prefix='Type', drop_first=True)
X = pd.concat([pdf[feature_cols], type_dummies], axis=1).astype(float)
y = pdf['y'].astype(int)
groups = pdf['game_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

print(f"Train points: {len(X_train)} ({groups.iloc[train_idx].nunique()} games)")
print(f"Test points:  {len(X_test)} ({groups.iloc[test_idx].nunique()} games)")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
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

    # Youden's-J optimal threshold (fit on train) avoids a degenerate all-one-class
    # confusion matrix given the weak signal / 56% base rate at the default 0.5 cut
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

    # confusion matrices (train & test)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, yt, pt, split in [(axes[0], y_train, pred_train, 'Train'), (axes[1], y_test, pred_test, 'Test')]:
        cm = confusion_matrix(yt, pt)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Side1 wins', 'Side0 wins'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
        ax.set_title(f'{split} (n={len(yt)})')
    fig.suptitle(f'{name} — Confusion Matrix')
    fig.tight_layout()
    safe = name.lower().replace(' ', '_')
    fig.savefig(f'/home/claude/data/cm_{safe}.png', dpi=170)
    plt.close(fig)

    ax_roc.plot(fpr_te, tpr_te, label=f'{name} (test AUC={auc_te:.3f})')

ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Chance')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curves — Test Set (Point-Winner Prediction)')
ax_roc.legend(loc='lower right')
fig_roc.tight_layout()
fig_roc.savefig('/home/claude/data/roc_all_models.png', dpi=170)
plt.close(fig_roc)

# Feature importance (Random Forest) as an extra screenshot
rf = models['Random Forest']
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 4.5))
imp.plot(kind='barh', ax=ax, color='#2b6cb0')
ax.set_title('Random Forest — Feature Importance')
ax.set_xlabel('Importance')
fig.tight_layout()
fig.savefig('/home/claude/data/rf_feature_importance.png', dpi=170)
plt.close(fig)

import json
with open('/home/claude/data/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved all model artifacts.")
print(pd.DataFrame(results).T.round(4))
