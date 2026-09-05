# Quantifying Psychological Momentum in Professional Badminton
Project Source Code — Working with AI Project, Sports Analytics, IIM Ranchi

## Contents
Eleven Python scripts, run in numeric order. Each reads/writes intermediate
pickle files in its working directory, so run them from the same folder,
in sequence, after placing the raw dataset CSV
(`bwf-ss-gamedata-2015-2017-new.csv`, from Kaggle: canggih/badminton-game-data-
bwf-super-series-20152017) in that same folder.

| Script | Purpose |
|---|---|
| `step1_parse.py` | Parse & validate point-by-point sequences from the raw CSV |
| `step2_runs.py` | Run-length distribution & pooled conditional win probabilities |
| `step3_logit.py` | Cluster-robust logistic regression models (M1-M4) |
| `step4_perm.py` | 300-fold permutation test vs. an i.i.d. null |
| `step5_subgroups.py` | Sub-group robustness checks (game closeness, round, season) |
| `step6_charts.py` | Generate report figures (run-length, conditional probability, permutation) |
| `step7_models.py` | Next-point predictive models: Logistic Regression, Random Forest, XGBoost |
| `step8_winprob_model.py` | In-play win-probability model at the 11-point checkpoint |
| `step9_tableau_prep.py` | Aggregate tables prepared for the Tableau extract |
| `step10_build_hyper.py` | Build the Tableau `.hyper` data extract |
| `step11_build_twb.py` | Generate the Tableau `.twb` workbook XML and package as `.twbx` |

## Dependencies
```
pandas, numpy, scipy, statsmodels, scikit-learn, xgboost, matplotlib, tableauhyperapi
```
Install with:
```
pip install pandas numpy scipy statsmodels scikit-learn xgboost matplotlib tableauhyperapi
```

## Note on paths (fixed)
All scripts use paths relative to the current working directory. Place the
raw dataset CSV in the same folder as the scripts, then run each script
from that folder (e.g. `cd code && python3 step1_parse.py`). Each script
reads/writes its intermediate `.pkl`/`.png`/etc. files in that same folder,
so run them in order from one consistent working directory. (An earlier
version of this package had hardcoded absolute paths tied to the original
build environment and would not run elsewhere — this has been corrected
and verified by running the full pipeline end-to-end in a clean, isolated
directory.)

## Output
Running the full sequence reproduces: parsed/validated game data, all
statistical test results, all report figures, three trained predictive
models per task (with confusion matrices and ROC curves), and a Tableau
`.hyper`/`.twbx` data package.

Full write-up of results, methodology, and findings is in the main project
report and the accompanying PDF deliverables (Executive Summary, EDA
Analysis, Assumptions, Modelling Techniques) — not included in this
code-only package; see the full project repository for those plus the
raw dataset and generated outputs.
