# Frozen experiment plan, 2026-09-20

Question: can a Random Forest prioritize failed SECOM production entities for additional review?
This is retrospective quality classification, not demonstrated pre-failure prediction or deployment.
Sensor acquisition timing is undocumented. No root-cause, causal, or financial savings claims.

Source: Aurelien Geron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow,
2nd edition: printed p.199 (PDF p.225) 500-tree, max_leaf_nodes=16 Random Forest;
p.67 (PDF p.93) training-only median imputation; pp.96-97 (PDF pp.122-123) thresholds.
Adapt this recipe to SECOM, not the author's example dataset. Add seed 42 and n_jobs=1
for repeatability. No grid search, no feature selection, no class rebalancing.

Sort timestamps stably. Earliest ~60% training, next ~20% validation, latest ~20% test.
Keep equal timestamps on the same side of boundaries. Fit imputer and forest only on training.
Choose the validation threshold maximizing failure recall subject to <=10% false-positive rate.
Tie-break: fewer false positives, then higher threshold. Include an above-maximum threshold
so review-none is feasible. The 10% budget is an illustrative assumption, not a factory requirement.
Do not refit after threshold choice. Apply once to held-out test, with no test-driven revisions.

Baselines: always pass (zero alerts) and the same forest at threshold 0.5.
Report confusion counts, recall, precision, false-positive rate, alert fraction, balanced accuracy,
accuracy and average precision. Quantify recall/FPR uncertainty using Wilson intervals.
No accuracy-only headline. A failure to improve is a publishable finding, not a reason to tune.

Core <=50 readable physical lines. Preparation, evaluator, plots and documentation separate.
Run the same frozen experiment twice to check deterministic predictions. Independent review
must assess leakage, arithmetic, baselines, calibration/test separation, source attribution,
small-sample uncertainty and claim scope. No automatic LinkedIn posting.
