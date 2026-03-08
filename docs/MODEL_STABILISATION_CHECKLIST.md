# Model Stabilisation Checklist

Use this checklist before marking a rebuild cycle complete.

- [ ] `config/domain_rules.yaml` updated manually for gameweek trust status.
- [ ] Polluted weeks rebuilt with `scripts/backfill_recommended_squad_pit.py`.
- [ ] Backfill validation metadata reviewed (`backfill_trusted=true`, status `validated_trusted`).
- [ ] Rebuilt trusted weeks moved from `excluded_gameweeks` / `backfilled_but_untrusted_gameweeks` into `trusted_backfilled_gameweeks`.
- [ ] `scripts/train_model.py` run and `logs/model_weekly_report.json` regenerated.
- [ ] `scripts/run_tuning_dual.py` run and feasible candidate chosen under balanced gates.
- [ ] `scripts/run_autonomous_loop.py --force-drift` run and evidence captured.
- [ ] Candidate evidence includes:
  - [ ] pre/post/delta calibration metrics
  - [ ] weekly backtest summary and per-week details
  - [ ] per-position summaries and position bias checks
  - [ ] ranking metrics (`top_k_hit_rate`, `rank_correlation`, `selected_xi_regret`)
  - [ ] collapse and instability metrics (`prediction_collapse_weeks`, `bias_flip_weeks`)
- [ ] Baseline comparison recorded (candidate vs active model) and archived.
