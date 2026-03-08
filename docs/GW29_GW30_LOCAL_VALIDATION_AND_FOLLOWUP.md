# GW29/GW30 Local Validation And Follow-up Deep Dive

Date: 2026-03-08

## Scope

This note extends the earlier reviews in:

- `/Users/am/Sync/fpl-workspace/GW29_MODEL_DEEP_DIVE.md`
- `/Users/am/Sync/fpl-workspace/5_3_findings.md`

It uses local data only and answers two questions:

1. What squad does the repaired local pipeline produce for GW29 and GW30?
2. After the stabilisation pass, what is still wrong with the model and where should the next improvement work focus?

## Method

All analysis used the local snapshot only:

- data source: `pipeline/data/training/latest.parquet`
- policy: `TRAINING_DATA_SOURCE=local`, `TRAINING_DATA_POLICY=LOCAL_ONLY`

Validation runs:

- GW29 retrospective point-in-time rebuild:
  - train on rows with `target_gameweek_id < 29`
  - infer from rows with `gameweek_id <= 28`
  - optimise a legal 15-player squad
- GW29 oracle comparison:
  - optimise directly on actual GW29 points from local data
- GW30 forecast:
  - use the repaired local artefact in `pipeline/logs/model.bin`
  - infer from local data up to GW29
  - set `ALLOW_INVALID_FORWARD_PUBLISH=1` for test execution only

Every reported squad was validated locally for:

- 15 total players
- exact FPL position mix: `2/5/5/3`
- max `3` players per team
- budget within configured limit
- optimiser objective total matching XI total plus captain bonus

## Executive Summary

The first pass fixed the most dangerous failure mode from the earlier deep dives:

- predictions are no longer collapsed
- calibration is no longer selected when it harms holdout quality
- replay/live semantics are now aligned enough for local comparison
- forward publication is correctly blocked when the artefact fails ranking gates

However, the model is still not good enough to trust.

The remaining failure is now clearer:

- scale is broadly plausible
- ranking is still weak
- the model still misses too many ceiling outcomes
- simple baselines are still stronger than the model on recent top-player capture

That means the next phase should not be another calibration pass. It should focus on ranking signal, target design, and feature quality.

## GW29 Predicted Squad

Point-in-time local rebuild for GW29:

| Pos | Role | Player | EP | Actual |
|---|---|---|---:|---:|
| GK | VC | Pickford | 4.08 | 6.00 |
| GK | Bench | Martinez | 2.98 | 1.00 |
| DEF | XI | J.Timber | 4.05 | 13.00 |
| DEF | XI | Saliba | 3.97 | 0.00 |
| DEF | XI | Tchatchoua | 3.94 | 5.00 |
| DEF | Bench | Kayode | 3.89 | 6.00 |
| DEF | Bench | Hall | 3.23 | 2.00 |
| MID | C | Mbeumo | 6.00 | 1.00 |
| MID | XI | Soucek | 5.50 | 3.00 |
| MID | XI | J.Gomes | 4.87 | 3.00 |
| MID | XI | Semenyo | 4.75 | 7.00 |
| MID | Bench | M.Salah | 3.20 | 9.00 |
| FWD | XI | Welbeck | 5.27 | 1.00 |
| FWD | XI | Beto | 4.85 | 2.00 |
| FWD | XI | Ekitike | 4.76 | 2.00 |

GW29 totals:

- predicted 15-man squad EP: `65.33`
- predicted optimiser total: `58.03`
- actual 15-man squad total: `61.00`
- actual optimiser total: `44.00`

## GW29 Oracle Squad

Oracle squad built from actual GW29 points:

| Pos | Role | Player | Actual |
|---|---|---|---:|
| GK | XI | Ellborg | 10.00 |
| GK | Bench | Cox | 0.00 |
| DEF | VC | Tarkowski | 15.00 |
| DEF | XI | J.Timber | 13.00 |
| DEF | XI | Gabriel | 11.00 |
| DEF | XI | Senesi | 11.00 |
| DEF | Bench | Hill | 8.00 |
| MID | XI | Sarr | 15.00 |
| MID | XI | Anderson | 12.00 |
| MID | XI | Andre | 12.00 |
| MID | XI | Summerville | 11.00 |
| MID | XI | Garner | 10.00 |
| FWD | C | Joao Pedro | 19.00 |
| FWD | Bench | Haaland | 2.00 |
| FWD | Bench | Isak | 0.00 |

GW29 oracle totals:

- oracle 15-man squad total: `149.00`
- oracle optimiser total: `158.00`
- GW29 optimiser regret versus oracle: `114.00`

Interpretation:

- the repaired model built a legal squad and avoided the previous collapse
- it still failed badly at identifying the true GW29 spike players
- this remains a ranking problem more than a scaling problem

## GW30 Predicted Squad

Forward-style local GW30 forecast from the repaired artefact:

| Pos | Role | Player | EP |
|---|---|---|---:|
| GK | XI | Pickford | 4.09 |
| GK | Bench | Gillespie | 2.60 |
| DEF | XI | Saliba | 5.49 |
| DEF | XI | Chalobah | 4.44 |
| DEF | XI | O'Reilly | 4.35 |
| DEF | Bench | Boly | 2.60 |
| DEF | Bench | Heaven | 1.26 |
| MID | C | Rice | 7.09 |
| MID | XI | Wilson | 5.21 |
| MID | XI | Saka | 5.16 |
| MID | XI | Rogers | 5.03 |
| MID | Bench | Olusesi | 1.62 |
| FWD | XI | Joao Pedro | 5.97 |
| FWD | XI | Haaland | 5.29 |
| FWD | VC | Watkins | 4.61 |

GW30 totals:

- predicted 15-man squad EP: `64.81`
- predicted optimiser total: `63.81`
- budget used: `99.6`

Important note:

- this GW30 squad was generated for local test purposes only
- the active artefact still fails ranking/regret gates and remains correctly blocked for normal forward publication

## What Improved Since The Earlier Deep Dives

The March 8 stabilisation changes solved the earlier structural failures identified in the first two reviews.

### 1. Prediction collapse is fixed

The earlier reviews showed a collapsed forward distribution with near-zero outputs almost everywhere.

Current local weekly report:

- `prediction_collapse_weeks = 0`
- `prediction_to_actual_ratio_min = 0.802`
- `prediction_to_actual_ratio_max = 0.929`

This is a real improvement. The model is no longer obviously unusable on scale alone.

### 2. Harmful calibration is no longer being selected

Earlier finding:

- calibration worsened MAE/RMSE but still won

Current state:

- `selected_variant = none`
- `delta_mae = 0`
- `delta_rmse = 0`

This means the guardrail worked. Calibration is no longer suppressing predictions into the floor.

### 3. Replay and live semantics are now suitable for local comparison

The earlier review showed that PIT replay and live inference used different post-processing semantics. That made historical debugging less trustworthy.

The current local runs now produce comparable behaviour without the earlier path drift dominating the conclusions.

## What Is Still Failing

### 1. Ranking quality is still far below an acceptable level

Current local weekly summary:

- `top_k_hit_rate_mean = 0.145`
- `rank_correlation_mean = 0.096`
- `selected_xi_regret_mean = 93.0`
- `selected_xi_actual_total_mean = 45.8`
- `oracle_xi_actual_total_mean = 138.8`

This is the central remaining issue.

The model now produces non-collapsed numbers, but they are still not ordering players well enough to build a competitive XI.

### 2. The model still loses to simple baselines on recent top-player capture

Local comparison on recent holdout weeks `25-29`:

| Baseline | Top-k hit rate mean | Rank corr mean | Selected XI actual mean | Regret mean |
|---|---:|---:|---:|---:|
| Current repaired model | 0.145 | 0.096 | 45.8 | 93.0 |
| `three_week_players_roll_avg_points` | 0.327 | 0.298 | 43.0 | 58.6 |
| `five_week_players_roll_avg_points` | 0.345 | 0.279 | 46.6 | 55.0 |
| `total_points_z_score` | 0.291 | 0.118 | 49.0 | 52.6 |

This is an important result.

The repaired model is no longer broken in the same way, but it is still not beating straightforward heuristics on the ranking job that matters most.

### 3. Tail outcomes are still badly underpredicted

Current holdout `p95` by position from `model.metrics.txt`:

| Position | Actual p95 | Predicted p95 |
|---|---:|---:|
| GK | 9.00 | 4.85 |
| DEF | 9.00 | 4.18 |
| MID | 10.00 | 4.43 |
| FWD | 9.00 | 5.27 |

The model is still too conservative in the upper tail.

This helps explain why it misses ceiling weeks:

- Joao Pedro GW29 actual `19`, predicted `3.38`
- Sarr GW29 actual `15`, predicted `3.15`
- J.Timber GW29 actual `13`, predicted `2.40`
- Tarkowski GW29 actual `15`, predicted `5.04`
- Senesi GW29 actual `11`, predicted `1.84`

### 4. The model may now be overfitting availability and floor more than upside

Current top feature importances:

1. `minutes_band_61_90` = `0.3407`
2. `minutes_band_0_30` = `0.1434`
3. `minutes_played_z_score` = `0.0360`
4. `minutes_played` = `0.0249`

That is a warning sign.

It suggests the repaired model may now be leaning heavily on playing-time proxies and safe-floor signals, while underweighting the features that separate ordinary starters from week-winning upside players.

This does not mean minute features are wrong. It means they may be too dominant relative to the rest of the signal.

### 5. The optimiser is still not the primary bottleneck

The earlier deep dives were right on this point.

The optimiser can only choose from the ranking it is given. The current evidence still shows:

- bad overlap with top actual performers
- very large XI regret
- persistent ceiling misses

That points upstream to model signal quality, not downstream optimisation.

## Updated Diagnosis

The original findings still hold, but the failure hierarchy is now clearer.

### Confirmed as fixed enough for now

- target contamination from zero-filled shifted labels
- harmful calibration selection
- replay/live post-processing drift
- collapse detection and publication blocking

### Still the main unresolved problem

- player ranking
- ceiling capture
- squad-level regret

### Likely next root causes

1. **Target design is still too weak for ranking upside**
   - predicting raw next-gameweek points may be too noisy as a single-objective regression target
   - the model appears better at central tendency than at ordering spike outcomes

2. **Feature set is still too floor-oriented**
   - recent rolling means and minutes features reward safety
   - they do not appear to separate explosive outcomes strongly enough

3. **Validation strategy is too permissive on baseline competitiveness**
   - the artefact can now be technically stable while still being strategically poor
   - current gates catch collapse, but not failure against naive alternatives

## Where The Next Improvements Should Focus

### Priority 1. Add baseline-beating gates

The model should not be eligible for publication unless it beats at least one strong local heuristic baseline on the recent validation block.

Recommended new gates:

- model `top_k_hit_rate_mean` must beat `five_week_players_roll_avg_points`
- model `selected_xi_regret_mean` must beat `five_week_players_roll_avg_points`
- model `rank_correlation_mean` must beat a simple rolling-average baseline

If it cannot outperform naive baselines, it should not ship.

### Priority 2. Rework the target for ranking, not just point error

Current training is still optimised like a standard regression problem.

That is probably insufficient for squad selection.

Strong candidates for the next pass:

- two-stage modelling:
  - probability of meaningful minutes / start
  - conditional points given minutes
- ranking-aware evaluation during tuning:
  - select models by regret/top-k capture first, MAE second
- alternative target variants:
  - clipped target
  - log-plus-linear hybrid target
  - per-position targets

### Priority 3. Improve ceiling-sensitive features

The next feature pass should focus on information that separates upside players from merely safe starters.

Examples to test locally:

- more recent role-sensitive attacking involvement
- set-piece or penalty responsibility proxies if already derivable locally
- team attacking environment trend versus opponent concession trend
- home/away and fixture-context interactions
- per-90 variants instead of absolute recent totals where minutes already dominate

The key test is whether these features improve top-k capture, not only MAE.

### Priority 4. Reduce minute-band dominance

The current feature importance profile suggests the model may be over-relying on coarse minute buckets.

Questions to test:

- does removing one or both minute-band dummies improve ranking metrics?
- do continuous minute or start-likelihood features outperform bucketed bands?
- does the model simply learn "starts = safe 3-5 points" and fail to model upside?

### Priority 5. Move evaluation closer to the actual decision problem

Current reports already include squad regret. That is good.

The next step is to use those metrics as primary tuning criteria, not as downstream diagnostics only.

Recommended tuning objective order:

1. minimise `selected_xi_regret_mean`
2. maximise `top_k_hit_rate_mean`
3. maximise `rank_correlation_mean`
4. only then consider MAE/RMSE

## Questions The Next Dive Should Explicitly Answer

1. Does the model beat `five_week_players_roll_avg_points` on recent local validation after any proposed change?
2. Which features most improve top-k hit rate rather than plain MAE?
3. Does a two-stage minutes-plus-points model materially reduce XI regret?
4. Are defender and goalkeeper spikes structurally harder because the feature set lacks clean-sheet event signal?
5. Are holdout splits representative enough, or is the current random split still flattering MAE while hiding week-level ranking weakness?
6. Would a time-based validation split make the failure mode even more obvious?
7. Does per-position modelling outperform a single shared regressor on ranking metrics?

## Bottom Line

The initial deep dives were directionally correct:

- the original collapse was caused by a real engineering defect and harmful calibration
- fixing those problems was necessary

After the repair, the diagnosis is sharper:

- the model is now stable enough to measure honestly
- it is still not competitive enough to trust
- the next gains will come from ranking-aware modelling and better upside signal, not from more calibration work

Forward publication should remain disabled until the model can beat naive local baselines on recent local validation.
