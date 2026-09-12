# Validation

Every number here was produced by the scripts in this repository and can be
regenerated. Where a result is weak or a mechanism does not do what its name
suggests, it is written down as such.

## Metric definitions

The problem statement names a set of figures of merit. Each is implemented in
`backend/app/evaluation/metrics.py` with the definition below, because several of
these names admit more than one reasonable reading.

| metric | definition |
|---|---|
| `probability_of_detection` | HIT / (HIT + MISS): detections among looks that landed on genuine activity |
| `probability_of_false_alarm` | FALSE_ALARM / (FALSE_ALARM + CORRECT_NON_DETECTION) |
| `sensitivity_index` | Pd - Pfa, Youden's J: detector quality in one number |
| `miss_rate` | MISS / (HIT + MISS) |
| `average_intercept_rate` | detections per time slot: the headline efficiency number |
| `intercept_ratio` | fraction of activity bursts intercepted at least once |
| `average_intercept_delay` | mean slots from activity onset to first interception of that burst |
| `intercept_time_error_rms` | root mean square of those delays, against an ideal of zero |
| `detection_rate` | HIT / scans |
| `scan_efficiency` | any detection / scans |
| `coverage` | fraction of bands looked at least once |
| `coverage_entropy` | normalised entropy of scan effort across bands |
| `percentage_correct_predictions` | accuracy of the model probability at 0.5, scored against the realised observation |
| `average_reward` | mean of `+detection - false_alarm_penalty - miss_penalty - scan_cost` |
| `average_adaptation_time` | slots after a change point until the rolling detection rate regains 90% of its pre-change level |

Measured Pd and Pfa are properties of the *receiver*, not the scheduler, so they
should come out near the configured `receiver.pd` and `receiver.pfa` for every
policy. They do, which is a useful sanity check on the whole pipeline.
`tests/test_engine.py::test_measured_receiver_statistics_match_the_configuration`
asserts it.

## Methodology

1. Build a synthetic environment from a fixed seed.
2. Run every policy against that same environment realisation and the same
   receiver conditions.
3. Record every decision and observation.
4. Compute identical metrics for all policies.
5. Repeat across seeds and report mean and spread.

Because the environment is rebuilt from the seed, policies are compared on the
same activity, not merely on the same distribution.

## Why results are reported as paired ratios

Absolute metrics carry large between-seed variance. In the layer ablation the
complete strategy scored 0.2607 +/- 0.0832 intercepts per slot over three seeds:
the spread is a third of the mean, and no two arms are separable from it. That
variance is not policy noise, it is environment noise, because one seed's
environment simply contains more activity than another's.

Every policy runs against the *identical* environment realisation for a given
seed, so the per-seed ratio against a reference policy is a paired measurement
and removes that shared variance entirely. `benchmark.paired_comparison`
computes it, and it is the comparison worth reading. Absolute means are context.

## Scheduler comparison

`python scripts/run_experiment.py --scenario shift --seeds 101 202 303 404 505 --slots 1500`

| strategy | intercepts/slot | intercepted | delay | Pd | Pfa | efficiency | reward |
|---|---|---|---|---|---|---|---|
| **Smart Scan** | **0.2760** | 0.0839 | 2.52 | 0.900 | 0.028 | 0.295 | 375.3 |
| Greedy ML | 0.2513 | 0.0819 | 2.47 | 0.895 | 0.028 | 0.272 | 338.1 |
| UCB | 0.2235 | 0.0760 | 7.07 | 0.908 | 0.028 | 0.244 | 297.9 |
| Thompson Sampling | 0.1648 | 0.0822 | 4.12 | 0.890 | 0.028 | 0.188 | 209.0 |
| Round-Robin | 0.0471 | 0.0949 | 6.17 | 0.922 | 0.028 | 0.073 | 34.7 |
| Random | 0.0440 | 0.0765 | 5.50 | 0.907 | 0.028 | 0.070 | 30.0 |

Paired, per seed, on `average_intercept_rate`:

| strategy | vs Round-Robin | wins | vs Greedy ML | wins |
|---|---|---|---|---|
| Smart Scan | 5.88x +/- 1.55 | 5/5 | 1.09x +/- 0.21 | 4/5 |
| Greedy ML | 5.44x +/- 1.17 | 5/5 | 1.00x | - |
| UCB | 4.74x +/- 0.64 | 5/5 | 0.90x +/- 0.19 | 2/5 |
| Thompson Sampling | 3.51x +/- 0.92 | 5/5 | 0.65x +/- 0.15 | 0/5 |
| Random | 0.94x +/- 0.13 | 2/5 | 0.18x +/- 0.02 | 0/5 |

Reading it honestly:

- **The closed loop is worth a great deal over an open-loop sweep.** Smart Scan
  intercepts 5.9x more activity per slot than round-robin and wins on every seed.
  Even Thompson Sampling with no ML at all is 3.5x round-robin, so a large part
  of the gain comes simply from reacting to what the receiver observes.
- **The advantage over Greedy ML is modest and not consistent.** 1.09x on
  average, winning 4 of 5 seeds, with a per-seed range of 0.74x to 1.36x. Five
  seeds cannot establish that as a real effect. It should be treated as
  "comparable, plausibly slightly better" until it is run over more seeds.
- **Exploration costs something when the model is good.** UCB and Thompson both
  land below Greedy ML here. On a scenario whose family the model was trained on,
  exploitation is hard to beat; exploration is insurance against the model being
  wrong, and this scenario does not collect on that insurance often.
- **Round-Robin wins on `intercept_ratio`** (0.095 against 0.084), for the same
  reason: a blind sweep touches more distinct bursts, while Smart Scan
  concentrates where activity pays out and wins on totals and on delay.
- Pd and Pfa are flat across policies, as they must be.

## Layer ablation

`python scripts/run_experiment.py --ablation --scenario shift --seeds 101 202 303 --slots 1500`

Paired against the mathematical layer alone (Thompson with no ML prior):

| arm | ratio | wins |
|---|---|---|
| Offline model only (Greedy ML) | 1.443 +/- 0.255 | 3/3 |
| Complete Smart Scan | 1.385 +/- 0.298 | 3/3 |
| Smart Scan without the adaptive layer | 1.258 +/- 0.223 | 3/3 |
| Mathematical layer only | 1.000 | - |

The ML prior is the dominant contributor: adding it to the mathematical layer is
worth 26% to 44%, on every seed. The adaptive layer adds roughly 10% on top
(1.385 against 1.258), which at three seeds is suggestive rather than
established. Greedy ML is level with or slightly ahead of the complete strategy,
consistent with the benchmark above.

## Offline model

`python scripts/train_models.py --runs-per-scenario 3 --slots 1500`

245,271 rows across all seven scenarios, positive rate 0.051. Split by *run*, so
no run contributes to two partitions.

| model | test ROC-AUC | val ROC-AUC | Brier | accuracy | precision | recall |
|---|---|---|---|---|---|---|
| XGBoost | 0.748 | 0.820 | 0.0438 | 0.951 | 0.689 | 0.119 |
| Random Forest | 0.734 | 0.813 | 0.0436 | 0.951 | 0.706 | 0.119 |

XGBoost is chosen on measured ranking quality, and the margin over the Random
Forest baseline is small: 0.748 against 0.734 test ROC-AUC. That is worth stating
plainly rather than claiming a decisive win.

Recall at a 0.5 threshold is low (0.119), which looks alarming and is not the
relevant quantity. With a 5% base rate the model rarely crosses 0.5, but the
scheduler never thresholds: it uses probabilities to *rank* candidate bands and
to set Beta priors. ROC-AUC and calibration are the metrics that matter here, and
the end-to-end benchmark above is the real test.

Top features by gain: `rate_long`, `last_detected`, `rate_medium`,
`consec_detections`, `detections_log`, `observed_rate`, `since_detection_norm`.
All are observation-derived, which is the intended behaviour.

## Change detection: what works and what does not

This is the weakest part of the system and the measurements say so.

**Attempt 1 - Page-Hinkley on the offline model's prediction error.** Fired 4 to
6 times per 1500-slot run on environments with a single change point, and just as
often on stationary ones. The cause is real: this system generates its own
non-stationarity. As the scheduler concentrates on productive bands, the
distribution of what gets scanned changes continuously by design, and a
cumulative detector reads the policy's own improvement as environmental change.

**Attempt 2 - two-window test on the same signal.** Compare mean error over the
recent window against the preceding baseline, alarm past a threshold in standard
errors. Alarms fell to 1-3 per run, but detection latency after a real change
point was 330-380 slots and stationary runs alarmed just as often. No
discrimination.

**Attempt 3, current default - two-window test on belief contradiction.** Score
only looks at bands whose accumulated evidence exceeds `drift_min_evidence`, and
score them against the *belief* rather than the offline model. Exploring an
unknown band no longer registers; what remains is closer to "is what we
confidently believed still true".

Measured over six seeds, 1600 slots, `periodic` as the stationary control
(`mixed` is a poor control: its burst and intermittent sources genuinely start
and stop, so an alarm there is not necessarily false):

| signal | z | alarms on stationary control | alarms on shift | change point detected |
|---|---|---|---|---|
| belief contradiction | 3.0 | 2, 3, 2, 3, 2, 3 | 3, 3, 2, 3, 2, 4 | 6 of 6, mean latency **114** slots |
| belief contradiction | 4.0 | 1, 3, 0, 3, 2, 3 | 2, 2, 1, 2, 0, 3 | 4 of 6, mean latency 218 slots |
| belief contradiction | 5.0 | 1, 2, 0, 1, 2, 0 | 1, 1, 1, 2, 0, 1 | 5 of 6, mean latency 321 slots |
| model error | 3.0 | 2, 4, 0, 3, 3, 5 | 4, 3, 3, 3, 2, 5 | 6 of 6, mean latency 194 slots |
| model error | 4.0 | 2, 4, 0, 2, 3, 2 | 4, 2, 1, 4, 1, 3 | 6 of 6, mean latency 298 slots |
| model error | 5.0 | 0, 2, 0, 0, 2, 1 | 1, 2, 0, 1, 1, 3 | 5 of 6, mean latency 316 slots |

Belief contradiction beats model error on latency at every threshold, which is
why it is the default. Raising the threshold buys fewer alarms on the control at
the cost of much slower detection, and never buys discrimination.

**Conclusion.** At z = 3.0 the detector catches every scripted change point,
about 114 slots after it happens, and that is a genuine improvement over the
370-slot latency of the model-error signal. But it also fires roughly 2.5 times
per run on a stationary environment, so **it cannot be used to assert that an
environment shift occurred.** Sensitivity is good; specificity is not.

What the alarm is therefore used for, and all it is used for:

- shorten the evidence half-life, so the belief re-learns faster;
- withdraw adaptive influence and demote the layer to shadow mode.

Both are conservative. A false alarm costs a little forgetting and a pause in
adaptive influence; it never causes a bad scan directly. The UI labels the
condition "belief contradicted" rather than "environment shift", and the event
detail says the same, because the measurement does not support the stronger
claim.

Adaptation itself does not depend on this detector. The Beta evidence decay
(`bayesian.decay_lambda`, half-life about 139 slots) re-learns after a change on
its own, which is visible in `average_adaptation_time` and is partly why the
error signal is so hard to read: the system often recovers before degradation
accumulates enough to notice.

## Reproducibility

- A run is fully determined by seed plus configuration
  (`test_runs_are_reproducible_from_the_seed`).
- Environment, receiver and decision randomness use separate seeded generators.
- Every decision, observation, prediction, posterior and adaptive event is
  persisted with its run id and slot.
- 72 tests: `python -m pytest tests -q`.

## Known limitations

1. Change detection has poor specificity, as measured above.
2. Absolute performance is specific to the synthetic environment. The comparison
   between policies is the transferable result; the raw intercept rate is not.
3. The offline model's advantage over its Random Forest baseline is small.
4. Smart Scan's edge over Greedy ML is not established at five seeds. The gain
   over open-loop scanning is large and consistent; the gain over greedy
   exploitation of the same model is not yet demonstrated.
5. The model is trained on the same scenario families it is evaluated on, which
   flatters exploitation. Training on one family and testing on an unseen one
   would be the sterner test of whether exploration earns its place, and has not
   been run.
6. Truth-dependent metrics (measured Pd, Pfa, intercept ratio and delay) require
   simulation or labelled replay. Against a live receiver they go quiet; the
   scheduler still works, because it consumes detections only.
7. Beta-Bernoulli treats activity as binary per band and per slot. Signal
   structure below that resolution is out of scope by design.
8. The adaptive layer usually remains in shadow mode over a 1500-slot run: its
   acceptance criteria are deliberately strict and drift alarms reset the
   stability streak. Longer runs, or relaxed thresholds in `configs/default.yaml`,
   are needed to exercise the assisted and control states.


