# Architecture

## Layers

```
                            OPERATOR
                               |
                     PySide6 workstation
                               |
                  FastAPI local service (loopback)
                               |
                      SimulationController
                               |
                        SmartScanEngine
                               |
        +----------------------+----------------------+
        |                      |                      |
   SIMULATION             INTELLIGENCE             STORAGE
        |                      |                      |
  environment            feature engine           SQLAlchemy
  receiver model          XGBoost (L1)              SQLite
  ReceiverAdapter    Beta-Bernoulli belief (L2)
                      Thompson Sampling (L2)
                       adaptive learner
                               |
                        evaluation engine
```

Each layer is replaceable. The scheduler does not know whether observations came
from a simulator or a receiver. The UI does not know how the engine works. The
engine does not know how anything is stored.

## One decision cycle

`SmartScanEngine._decision_cycle` is the whole loop, in order:

1. Ask the receiver which bands it is allowed to tune to.
2. Build the feature matrix for every band from observation history.
3. Run the offline model once, batched over all bands.
4. Let the adaptive layer form its own estimate and its shadow recommendation.
5. Build a `DecisionContext` and hand it to the scheduler.
6. Anchor Beta priors on the (possibly blended) probability, draw Thompson
   samples over the candidate set, select a band.
7. Dispatch a `ScanCommand` through the adapter; receive a normalised observation.
8. Score the outcome in the evaluation engine.
9. Update features, belief evidence, evidence decay, and the adaptive monitors.
10. Buffer the row for persistence.

## The ground-truth barrier

The simulator holds a hidden activity grid. The rule is that it reaches exactly
two places, both after the fact:

| component | ground truth | why |
|---|---|---|
| environment | yes | it generates it |
| receiver model | internally | to turn truth into an imperfect observation |
| feature engine | **no** | built only from `NormalizedObservation` |
| offline model | **no** | consumes features; truth is a training label only |
| Bayesian learner | **no** | consumes detections |
| scheduler | **no** | consumes probabilities and posteriors |
| adaptive layer | **no** | consumes prediction error against observations |
| evaluation engine | yes | to measure performance afterwards |

`environment.is_active()` is called in exactly one place in the live path, in
`engine._decision_cycle`, and its value goes straight to the evaluation engine.
`tests/test_engine.py::test_scheduler_never_receives_ground_truth` asserts that
the decision context carries no route to it.

Offline training is the one place truth becomes a label. Even there it is never a
feature: `ml/dataset.py` builds features from observation history and attaches
the label separately, and an `observation` labelling mode exists that uses no
truth at all, for the day a dataset has to be built from recorded receiver data.

## Layer 1 - offline prediction

28 features per band, all observation-derived: lifetime and windowed detection
rates, two EWMAs, staleness, consecutive-outcome runs, observed transition
statistics, scan-effort share, cross-band ranks, cyclic phase context, per-phase
history, burstiness, and last outcome. Version `fe-1.0`; a model trained on a
different feature version is refused at load rather than silently misused.

XGBoost is the primary model, Random Forest the baseline, both trained under the
same group-wise split (splitting by *run*, so no run contributes to two
partitions). Probabilities are isotonically calibrated on the validation split
and kept only if calibration actually improves, because the mathematical layer
consumes them as a prior: a miscalibrated probability poisons the Beta prior even
when its ranking is fine.

Every artifact carries model id, version, feature version, dataset version,
training date, scenario coverage, hyperparameters, validation metrics and
calibration metrics.

## Layer 2 - mathematical decision

Each band holds a Beta distribution:

```
alpha_b = k * p_ML(b) + H_b        beta_b = k * (1 - p_ML(b)) + M_b
H_t = lambda * H_(t-1) + hit_t     M_t = lambda * M_(t-1) + miss_t
```

The model anchors the prior; decayed detection and non-detection counts supply
the likelihood. In `dynamic` prior mode the anchor is re-applied every slot, so a
fresh prediction moves belief immediately; `static` fixes it at first contact.

Thompson Sampling draws one sample per candidate and takes the highest. High
posterior mean drives exploitation, wide posteriors give under-observed bands
their chance. The candidate set is the top-k by probability plus the stalest
available band, which prevents starvation, and a small forced-exploration rate
puts a floor under coverage.

## The hidden adaptive layer

It maintains its own faster-forgetting belief, scores itself against the offline
model on every executed scan, watches for change, and is gated behind a promotion
ladder:

```
OFFLINE_PRIOR -> LEARNING -> SHADOW -> ADAPTIVE_ASSISTED -> ADAPTIVE_CONTROL
                             ^                 |
                             +-- drift alarm --+
```

Promotion is never on a slot count. It requires observation volume, recent Brier
score, calibration error, a measured advantage over the offline model, and no
active drift alarm - all holding for several consecutive evaluations. Influence
is capped at `adaptive.max_influence` in ADAPTIVE_ASSISTED. A drift alarm
demotes to SHADOW immediately and shortens the evidence half-life.

### Why the change detector is a two-window test

The obvious signal is the offline model's prediction error, watched with
Page-Hinkley. That was tried first and it alarmed several times per run on a
*stationary* environment. The cause is real and interesting: this system creates
its own non-stationarity. As the scheduler concentrates on productive bands, the
distribution of what gets scanned changes continuously by design, so a cumulative
detector reads the policy's own improvement as environmental change.

The default detector therefore compares mean error over the recent window against
the preceding baseline window and alarms only on a gap exceeding both an absolute
floor and a threshold in standard errors. Gradual self-inflicted drift moves both
windows together and cancels. Page-Hinkley remains available via
`adaptive.drift_detector: page_hinkley`, fed a smoothed error. Calibration
figures are in [VALIDATION.md](VALIDATION.md).

## Persistence

SQLite through SQLAlchemy: `scenarios`, `experiments`, `runs`, `bands`,
`observations`, `predictions`, `beliefs`, `decisions`, `adaptive_events`,
`metrics`, `models`, `logs`.

A run produces one decision per slot but a probability and a posterior for
*every* band at every slot. Storing the full grid would dominate the database, so
`RunRecorder` writes the acted-on band every slot and a complete all-band
snapshot on a stride, and flushes in batches so the loop is never paced by disk.

## Threading

The simulation runs on a worker thread owned by `SimulationController`. The
FastAPI service answers from its own threads. The Qt UI polls over HTTP on
timers split by cost: status at 4 Hz, snapshots at ~1.7 Hz, the activity grid and
background jobs at 0.5 Hz. Simulation rate is independent of display rate, so the
loop can run flat out while the interface stays responsive.

Long jobs - dataset generation, training, benchmarks - run through a `JobManager`
on their own threads and report progress the UI polls, so no request ever blocks
on a multi-minute computation.

## Reproducibility

A run is fully determined by its seed and configuration. Environment, receiver
and decision randomness come from separate generators derived from that one seed.
`tests/test_engine.py::test_runs_are_reproducible_from_the_seed` asserts that two
engines with the same seed produce identical outcome counts and reward.
