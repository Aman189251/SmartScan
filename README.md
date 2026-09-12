# Smart Scan Strategy

Adaptive, machine-learning-assisted scan scheduling for a constrained Electronic
Support receiver, built as a native Windows research workstation.

**DRDO Problem Statement 26055** - *Smart Scan strategy for Electronic Warfare in
the absence of prior reliable intelligence of emitters and their operating
characteristics.*

---

## What the problem is

A receiver cannot watch the whole surveillance space at once. Its instantaneous
bandwidth is an order of magnitude smaller than the spectrum it is responsible
for, so it must sweep, and every look it spends in one place is a look it did not
spend somewhere else. Conventional open-loop strategies sweep on a fixed,
pre-mission schedule: fast and predictable, but they keep paying attention to
quiet regions and can be slow to reach new activity.

This system replaces the fixed sweep with a closed loop that learns where to
look next, and measures whether that actually helps.

## How it decides

```
  observation history
          |
          v
  [ Layer 1 ]  offline XGBoost  ->  activity probability per band
          |
          v
  [ Layer 2 ]  Beta-Bernoulli belief + Thompson Sampling  ->  next band
          |
          v
  [ Layer 3 ]  scheduler validates receiver constraints, dispatches, records
          |
          v
  normalised observation  ->  [ hidden adaptive learner ]  ->  back to Layer 2
```

The model never picks a band on its own. It contributes prior knowledge; live
evidence updates that prior; and the mathematical layer turns belief *and its
uncertainty* into an action. That separation is what lets exploration happen at
all: a band nobody has looked at recently is uncertain, and uncertainty is worth
something.

The adaptive learner runs continuously in the background. It starts in shadow
mode with **zero** influence and is promoted only after measured acceptance
criteria hold for several consecutive evaluations. A drift alarm withdraws its
influence immediately.

## Does it work

Six policies on identical seeded environments, `shift` scenario, five seeds,
1500 slots each (`python scripts/run_experiment.py --scenario shift --seeds 101 202 303 404 505`):

| strategy | intercepts/slot | vs Round-Robin (paired) | delay | Pd | Pfa | efficiency |
|---|---|---|---|---|---|---|
| **Smart Scan** | **0.276** | **5.88x** +/-1.55, wins 5/5 | 2.52 | 0.900 | 0.028 | 0.295 |
| Greedy ML | 0.251 | 5.44x +/-1.17, wins 5/5 | 2.47 | 0.895 | 0.028 | 0.272 |
| UCB | 0.224 | 4.74x +/-0.64, wins 5/5 | 7.07 | 0.908 | 0.028 | 0.244 |
| Thompson Sampling | 0.165 | 3.51x +/-0.92, wins 5/5 | 4.12 | 0.890 | 0.028 | 0.188 |
| Round-Robin | 0.047 | 1.00x | 6.17 | 0.922 | 0.028 | 0.073 |
| Random | 0.044 | 0.94x +/-0.13, wins 2/5 | 5.50 | 0.907 | 0.028 | 0.070 |

The comparison to read is the paired column. Absolute rates carry large
between-seed variance because one seed's environment simply holds more activity
than another's; since every policy runs on the *identical* environment for a
given seed, the per-seed ratio removes that shared variance.

- Smart Scan intercepts **5.9x** more activity per slot than the conventional
  round-robin sweep, and wins on every seed.
- Against Greedy ML it is **1.09x +/-0.21, winning on 4 of 5 seeds**. That is a
  modest and not-fully-consistent edge, not a decisive one, and the layer
  ablation in [docs/VALIDATION.md](docs/VALIDATION.md) says the same thing: on
  this scenario the offline model is doing most of the work.
- Measured Pd and Pfa sit at the configured receiver characteristics for every
  policy, as they must: those are properties of the receiver, not the scheduler.

One caveat visible in the full metric set: round-robin touches a slightly larger
*fraction of distinct activity bursts* (0.095 against 0.084), because a blind
sweep spreads itself evenly. Smart Scan concentrates where activity pays out and
wins heavily on total interceptions and on delay. Which of those matters is a
mission question, so both are reported rather than one.

## Quick start

```powershell
cd SmartScan
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe scripts\train_models.py    # ~1 minute, optional
.\.venv\Scripts\python.exe frontend\main.py           # launch the workstation
```

The application runs entirely offline. It starts its own local service on
`127.0.0.1:8077`, opens the desktop window, and needs no internet connection.

Without a trained model the system still runs: it falls back to a transparent
observation-only heuristic and says so, in the status bar and in the API, so a
demonstration can never be mistaken for a trained result.

### Other entry points

```powershell
scripts\train_models.py     --runs-per-scenario 3     # dataset + XGBoost + Random Forest
scripts\run_experiment.py   --scenario shift          # headless benchmark
scripts\run_experiment.py   --ablation                # layer-by-layer contribution
scripts\run_api.py          --port 8077               # service only, no GUI
scripts\maintain.py         --status                  # database and registry size
python -m pytest tests -q                             # 72 tests
```

A run records one observation, decision, prediction and belief per slot, so a few
dozen benchmark runs reach a few hundred megabytes. `scripts\maintain.py
--prune-details --keep 8 --vacuum` drops per-slot detail from older runs while
keeping every run summary, metric and adaptive event. On the reference database
that took 98.7 MB down to 12.6 MB.

## The interface

Four views, all reading the same local service.

- **Dashboard** - the current scan, the spectrum/time map, the model-against-belief
  bars, the scan timeline, and a decision panel that explains the current choice
  in the numbers the scheduler actually used.
- **Learning monitor** - offline model provenance and metrics, the five-state
  adaptive ladder, the acceptance criteria as a live checklist, and the drift log.
- **Experiment arena** - run every policy on identical environments and compare.
- **Models and runs** - registry, training, feature importance, run history.

## Layout

```
SmartScan/
  backend/app/
    simulator/    synthetic environment, scenarios, receiver model
    receiver/     ReceiverAdapter boundary (simulator, replay, external)
    ml/           features, dataset generation, training, registry, prediction
    adaptive/     Beta-Bernoulli belief, drift detection, promotion ladder
    scheduler/    Random, Round-Robin, Greedy ML, UCB, Thompson, Smart Scan
    evaluation/   metrics and the experiment engine
    core/         engine, controller, shared types
    db/           SQLAlchemy models and the batched recorder
    api/          FastAPI routers
  frontend/       PySide6 workstation
  configs/        default.yaml - every tunable parameter
  docs/           architecture, integration, validation
  scripts/        train, benchmark, serve
  tests/          72 tests
```

## Scope

This is a **synthetic research simulator**. Bands are abstract sensing regions
(B01..B32), activity is generated by the simulator, and Pd/Pfa are simulation
parameters rather than equipment specifications. The system contains no real
emitter parameters, no signal exploitation, no decryption, no jamming and no
targeting. It studies one question: given a limited number of looks, where
should the next one go.

Integration with an external or laboratory receiver happens at a single
boundary, `backend/app/receiver/adapter.py`. See [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - layers, data flow, the ground-truth barrier
- [docs/INTEGRATION.md](docs/INTEGRATION.md) - receiver adapter contract and the path to external hardware
- [docs/VALIDATION.md](docs/VALIDATION.md) - metric definitions, methodology, calibration results

