# Integration guide

## The boundary

Everything above the adapter is hardware independent. Features, the offline
model, Bayesian belief, scheduling, evaluation, persistence and the UI all
consume one type, `NormalizedObservation`, and none of them know or care where
it came from.

```
  Smart Scan core
        |
        v
  ReceiverAdapter          <-- the only file that knows an external interface
   |        |        |
   |        |        +--  ExternalReceiverAdapter   (lab / external service)
   |        +-----------  RecordedDataAdapter       (replay of stored logs)
   +--------------------  SimulatorReceiverAdapter  (synthetic environment)
```

Integrating a new source of observations means writing one subclass. No other
module changes.

## The interface

```python
class ReceiverAdapter(ABC):
    def connect(self) -> None
    def disconnect(self) -> None
    def get_capabilities(self) -> dict          # n_bands, capacity, scan_duration, ...
    def configure_scan(self, **params) -> None
    def start_scan(self, command: ScanCommand) -> None
    def stop_scan(self) -> None
    def get_observation(self) -> NormalizedObservation
    def get_status(self) -> dict
    def health_check(self) -> dict              # provided; override if you can do better
```

`execute(command)` is a convenience that dispatches and collects in one call.
The adapter is also a context manager, so `with adapter: ...` connects and
disconnects.

## The normalised observation

Every adapter returns this, whatever the source:

| field | meaning |
|---|---|
| `run_id` | run this observation belongs to |
| `source_id` | which adapter produced it (`simulator`, `recorded`, `external`) |
| `slot` | discrete decision step |
| `timestamp` | wall-clock time |
| `band_id` | abstract band that was observed |
| `scan_duration` | slots the look consumed |
| `outcome` | `HIT` / `MISS` / `FALSE_ALARM` / `CORRECT_NON_DETECTION` |
| `detected` | whether the receiver reported energy |
| `quality` | normalised confidence in [0, 1] |
| `receiver_state` | adapter-reported state string |
| `metadata` | free-form, never consumed by the intelligence layers |

### A note on outcomes from a live source

`HIT` versus `FALSE_ALARM`, and `MISS` versus `CORRECT_NON_DETECTION`, are
distinctions that require ground truth. A simulator has it. A live receiver does
not. `ExternalReceiverAdapter` therefore maps a bare detection to `HIT` and a
non-detection to `CORRECT_NON_DETECTION` unless the external service explicitly
supplies a four-way outcome.

This matters for what you can measure. With a live source, the Bayesian layer,
the scheduler and the adaptive layer all work normally, because they consume
`detected` only. The *evaluation* metrics that need truth - measured Pd, measured
Pfa, intercept ratio, intercept delay - are only meaningful in simulation or
replay against a labelled recording.

## Reference wire protocol

`ExternalReceiverAdapter` speaks line-delimited JSON over TCP. One request
object per line, one response object per line.

```
-> {"cmd": "capabilities"}
<- {"n_bands": 32, "capacity": 1, "scan_duration": 1}

-> {"cmd": "configure", "params": {"dwell": 2}}
<- {"ok": true}

-> {"cmd": "scan", "run_id": "r-01", "slot": 41, "band_id": 7, "dwell": 1}
<- {"outcome": "HIT", "quality": 0.93, "receiver_state": "READY"}

-> {"cmd": "status"}
<- {"state": "READY", "current_band": 7}

-> {"cmd": "stop"}
<- {"ok": true}
```

A response may send `{"detected": true}` instead of `outcome`.

**This wire format is this project's own convention.** It exists so the
integration path can be exercised end to end before any real interface is
available. It is not a claim of compatibility with any specific external or
undisclosed system. When an approved interface specification exists, implement it
as a new `ReceiverAdapter` subclass; the rest of the codebase is unaffected.

## Using an adapter

```python
from backend.app.core.engine import SmartScanEngine
from backend.app.receiver.adapter import ExternalReceiverAdapter

adapter = ExternalReceiverAdapter(host="10.0.0.5", port=9500)
engine = SmartScanEngine(cfg, strategy="smart_scan", adapter=adapter)
engine.run()
```

For repeatable research against recorded data:

```python
from backend.app.receiver.adapter import RecordedDataAdapter

adapter = RecordedDataAdapter("datasets/session_04.jsonl")
```

## Integration checklist

1. Implement the subclass; map the external outcome vocabulary onto `Outcome`.
2. Report honest capabilities: band count, looks per slot, scan duration, any
   revisit lockout. The scheduler respects declared constraints and the receiver
   model rejects commands that violate them.
3. Decide how `slot` maps to real time. The core is slot-driven; a real receiver
   is clock-driven. The adapter owns that translation.
4. Run `health_check()` before a session and treat a failure as a stop condition.
5. Remember that truth-dependent metrics go quiet on a live source. Use replay
   against labelled recordings when you need them.

## Deployment

```powershell
pyinstaller --noconfirm --windowed --name SmartScan `
  --add-data "configs;configs" --add-data "models;models" `
  frontend\main.py
```

The application is offline-first: local SQLite, local service on loopback, no
cloud dependency and no internet access required at run time.
