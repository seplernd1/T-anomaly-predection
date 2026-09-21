# Seple Device Audit & Analytics

A Python/Jupyter-based system for scraping ThingsBoard devices, harvesting telemetry, generating reports, and creating interactive dashboards.

## Project Structure

- **`TB_Full_Harvest_v11.ipynb`**: Current notebook for device discovery, full data extraction, and state processing.
- **`run_nightly_audit.py`**: Automation script to run the audit daily.
- **`pull_current_state_snapshots.py`**: Pulls daily current-state snapshots for every device.
- **`run_current_state_snapshot.bat`**: Windows runner for the current-state snapshot pull.
- **`schedule_current_state_snapshot.ps1`**: Registers the current-state pull in Windows Task Scheduler.
- **`tb_audit_v7.xlsx`**: Generated Excel report.
- **`tb_dashboard_v7.html`**: Interactive Plotly/Dash dashboard.
- **`.env`**: Environment variables for ThingsBoard credentials.

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Itinerant18/Data_scrapping_thgingsboard_ml-intern.git
    cd Data_scrapping_thgingsboard_ml-intern
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: `requirements.txt` is assumed to contain `pandas`, `plotly`, `requests`, `selenium`, `openpyxl`, `dotenv`)*

3. **Configure Environment:**
    Create a `.env` file in the root directory:

    ```env
    TB_HOST=https://seple.iot-private.cloud
    TB_EMAIL=your_email@example.com
    TB_PASSWORD=your_password
    REQUEST_DELAY=0.2
    ```

## Usage

### Run the Notebook Manually

Open `TB_Full_Harvest_v11.ipynb` in Jupyter/VS Code and run the cells.

#### Long-run safety (memory + resume)

The 365-day fetch pulls ~821 keys × up to 5000 points for ~160 devices. Holding all of that in RAM is what used to kill the run silently at ~13/160 devices (no traceback — the OS reclaimed the process). Each device's payload is now written to `harvest_cache/ts/<device_id>.json.gz` as soon as it arrives, only a point-count summary stays in memory, and the alignment step streams one device at a time, so peak usage is one device instead of the whole fleet.

- A crashed or cancelled run **resumes**: cached devices are skipped. `REFRESH_TS_CACHE=1` re-fetches them, `TS_CACHE_DIR` relocates the cache.
- `harvest_cache/` is gitignored (bulky, re-fetchable).
- Run it visibly or redirect the logs — a detached run produces no console output:

```bash
python -u -m papermill TB_Full_Harvest_v11.ipynb harvest_cache/TB_Full_Harvest_v11_executed.ipynb
```

Two failure modes worth knowing:

1. `automate_harvest.bat` calls `.venv\Scripts\python.exe`, which currently has **no papermill installed**, so that job fails immediately with `ModuleNotFoundError`. It also runs papermill **in place**, so a crash (or the IDE autosaving its buffer) can leave the notebook half-executed. Install papermill into `.venv` and/or prefer the command above with a separate output notebook.
2. A malformed first line in `.env` (any characters typed before `TB_HOST`) makes `python-dotenv` skip the line and the notebook dies at Cell 3 with `KeyError: 'TB_HOST'`. Keep `.env` to plain `KEY=value` lines; it must stay gitignored.

### Automated Daily Execution

The `run_nightly_audit.py` script automates the notebook run:

```bash
python run_nightly_audit.py
```

This will:

1. Execute the notebook.
2. Generate an output report in the `audit_reports/` folder.
3. Send a notification to Slack and Email (if configured).

### Pull ALL Data (One Command)

`pull_all_data.py` is a single orchestrator that pulls every data type from the tenant into a versioned run folder (`data_harvest/runs/<run_id>/`) with a `manifest.json` recording row counts and per-step errors. Telemetry uses retried, window-split fetching with per device/key coverage records and checkpoints; a 5000-point (cap-sized) response is treated as truncated until proven otherwise.

```bash
pip install -r requirements.txt                  # ambient 3.14 has everything; .venv does NOT
python pull_all_data.py --plan-only              # show what would run, fetch nothing
python pull_all_data.py --max-devices 5 --max-keys-per-device 2   # smoke test
python pull_all_data.py --device-offset 30 --max-devices 3        # sample specific devices
python pull_all_data.py --telemetry-days 90      # limit telemetry lookback
python pull_all_data.py --all-asset-relations    # also walk standalone asset trees
python pull_all_data.py --resume-from <run_id>   # resume: frozen window, skip done chunks
python pull_all_data.py --allow-partial          # exit 0 despite incomplete windows (default: exit 5)
```

TLS is verified by default. This tenant uses a private cert, so passes need `--insecure-skip-tls-verify` (or `TB_INSECURE_TLS=1`) until the cert is fixed — the flag prints a warning.

Useful flags: `--skip-telemetry`, `--skip-events`, `--skip-alarms`, `--skip-relations`, `--key-filter "active|heartbeat"` (regex on telemetry keys), `--min-chunk-hours`, `--max-retries`, `--run-id`.

Run folder contents (`registry/`, `telemetry/`, `events/`, `alarms/`, `coverage/`, `quarantine/`, `reports/`):

- `registry/devices.csv` (+`customers.csv`, `assets.csv`) — entity registry with `eligible`, `eligibility_reason`, retrieval metadata
- `registry/relations.csv` — bank → branch → device graph (walked per device)
- `registry/device_attributes.csv` — all scopes (SERVER/CLIENT/SHARED) incl. connectivity keys
- `registry/device_telemetry_keys.csv` — discovered telemetry keys per device
- `telemetry/telemetry_<device_id>.parquet` — timeseries per device (device_id, raw_key, key, ts, ts_iso, value)
- `events/events_<type>.csv` (LC_EVENT, ERROR, STATS, DEBUG) and `alarms/alarms.csv` — outage evidence
- `coverage/telemetry_coverage.jsonl` + `source_coverage.jsonl` + `checkpoints.json` — per device/key/window provenance and resume state
- `reports/device_exclusions.csv`, `reports/completeness.json` — eligibility audit and run completeness
- `manifest.json` — per-step rows/files/errors/timing, both windows, coverage summary

Device eligibility (`REAL_DEVICE_*` env: name patterns, `REAL_DEVICE_REQUIRE_CUSTOMER`, `REAL_DEVICE_MIN_KEYS`, allow/deny lists) classifies real vs test/demo/simulator devices with per-device reasons; only eligible devices are pulled.

### Build Anomaly Dataset (raw telemetry only)

`build_anomaly_dataset.py` builds the unsupervised anomaly dataset from a pull run's validated parquet — never from the rule-derived notebook exports. Default-deny feature policy (`feature_policy.json`): unknown keys are excluded until the lead allowlists reviewed sensor keys.

```bash
python build_anomaly_dataset.py --run-dir data_harvest/runs/<run_id> --plan-only
python build_anomaly_dataset.py --run-dir data_harvest/runs/<run_id>
```

Outputs in `training_data/anomaly_<run_id>/`: `anomaly_features.parquet`, `scaler.json` (fit on train only), `feature_dictionary.csv`, `data_card.md`, `exclusion_report.json`, `split_report.json`.

### Build Training Data (leakage-free)

`build_training_dataset.py` reads the Postgres tables (`public.device_telemetry`, `public.device_event`, hierarchy tables) and produces a point-in-time feature matrix for anomaly detection. It also *prepares* the 24-hour outage-risk label but refuses to invent it: without independent verified evidence, every sample is censored and no supervised model should be trained.

```bash
# connection string comes from the environment (never from a file in the repo)
export TRAIN_DB_URL='postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DB?sslmode=require'
# ...or set PGHOST / PGPORT / PGDATABASE / PGUSER / PGPASSWORD

python build_training_dataset.py --check-db          # verify connection + expected columns (read-only)
python build_training_dataset.py --plan-only         # print the extraction plan, no DB calls
python build_training_dataset.py --start 2026-01-01 --end 2026-04-01
python build_training_dataset.py --skip-extract      # rebuild features from staged parquet
python build_training_dataset.py --max-partitions 2 --max-devices 10   # smoke test
```

Outputs in `training_data/` (gitignored):

- `anomaly_samples.parquet` — one row per device anchor: `device_id`, `anchor_ts`, feature columns, `eligibility`, `censor_reason`, `split` (plus `label_status`, `y_outage_24h`, `max_source_ts`)
- `outage_evidence_audit.parquet` — `device_id`, `evidence_ts`, `evidence_type`, `evidence_strength`, `source_event_id`, `observed_at`, `redacted_evidence_reference` (no raw payloads or values)
- `data_quality_report.md` / `data_quality_metrics.json` — coverage, quarantine reasons, parse failures, missingness/freshness, duplicates, latency, weak-evidence counts, censored-vs-eligible counts, split balance

Configuration: `training_config.json` (windows, exclusions, label rules, splits, purge gap).

Guarantees:

- **Read-only DB access.** Every statement must be `SELECT`/`WITH`, and the session opens with `default_transaction_read_only = on`.
- **Bounded extraction.** Daily partitions × key batches with a per-query row cap; the hypertable is never aggregated whole.
- **Point-in-time features.** A row at anchor `T` uses only records with `time <= T`; `max_source_ts` is recorded per sample and asserted ≤ `anchor_ts` (`--no-strict-leakage-check` to disable the strict guard).
- **Capped imputation.** Carry-forward stops after `fill_cap_hours`; beyond that the value is NaN and `*__last_age_h` / `*__stale` remain.
- **Exclusions.** Identifiers (`device_id`, `imei`, `ip`), GPS, secrets, raw payloads, rule-derived scores, severity/alarm flags and target-like attributes never become features.
- **Chronological, grouped, purged splits.** Customers/branches/devices stay in one split; purge bands are cut at the midpoint between adjacent splits so no label horizon or feature window overlaps.
- **Labels only from verified evidence.** Lifecycle connect/disconnect events and offline/no-data alarms are strong evidence. `device_event.payload.data.currentAttr` and other state snapshots are *weak* evidence and can never create a positive label.

If no verified evidence exists, the run exits with code `3` and prints a clear **STOP** notice; use the `anomaly_eligible == True` rows for unsupervised anomaly detection in the meantime. Tests: `python -m pytest tests -q` (44 tests, including a proof that no post-anchor data can enter a feature row).

### Key Spec (new telemetry key document)

The key-update document (`key_spec_update.docx` / `.pdf`, same content) defines the new namespaced key spec (`gateway.*`, `cctv.*`/`rock.*`, `system_status.*`, `basSystemIntegration.*`, `timeLock.*`, `accessControl.*`). Tooling around it:

```bash
python build_key_spec_manifest.py        # doc -> key_spec_manifest.csv/.json + key_spec_aliases.json
python verify_key_spec.py                # manifest vs live tenant -> key_spec_verification.csv
```

- `key_spec_manifest.csv` — 66 logical keys, grouped (gateway/cctv/bas/ias/fas/timelock/access_control/power/statusbox), with intent + derivation notes
- `key_spec_aliases.json` — tenant key -> canonical spec name (namespace flattening + case drift like `Rock.*` vs `rock.*`)
- `key_spec_verification.csv` — per-key live-device counts; **23/66 keys are live**, 43 not yet posted by devices
- `pull_all_data.py --alias-map key_spec_aliases.json` — telemetry output gets a `raw_key` column (tenant name) plus `key` (canonical spec name)

Note: most `basSystemIntegration.*`, `gateway.*`, `ias/fas/timelock/accessControl.*` keys are spec-ahead-of-firmware — devices don't post them yet. The notebook's auto-discovery will pick them up automatically once firmware rolls out; only scoring logic needs updating then.

The harvest notebook (`TB_Full_Harvest_v11.ipynb`) now includes **Group J (`J:newspec`)** for these keys: it fetches the `rock` and `basSystemIntegration` JSON payloads plus the flat `heartbeat` key, parses them into daily derived columns (`rock_health`, `rock_unhealthy`, `rock_nvr_offline`, `rock_cam_count`, `rock_sd_na_count`, `bas_hb`, `bas_panel_state`, `bas_main_status`, `bas_battery_status`, `bas_zone_triggered`, `heartbeat_flat_offline`), scores them in the fault model (skip-when-missing, so dormant keys never create false alarms), and exports them in the JSONL feature lists.

### Nightly Current-State Snapshots

Pull the current `active`, `lastDisconnectTime`, and `lastConnectTime` state for every device:

```bash
python pull_current_state_snapshots.py
```

Outputs are written under `current_state_snapshots/`:

- `current_state_YYYYMMDD.csv`: one row per device for that nightly snapshot.
- `offline_recoveries.csv`: derived outage rows where a later snapshot shows the device returned online.
- `nightly_current_state.log`: batch-run log when using `run_current_state_snapshot.bat`.

Register only the current-state pull in Windows Task Scheduler:

```powershell
.\schedule_current_state_snapshot.ps1 -RunTime 02:00
```

The existing `automate_harvest.bat` also calls `run_current_state_snapshot.bat` before staging and committing outputs, so an existing nightly harvest schedule will now collect these snapshots too.

### View the Dashboard

Open the generated HTML file:

```bash
open tb_dashboard_v7.html
```
