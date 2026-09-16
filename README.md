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

`pull_all_data.py` is a single orchestrator that pulls every data type from the tenant into one timestamped run folder (`data_harvest/all_data_<ts>/`) with a `manifest.json` recording row counts and per-step errors:

```bash
python pull_all_data.py                          # full run: registry, relations, attributes,
                                                 # telemetry keys + series, events, alarms
python pull_all_data.py --telemetry-days 90      # limit telemetry lookback
python pull_all_data.py --plan-only              # show what would run, fetch nothing
python pull_all_data.py --max-devices 5 --max-keys-per-device 2   # smoke test
python pull_all_data.py --device-offset 30 --max-devices 3        # sample specific devices
python pull_all_data.py --all-asset-relations    # also walk standalone asset trees
```

Useful flags: `--skip-telemetry`, `--skip-events`, `--skip-alarms`, `--skip-relations`, `--key-filter "active|heartbeat"` (regex on telemetry keys).

Run folder contents:

- `devices.csv` / `customers.csv` / `assets.csv` — entity registry
- `relations.csv` — bank → branch → device graph (walked per device)
- `device_attributes.csv` — all scopes (SERVER/CLIENT/SHARED) incl. connectivity keys
- `device_telemetry_keys.csv` — discovered telemetry keys per device
- `telemetry/telemetry_<device_id>.csv` — timeseries per device (device_id, key, ts, ts_iso, value)
- `events_<type>.csv` (LC_EVENT, ERROR, STATS, DEBUG) and `alarms.csv` — outage ground truth
- `manifest.json` — per-step rows/files/errors/timing

### Key Spec (new telemetry key document)

The key-update document (`Untitled document (1).docx` / `.pdf`, same content) defines the new namespaced key spec (`gateway.*`, `cctv.*`/`rock.*`, `system_status.*`, `basSystemIntegration.*`, `timeLock.*`, `accessControl.*`). Tooling around it:

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
