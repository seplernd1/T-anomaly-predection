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
