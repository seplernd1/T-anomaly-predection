# Seple Device Audit & Analytics

A Python/Jupyter-based system for scraping ThingsBoard devices, harvesting telemetry, generating reports, and creating interactive dashboards.

## Project Structure

- **`TB_Device_Audit_v7.ipynb` / `TB_Device_Audit_v6.ipynb`**: Core notebook for device discovery and data collection.
- **`TB_Full_Harvest_v8.ipynb`**: Comprehensive notebook for full data extraction and state processing.
- **`run_nightly_audit.py`**: Automation script to run the audit daily.
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

Open `TB_Device_Audit_v7.ipynb` or `TB_Full_Harvest_v8.ipynb` in Jupyter/VS Code and run the cells.

### Automated Daily Execution

The `run_nightly_audit.py` script automates this process:

```bash
python run_nightly_audit.py
```

This will:

1. Execute the notebook.
2. Generate an output report in the `audit_reports/` folder.
3. Send a notification to Slack and Email (if configured).

### View the Dashboard

Open the generated HTML file:

```bash
open tb_dashboard_v7.html
```
