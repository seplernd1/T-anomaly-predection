# run_nightly_audit.py
import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
NOTEBOOK_PATH = "TB_Device_Audit_v7.ipynb"
OUTPUT_DIR    = Path("audit_reports")
LOG_FILE      = OUTPUT_DIR / "audit_scheduler.log"
TIMESTAMP     = datetime.now().strftime("%Y%m%d_%H%M")

# ── Setup ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def run_notebook():
    logging.info(f"🚀 Starting nightly audit run: {TIMESTAMP}")
    try:
        # Execute notebook headlessly using papermill
        cmd = [
            sys.executable, "-m", "papermill",
            NOTEBOOK_PATH,
            str(OUTPUT_DIR / f"TB_Audit_Executed_{TIMESTAMP}.ipynb"),
            "--log-output",
            "--execution-timeout", "3600"  # 1 hour max
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info("✅ Notebook executed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Notebook failed:\n{e.stderr}")
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected error: {str(e)}")
        return False

def send_summary(success):
    import requests
    import smtplib
    import glob
    from email.message import EmailMessage

    status = "✅ SUCCESS" if success else "❌ FAILED"
    msg = f"[ThingsBoard Audit] Nightly run {TIMESTAMP}: {status}"
    logging.info(msg)
    print(msg)

    # Slack Webhook (Recommended)
    SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")  # Set in Windows env vars or .env
    if SLACK_WEBHOOK:
        try:
            # Tagging Animesh using their Slack Member ID
            payload = {"text": f"<@U0B12Q7TYSW>\n{msg}"}
            requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
            logging.info("✅ Slack notification sent")
        except Exception as e:
            logging.warning(f"Slack notify failed: {e}")

    # Email (SMTP)
    if os.getenv("SMTP_PASSWORD"):
        try:
            emsg = EmailMessage()
            emsg.set_content(msg + "\n\nPlease find the attached report.")
            emsg["Subject"] = f"TB Audit {TIMESTAMP}"
            emsg["From"] = os.getenv("SMTP_EMAIL")
            emsg["To"] = "animesh.choudhury@seple.in"
            
            # Attach the latest excel report
            list_of_files = glob.glob("tb_device_audit_*.xlsx")
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                with open(latest_file, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(latest_file)
                emsg.add_attachment(
                    file_data, 
                    maintype='application', 
                    subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                    filename=file_name
                )
            
            with smtplib.SMTP("smtp.gmail.com", 587) as s:
                s.starttls()
                s.login(os.getenv("SMTP_EMAIL"), os.getenv("SMTP_PASSWORD"))
                s.send_message(emsg)
            logging.info("✅ Email notification sent with attachment")
        except Exception as e:
            logging.warning(f"Email notify failed: {e}")

if __name__ == "__main__":
    # Load .env if present (for local testing)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    success = run_notebook()
    send_summary(success)
    sys.exit(0 if success else 1)