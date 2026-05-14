# PLANNING — Dexter HMS Telemetry → ML Fault Classifier

**Owner:** rnd@seple.in
**Repo:** Data_scrapping_thgingsboard_ml-intern
**Pipeline status as of 2026-05-14:** v11 notebook running clean end-to-end on this tenant; data exports validated.

This doc lays out everything that remains to ship a useful, maintainable
fault-prediction pipeline. Each phase is sized to a realistic chunk of work
and has a concrete "done" condition.

---

## Where we are right now

**Working today:**
- `TB_Full_Harvest_v11.ipynb` runs end-to-end on 184 devices, 365 days, 528 telemetry keys (54 spec/legacy + 474 auto-discovered).
- Auto-discovery cell pulls fresh keys from TB on every run — no code edit needed when devices start posting new keys (unless they need scoring/parsing logic).
- Daily-snapshot scoring works on the keys that actually have data on this tenant (legacy CamelCase + discovered).
- Outputs:
  - `ts_daily_snapshots_<ts>.csv` — full 67 k device-day grid
  - `ts_summary_<ts>.xlsx` — Device Trend / Bank×Date Heat / Top Risk Days
  - `ml_training_timeseries.jsonl` — per-device-day training examples
  - `ml_training_combined.jsonl` — v11 snapshot examples + time-series examples
  - `tb_audit_v11_<ts>.xlsx` — point-in-time device audit

**Not working / stale:**
- `README.md` still references v6 / v7 notebooks.
- `run_nightly_audit.py` still points at `TB_Device_Audit_v7.ipynb` (broken if scheduled).
- 293 spec telemetry keys come back 0 % — tenant doesn't post them yet. Not a bug; future spec migration.
- Discovery has a ~32 % per-device error rate (60 / 184 devices) — usually JWT-related. Tolerable but worth retrying.

---

## Phase 0 — Validate the data we already have *(half a day)*

Don't train a model on data you haven't sanity-checked. Before any ML work:

1. **Open `ts_summary_<ts>.xlsx` → Top Risk Days sheet.** Pick the 5 highest-scoring CRITICAL device-days. For each, ask the ops team / cross-check the audit xlsx: *was this branch actually broken on that date?* If 3+ out of 5 are confirmed real faults, scoring is calibrated well enough to train on. If most are false positives, fix scoring (Cell 18) before training.
2. **Open `ts_daily_snapshots_<ts>.csv` → filter one known-bad device for the last 30 days.** Confirm `ts_severity` tracks reality day-by-day, not just static snapshot.
3. **Compare `Device Trend` sheet** — top-20 worst devices by `avg_fault_score`. Are these the same devices ops is already complaining about? Same → trust signal. Different → either we found something they missed, or our scoring is off.
4. **Spot-check 5 random JSONL lines in `ml_training_timeseries.jsonl`** — make sure `input:` tokens look meaningful (not all zeros, not a wall of identical strings) and `output:` severity matches the input's apparent state.

**Done when:** you can point at the data and say "yes, these CRITICAL days are real faults" with reasonable confidence.

---

## Phase 1 — Train the first model *(1–2 days)*

The harvest already produces the training file the `ml-intern` tooling expects.

1. **Train baseline:**
   ```
   ml-intern run \
     --data    ml_training_combined.jsonl \
     --base-model Qwen3-1.7B \
     --task    classification \
     --metric  weighted-f1 \
     --priority recall:CRITICAL
   ```
2. **Hold-out split:** do an 80/10/10 train/val/test split *by device, not by row* (or you'll leak — same device's earlier days predict its later days trivially).
3. **Inspect confusion matrix.** The key question: what % of true CRITICAL device-days do we catch? Recall on CRITICAL is the operations-critical metric — a false negative is a missed outage; a false positive is a wasted check.
4. **Establish baselines to beat:**
   - Trivial baseline: "always predict HEALTHY" — measures what % CRITICAL prevalence is.
   - Rule baseline: just our `ts_fault_score >= 70` threshold — measures whether the model adds anything over hand-tuned rules.
5. **Save the trained model + metrics report** under `models/<date>_<commit-sha>/`.

**Done when:** you have a model whose CRITICAL recall is ≥ rule baseline + 10 % AND whose FP rate is low enough that ops will actually look at its alerts.

**Decision point:** if the model can't beat the rule baseline, the problem is feature engineering, not the model. Go back to Cell 18 (scoring) and Cell 19 (JSONL features) before throwing bigger models at it.

---

## Phase 2 — Threshold tuning + actionability *(1 day)*

A classifier output of "CRITICAL" is useless unless ops know what to do.

1. **Cost-aware threshold:** plot precision/recall vs. score threshold; pick the threshold that matches the ops team's capacity (e.g. "we can investigate ~30 alerts/day → pick the threshold where we get ~30 CRITICAL/day").
2. **Reason-explanation:** every prediction must come with `ts_top_reasons` (already wired up in Cell 18). When alerting, surface the top 3 reasons so the ops team knows *why* the device flagged.
3. **Per-branch escalation:** branches with `sla_tier == 'Gold'` (or whatever the contract field is in `device_df`) → alert immediately. Bronze → batch into daily digest.

**Done when:** there's a written runbook saying "if a device hits CRITICAL with reason X, do Y".

---

## Phase 3 — Productionize the nightly run *(1 day)*

You already have `automate_harvest.bat` doing harvest + git commit + push. The missing pieces:

1. **Fix `run_nightly_audit.py`** — change `NOTEBOOK_PATH = "TB_Device_Audit_v7.ipynb"` to `TB_Full_Harvest_v11.ipynb`. Currently it points at a non-existent file.
2. **Update `README.md`** — replace all v6/v7/v8 references with v11.
3. **Set up Windows Task Scheduler** (or pick another scheduler) to run `automate_harvest.bat` nightly at e.g. 02:00 IST. Verify it actually fires.
4. **Wire Slack / email** — `run_nightly_audit.py` already has hooks; needs `SLACK_WEBHOOK` and `SMTP_PASSWORD` in env vars. After the nightly run, post a message: *"N devices CRITICAL today (was M yesterday). Top 5 branches: …"*
5. **Disk hygiene:** the harvest writes 5 timestamped xlsx + 2 jsonl files per run. After 30 days that's a lot. Add a "keep last 7 daily + last 12 monthly" rotation in `audit_reports/`.

**Done when:** nightly runs for 5 consecutive days without manual intervention and the team gets the Slack alert each morning.

---

## Phase 4 — Schema-drift early-warning *(½ day)*

Auto-discovery already catches new keys silently. But:

1. **Persist a `discovered_keys_history.json`** that records the set of keys seen on each run. Diff each new run against the previous one. When a brand-new key appears across ≥ 10 devices, post a Slack message: *"New telemetry key detected: `xyz` (12 devices). Consider adding to scoring."*
2. **Persist a `key_coverage_history.csv`** — date, key, coverage_pct. When a previously-active key drops to 0 % over 3+ days, that's a regression (firmware push removed it, agent broke, etc.) — alert.

**Done when:** you've forgotten how Cells 15–19 work because the pipeline tells you when something changes.

---

## Phase 5 — Stakeholder dashboard *(2–3 days)*

Right now insights live in xlsx + Slack alerts. For business-facing visibility:

1. **Pick the tool:** Streamlit (fast, Python-native, deploy as a service) or Looker Studio (BI-friendly, free, needs the data in BigQuery). Streamlit is the lower-friction choice given the team's stack.
2. **Three views minimum:**
   - **Fleet health overview** — KPIs (total CRITICAL today, 7-day trend, top-5 bad banks).
   - **Branch drill-down** — pick a branch → see 365-day score timeline, event log, last-known statuses.
   - **Device deep-dive** — pick a device → daily score + raw telemetry + which scoring reasons fired most.
3. **Data source:** read `ts_daily_snapshots_<ts>.csv` directly, or set up a small DuckDB / SQLite file under `data/` that the dashboard queries.

**Done when:** an ops lead can answer "which branches need attention this week" in < 30 seconds without asking you.

---

## Phase 6 — Feedback loop *(ongoing, starts week 4)*

The model will only get better if it learns from operational ground truth.

1. **Capture ops outcomes:** when an alert fires and the ops team investigates, record the verdict (true-fault / false-alarm / inconclusive) in a simple sheet or DB. Even 20 labels/week is enough to drive iteration.
2. **Re-train monthly** with the new labels merged in. Track CRITICAL recall and FP rate over time — both should trend right.
3. **Backlog the model lessons:** every false negative ("this device was broken and we didn't flag it") is a hypothesis for a new scoring rule or feature.

**Done when:** there's a Linear / GitHub Issues board for "model improvement ideas" and it's being worked through.

---

## Risks & open questions

| Risk | Mitigation |
|---|---|
| **Spec snake_case keys never get posted** (statusbox_*, heartbeat_*, etc.) — scoring stays reliant on legacy CamelCase | Acceptable for now; legacy keys have 30-50% coverage. Re-evaluate after device firmware update reaches > 50 % of fleet. |
| **JWT expiry mid-harvest** — 60 / 184 discovery errors last run | Add JWT auto-refresh in Cell 3 / Cell 16. Low priority since current errors are tolerable. |
| **CRITICAL day prevalence too low for ML** (if e.g. only 0.5 % of device-days are CRITICAL) | Use stratified sampling + class-weighted loss. If still too few, fall back to anomaly detection (one-class) instead of supervised classification. |
| **Bank-level data quality varies wildly** — some banks have great coverage, some have 0 % across the board | Score per-bank coverage; if a bank is below threshold, exclude or surface separately so it doesn't dilute model metrics. |
| **Schema drift** — Dexter releases v2 of the spec, fields rename | Auto-discovery catches new fields. Phase 4 alerts on disappearing fields. Migration is a one-cell edit. |
| **No PII or credentials leak via JSONL** — branch_name, device_id are present | Confirm with security: are device IDs OK to ship to a model training service? If not, hash them before JSONL write. |

---

## Cleanup the repo can use anytime *(½ day, can be parallel)*

- Delete `TB_Full_Harvest_v9.ipynb` and `TB_Full_Harvest_v10.ipynb` once v11 is production-stable (move to `archive/` first).
- Delete `dashboard_data.json`, `scratchpad.md`, `dump_cells.txt`, `list_cells.py` — all dev artefacts.
- Add `*.xlsx`, `*.jsonl`, `*.csv` to `.gitignore` (currently the synthetic-commit history shows these getting tracked).
- Run `simplify` on Cells 17 / 18 / 19 — there's some redundancy after three rounds of patches.

---

## Suggested order of execution

```
Week 1: Phase 0 (validate) + Phase 1 (train baseline)
Week 2: Phase 2 (thresholds + runbook) + Phase 3 (productionize)
Week 3: Phase 4 (drift alerts) + start Phase 5 (dashboard MVP)
Week 4+: Phase 5 polish + Phase 6 (feedback loop) ongoing
```

This is sequential because each phase depends on the previous. Phase 1 makes no
sense without validated data (Phase 0); Phase 3 makes no sense without a usable
model (Phase 1); Phase 6 makes no sense without alerts that ops are seeing
(Phase 3).

If only one thing gets done next, it should be **Phase 0**. If only two, add
**Phase 1**. If only three, **Phase 3** (the existing automation will silently
rot if not fixed).
