# Request: Alarm generation review + lifecycle-event retention on the SEPLE IoT tenant

**From:** ML/data engineering
**Date:** 2026-09-21
**Priority:** High — blocks verified outage labeling for the 24-hour outage-prediction model
**Tenant:** seple.iot-private.cloud (bank ATM-site security monitoring fleet)

---

## Summary

Two ThingsBoard-side capabilities are missing or have silently stopped, and they are the
only remaining blockers for training a fleet-wide outage-prediction model:

1. **Alarm generation appears to have stopped tenant-wide on 2026-03-30.**
2. **Device lifecycle events (connect/disconnect) are not retained (or not persisted at all).**

Everything else we need is already working: full device registry access, telemetry
history, attributes, and alarm *history up to* the March boundary.

## What we verified (evidence, not suspicion)

All checks were read-only API queries against the production tenant.

### Finding 1 — alarm history ends 2026-03-30, tenant-wide

- Alarm query for a known-active device (`BOB-AIRPORT`, type `HESTIA`):
  - 2025-09-21 → 2026-03-30: **8,944 alarms** (CAMERA TAMPER/DISCONNECT, IAS, DVR/NVR OFF, POWER OFF, BATTERY LOW).
  - 2026-04-01 → 2026-07-01: `totalElements = 0`.
  - 2026-07-01 → 2026-09-21: `totalElements = 0`.
- The same device kept posting telemetry after 2026-03-30 (it was alive).
- Two other devices show identical behavior (history stops at the same boundary).
- The oldest returned alarm sits at exactly `today − 365 days`, which also indicates a
  365-day alarm retention window is in effect.

**Interpretation:** alarms stopped being *generated* (or delivered to the alarm subsystem)
around 2026-03-30 — this is not a query or permission problem on our side. Possible causes
on the platform side: a rule-chain change, a device-profile alarm-rule migration, an
upgrade on ~2026-03-30, or an alarm-retention/cleanup job over-running.

**Question for Seple:** did anything change in the rule chains, device profiles, or
platform version around 2026-03-30? Can alarm generation be restored/re-enabled?

### Finding 2 — device events return zero rows for every type and window

Using the events API (`/api/events/DEVICE/{deviceId}?tenantId=...&eventType=...`) — with
`tenantId` supplied as the query parameter this TB build requires — we queried:

- event types `LC_EVENT`, `ERROR`, `STATS`
- windows: March 2026 (when alarms demonstrably existed) and recent weeks
- result: **HTTP 200 with zero rows in every combination, including unfiltered queries**

**Interpretation:** event persistence is disabled or purged on this deployment
(Telemetry/Event storage configuration). We are not asking for event backfill — we
understand that history cannot be recovered.

## What we are requesting

| # | Request | Why |
|---|---------|-----|
| 1 | Investigate/restore **alarm generation** (rule chains / device profiles) | Alarms with `clearTs` are our primary verified outage evidence |
| 2 | Enable persistence for **LC_EVENT (lifecycle) events** with at least **400-day retention** | Independent disconnect/reconnect evidence for outage labels |
| 3 | Confirm/extend **alarm retention** beyond 365 days (≥ 400 days preferred) | Training windows must predate any outage label horizon |
| 4 | If event persistence cannot be enabled, an alternative: a **scheduled alarm/device-state export** (CSV/API dump) we can collect nightly | We already run scheduled pull jobs and can consume either format |
| 5 | If available, expose **device availability/ingestion timestamps** (server receive time) | Distinguishes "device was quiet" from "data arrived late" — required for strictly leakage-free labels |

Items 1–3 are standard ThingsBoard configuration (device-profile alarm rules,
`tenant_profile` / configuration for event TTL). No custom development should be required
on your side beyond configuration and verification.

## Why this matters to the project

The goal of this workstream is a model that predicts whether a site will go offline in the
next 24 hours, with **verified** labels (disconnect→reconnect pairs, or offline alarms with
clear timestamps). Current honest status:

- **Anomaly detection from telemetry: GO** — no upstream changes needed.
- **24-hour outage prediction: NO-GO fleet-wide** — verified evidence exists for only
  2 of 167 devices and ends 2026-03-30.

With items 1–3 in place, labels can be produced for the whole fleet going forward, and
the model becomes trainable within one retention cycle.

## Our commitments

- All requests are read-only against production; nothing in this change set writes to
  the tenant.
- We rotate credentials and never store tenant data outside the approved
  access-restricted storage.
- We can validate restored alarm generation within minutes of the change using our
  existing probes (happy to run them together on a call).

## Contacts

ML/data engineering — via this repository's maintainers.
