"""Plan A: robustness pass on the GBM outage predictor.

5-fold GroupKFold (device-pure folds) + 3-fold expanding-window time CV over
the labelled era (anchor <= 2026-04-30, measured rows only). Frozen baseline
hyperparams — this measures variance, not tuning. OOF probabilities are
pooled for threshold analysis and an ops operating-point card.

Outputs -> training_data/cv_v2/:
  cv_metrics.json          per-fold + pooled metrics
  oof_predictions.parquet  per-row out-of-fold probabilities (group CV)
  ops_card.json            threshold sweep + chosen operating points
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
FEATS = ROOT / 'training_data/anomaly_20261005_041300_v2/anomaly_features.parquet'
LABELS = ROOT / 'training_data/supervised_outage_60m.parquet'
OUT = ROOT / 'training_data/cv_v2'
ERA_END = pd.Timestamp('2026-04-30', tz='UTC')

META = {'device_id', 'anchor_ts', 'split', 'group_key', 'censor_reason',
        'label_status', 'y_outage_24h', 'y', 'max_source_ts', 'n_rows_used',
        'n_rows_ignored_future', 'n_feature_values'}


def make_clf():
    # Frozen at the banked baseline configuration.
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0,
        class_weight='balanced', random_state=7)


def fold_metrics(y, p):
    out = {'n': int(len(y)), 'pos': int(y.sum())}
    if y.sum() == 0:
        return out
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    bi = int(np.argmax(f1s))
    out.update({
        'ap': float(average_precision_score(y, p)),
        'auc': float(roc_auc_score(y, p)),
        'f1_best': float(f1s[bi]),
        'best_thr': float(thr[min(bi, len(thr) - 1)]),
        'precision_best': float(prec[bi]),
        'recall_best': float(rec[bi]),
    })
    return out


def fit_fold(tr, te, feat):
    # HGB binning crashes on columns constant within THIS fold's train slice.
    nu = tr[feat].nunique(dropna=True)
    keep = [c for c in feat if nu.get(c, 0) > 1]
    clf = make_clf()
    clf.fit(tr[keep].to_numpy(dtype=np.float64), tr['y'].to_numpy())
    p = clf.predict_proba(te[keep].to_numpy(dtype=np.float64))[:, 1]
    return p, fold_metrics(te['y'].to_numpy(), p)


def agg(folds, keys=('ap', 'auc', 'f1_best')):
    stats = {}
    for k in keys:
        vals = [f[k] for f in folds if k in f and not np.isnan(f[k])]
        if vals:
            stats[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return stats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=str(FEATS))
    ap.add_argument('--labels', default=str(LABELS))
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--era-end', default='2026-04-30',
                    help="End date for the evaluated era (default: 2026-04-30, use 'none' or 'full' to disable).")
    a = ap.parse_args()
    feats_path, labels_path, out = Path(a.features), Path(a.labels), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fr = pd.read_parquet(feats_path)
    lb = pd.read_parquet(labels_path)
    lb['anchor_ts'] = pd.to_datetime(lb['anchor_ts'], utc=True)
    fr['anchor_ts'] = pd.to_datetime(fr['anchor_ts'], utc=True)
    m = fr.merge(lb[['device_id', 'anchor_ts', 'y_outage_24h', 'label_status']],
                 on=['device_id', 'anchor_ts'], how='inner')
    m = m[m['label_status'].isin(['measured', 'measured_negative'])]
    if a.era_end and a.era_end.lower() not in ('none', 'full', 'all'):
        era_end_ts = pd.Timestamp(a.era_end, tz='UTC')
        m = m[m['anchor_ts'] <= era_end_ts].reset_index(drop=True)
    else:
        m = m.reset_index(drop=True)
    m['y'] = (m['y_outage_24h'] == 1).astype(int)
    feat = [c for c in m.columns if c not in META]
    nuniq = m[feat].nunique(dropna=True)
    const = nuniq[nuniq <= 1].index.tolist()
    feat = [c for c in feat if c not in const]
    print(f'era rows={len(m)} pos={int(m.y.sum())} devices={m.device_id.nunique()} '
          f'features={len(feat)} (dropped {len(const)} const)')

    # ── primary: device-pure GroupKFold ─────────────────────────────────────
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(m), np.nan)
    group_folds = []
    for k, (tri, tei) in enumerate(gkf.split(m, groups=m['device_id'])):
        p, st = fit_fold(m.iloc[tri], m.iloc[tei], feat)
        oof[tei] = p
        st['fold'] = k
        group_folds.append(st)
        print(f'groupfold {k}: {st}')

    # ── secondary: expanding-window time CV (24h purge gap) ────────────────
    ms = m.sort_values('anchor_ts').reset_index(drop=True)
    t = ms['anchor_ts']
    cuts = [t.quantile(0.40), t.quantile(0.60), t.quantile(0.80)]
    gap = pd.Timedelta(hours=24)
    time_folds = []
    time_pred_rows = []
    for i, c in enumerate(cuts):
        hi = cuts[i + 1] if i + 1 < len(cuts) else None
        tr = ms[ms['anchor_ts'] < c]
        mask = ms['anchor_ts'] >= (c + gap)
        if hi is not None:
            mask &= ms['anchor_ts'] < hi
        te = ms[mask]
        if int(tr['y'].sum()) == 0 or int(te['y'].sum()) == 0:
            print(f'time fold {i}: skipped (no positives in a side)')
            continue
        p, st = fit_fold(tr, te, feat)
        st['fold'] = i
        time_folds.append(st)
        blk = te[['device_id', 'anchor_ts', 'y']].copy()
        blk['prob'] = p
        blk['time_fold'] = i
        time_pred_rows.append(blk)
        print(f'timefold {i}: {st}')

    # ── pooled OOF threshold analysis + ops card ──────────────────────────
    ok = ~np.isnan(oof)
    yv = m['y'].to_numpy()[ok]
    pv = oof[ok]
    days = float((m['anchor_ts'].max() - m['anchor_ts'].min()).total_seconds() / 86400)
    prec, rec, thr = precision_recall_curve(yv, pv)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    sweep = []
    for t_ in np.arange(0.05, 1.00, 0.05):
        flag = pv >= t_
        if flag.sum() == 0:
            continue
        tp = int((flag & (yv == 1)).sum())
        fp = int((flag & (yv == 0)).sum())
        fn = int((~flag & (yv == 1)).sum())
        sweep.append({
            'threshold': float(t_), 'flagged': int(flag.sum()),
            'alerts_per_day': float(flag.sum() / days),
            'precision': float(tp / max(tp + fp, 1)),
            'recall': float(tp / max(tp + fn, 1)),
            'f1': float(f1_score(yv, flag.astype(int), zero_division=0)),
        })
    # recall-first: max F1 subject to recall >= 0.5
    rf = [r for r in sweep if r['recall'] >= 0.5]
    recall_first = max(rf, key=lambda r: r['f1']) if rf else max(sweep, key=lambda r: r['f1'])
    # precision-first: max F1 subject to <= 2 alerts/day fleet-wide
    pf = [r for r in sweep if r['alerts_per_day'] <= 2.0]
    precision_first = max(pf, key=lambda r: r['f1']) if pf else None

    oof_frame = m.loc[ok, ['device_id', 'anchor_ts', 'y']].copy()
    oof_frame['oof_prob'] = pv
    oof_frame.to_parquet(out / 'oof_predictions.parquet', index=False)
    if time_pred_rows:
        pd.concat(time_pred_rows, ignore_index=True).to_parquet(out / 'time_cv_predictions.parquet', index=False)

    metrics = {
        'era': {'rows': int(len(m)), 'positives': int(m['y'].sum()),
                'devices': int(m['device_id'].nunique()), 'features': len(feat),
                'anchor_min': str(m['anchor_ts'].min()), 'anchor_max': str(m['anchor_ts'].max())},
        'baseline_single_split': {'test_f1': 0.1176, 'test_ap': 0.0890, 'test_auc': 0.7117},
        'group_cv': {'folds': group_folds, 'aggregate': agg(group_folds)},
        'time_cv': {'folds': time_folds, 'aggregate': agg(time_folds)},
        'pooled_oof': {'ap': float(average_precision_score(yv, pv)),
                       'auc': float(roc_auc_score(yv, pv)),
                       'recall_first': recall_first,
                       'precision_first': precision_first},
    }
    json.dump(metrics, open(out / 'cv_metrics.json', 'w'), indent=1)
    json.dump({'sweep': sweep, 'recall_first': recall_first,
               'precision_first': precision_first,
               'note': 'alerts_per_day over the labelled-era fleet; '
                       'recall-first = max F1 with recall>=0.5; '
                       'precision-first = max F1 with <=2 alerts/day'},
              open(out / 'ops_card.json', 'w'), indent=1)
    print('group CV:', json.dumps(agg(group_folds)))
    print('time  CV:', json.dumps(agg(time_folds)))
    print('pooled OOF AP/AUC:', metrics['pooled_oof']['ap'], metrics['pooled_oof']['auc'])
    print('recall-first point:', recall_first)
    print('precision-first point:', precision_first)
    print('saved ->', out)


if __name__ == '__main__':
    main()
