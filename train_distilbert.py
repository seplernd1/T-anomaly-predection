"""DistilBERT fine-tune on serialized anomaly rows -> outage within 24h.
Usage (tomorrow): .\\.venv\\Scripts\\python.exe -X utf8 train_distilbert.py
Needs HF_TOKEN env only for the final Hub push (skipped otherwise).
Top-60 GBM features -> text; same time-ordered splits; F1 on outage class."""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(r'C:\workspace\Data_scrapping_thgingsboard_ml-intern')
FEATS = ROOT / 'training_data/anomaly_20261005_041300_v2/anomaly_features.parquet'
LABELS = ROOT / 'training_data/supervised_outage_60m.parquet'
BASELINE = ROOT / 'training_data/baseline_60m/metrics.json'
OUT = ROOT / 'training_data/distilbert_outage_v1'

TOP_N = 60
MODEL_ID = 'distilbert-base-uncased'
MAX_LEN = 256
EPOCHS = 4
LR = 3e-5
BATCH = 8
ACCUM = 4  # effective batch 32 on 6GB VRAM


def main():
    top = list(json.load(open(BASELINE))['top_features'].keys())[:TOP_N]
    print('serializing top', len(top), 'features')
    fr = pd.read_parquet(FEATS, columns=['device_id', 'anchor_ts', 'split'] + top)
    lb = pd.read_parquet(LABELS)
    lb['anchor_ts'] = pd.to_datetime(lb['anchor_ts'], utc=True)
    fr['anchor_ts'] = pd.to_datetime(fr['anchor_ts'], utc=True)
    m = fr.merge(lb[['device_id', 'anchor_ts', 'y_outage_24h', 'label_status']],
                 on=['device_id', 'anchor_ts'], how='inner')
    m = m[m['anchor_ts'] <= pd.Timestamp('2026-04-30', tz='UTC')]
    m = m[m['label_status'].isin(['measured', 'measured_negative'])].copy()
    m['y'] = (m['y_outage_24h'] == 1).astype(int)

    def ser(row):
        bits = []
        for c in top:
            v = row[c]
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(f):
                continue
            bits.append(f'{c} {f:.3f}')
        return '; '.join(bits)

    m['text'] = [ser(r) for _, r in m[top].iterrows()]
    m = m[m['text'].str.len() > 0].reset_index(drop=True)
    # time-ordered 70/15/15 + 192h purge (mirrors baseline script)
    m = m.sort_values('anchor_ts').reset_index(drop=True)
    n = len(m)
    c1, c2 = int(n * 0.7), int(n * 0.85)
    m['split2'] = 'train'
    m.loc[c1:c2 - 1, 'split2'] = 'validation'
    m.loc[c2:, 'split2'] = 'test'
    b1 = m.loc[c1 - 1, 'anchor_ts'] + (m.loc[c1, 'anchor_ts'] - m.loc[c1 - 1, 'anchor_ts']) / 2
    b2 = m.loc[c2 - 1, 'anchor_ts'] + (m.loc[c2, 'anchor_ts'] - m.loc[c2 - 1, 'anchor_ts']) / 2
    pg = pd.Timedelta(hours=192)
    purge = ((m['anchor_ts'] > b1 - pg / 2) & (m['anchor_ts'] < b1 + pg / 2)) | \
            ((m['anchor_ts'] > b2 - pg / 2) & (m['anchor_ts'] < b2 + pg / 2))
    m = m[~purge].reset_index(drop=True)
    print(m['split2'].value_counts().to_dict(), 'pos:', m.groupby('split2')['y'].sum().to_dict())

    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments, DataCollatorWithPadding,
                              EarlyStoppingCallback)
    from sklearn.metrics import f1_score, average_precision_score
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = {}
    for sp in ('train', 'validation', 'test'):
        d = m[m['split2'] == sp][['text', 'y']].rename(columns={'y': 'label'})
        h = Dataset.from_pandas(d, preserve_index=False)
        h = h.map(lambda b: tok(b['text'], truncation=True, max_length=MAX_LEN),
                  batched=True, remove_columns=['text'])
        ds[sp] = h
    print({k: len(v) for k, v in ds.items()})

    n_pos = int((m[m['split2'] == 'train']['y'] == 1).sum())
    n_neg = int((m[m['split2'] == 'train']['y'] == 0).sum())
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    # mild positive weighting via focal-style pos_weight in custom trainer
    from torch import nn
    from transformers import Trainer as _T

    class WeightedTrainer(_T):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop('labels')
            out = model(**inputs)
            w = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=out.logits.device)
            loss = nn.functional.cross_entropy(out.logits, labels, weight=w)
            return (loss, out) if return_outputs else loss

    def f1_pos(p):
        pred = np.argmax(p.predictions, axis=1)
        return {'f1_outage': f1_score(p.label_ids, pred, pos_label=1, zero_division=0),
                'ap': average_precision_score(p.label_ids,
                      torch.softmax(torch.tensor(p.predictions), -1)[:, 1].numpy())}

    args = TrainingArguments(
        output_dir=str(OUT), num_train_epochs=EPOCHS, learning_rate=LR,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=16,
        gradient_accumulation_steps=ACCUM, fp16=torch.cuda.is_available(),
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='f1_outage', greater_is_better=True,
        logging_steps=50, report_to='none', seed=7)
    tr = WeightedTrainer(model=model, args=args, train_dataset=ds['train'],
                         eval_dataset=ds['validation'], processing_class=tok,
                         data_collator=DataCollatorWithPadding(tok),
                         compute_metrics=f1_pos,
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    tr.train()
    res = tr.evaluate(ds['test'])
    print('TEST:', res)
    json.dump({'test': res, 'top_n': TOP_N, 'model': MODEL_ID},
              open(OUT / 'metrics.json', 'w'), indent=1)
    tr.save_model(str(OUT / 'final'))
    tok.save_pretrained(str(OUT / 'final'))
    print('saved', OUT)


if __name__ == '__main__':
    main()
