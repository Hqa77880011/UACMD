import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .model import Model, extra_metrics


class Arrays(Dataset):
    def __init__(self, data, role):
        self.indices = np.flatnonzero(data['split'] == role)
        self.data = {k: v for k, v in data.items() if v.ndim and len(v) == len(data['y']) and v.dtype.kind in 'biuf'}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return {k: torch.as_tensor(v[i]) for k, v in self.data.items()}


def prepare(c, out, source):
    raw = dict(np.load(source, allow_pickle=False))
    if not {'x', 'y', 'group', 'split'} <= set(raw):
        raise ValueError('NPZ requires x, y, group, split')
    n = len(raw['y'])
    if any(len(raw[k]) != n for k in ['x', 'group', 'split']):
        raise ValueError('Sample dimensions must agree')
    roles = raw['split'].astype(str)
    if set(roles) != {'train', 'val', 'test'}:
        raise ValueError('split must contain train, val, test')
    for group in np.unique(raw['group']):
        if len(np.unique(roles[raw['group'] == group])) != 1:
            raise ValueError('A group crosses data partitions: ' + str(group))
    x = raw['x'].astype(np.float32)
    fit = x[roles == 'train']
    axes = (0,)
    mean = np.nanmean(fit, axis=axes, keepdims=True)
    scale = np.nanstd(fit, axis=axes, keepdims=True).clip(1e-6)
    raw['x'] = np.nan_to_num((x - mean) / scale).astype(np.float32)
    raw['y'] = raw['y'].astype(np.int64)
    if raw['y'].min() < 0 or raw['y'].max() >= c['classes']:
        raise ValueError('Labels must be zero based and below classes')
    raw['split'] = roles
    np.savez_compressed(out / 'data.npz', **raw)
    np.savez(out / 'normalization.npz', mean=mean, scale=scale)


def build(c, data):
    return Model(c, data['x'].shape[1:])


def train(c, out):
    torch.manual_seed(c['seed'])
    np.random.seed(c['seed'])
    data = dict(np.load(out / 'data.npz', allow_pickle=False))
    model = build(c, data).to(c['device'])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    train_loader = DataLoader(Arrays(data, 'train'), batch_size=c['batch_size'], shuffle=True)
    val_loader = DataLoader(Arrays(data, 'val'), batch_size=c['batch_size'])
    best, stale, curve = -np.inf, 0, []
    for epoch in range(c['epochs']):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(c['device']) for k, v in batch.items()}
            loss = model.loss(batch, c, epoch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        labels, predictions = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(c['device']) for k, v in batch.items()}
                logits = model(batch)['logits']
                labels.extend(batch['y'].cpu().numpy())
                predictions.extend(logits.argmax(-1).cpu().numpy())
        score = f1_score(labels, predictions, average='macro', zero_division=0)
        curve.append(dict(epoch=epoch, loss=np.mean(losses), validation_macro_f1=score))
        if epoch < max(c.get('source_epochs', 0), c.get('teacher_epochs', 0)):
            curve[-1]['validation_macro_f1'] = score
            continue
        if score > best:
            best, stale = score, 0
            torch.save(model.state_dict(), out / 'model.pt')
        else:
            stale += 1
        if stale >= c['patience']:
            break
    pd.DataFrame(curve).to_csv(out / 'learning.csv', index=False)


def run(c, out):
    data = dict(np.load(out / 'data.npz', allow_pickle=False))
    model = build(c, data).to(c['device'])
    model.load_state_dict(torch.load(out / 'model.pt', map_location=c['device'], weights_only=True))
    model.teacher = torch.nn.Identity()
    model.eval()
    collected, timings = {}, []
    with torch.no_grad():
        for batch in DataLoader(Arrays(data, 'test'), batch_size=c['batch_size']):
            batch = {k: v.to(c['device']) for k, v in batch.items()}
            if c['device'].startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model(batch)
            if c['device'].startswith('cuda'):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000 / len(batch['y']))
            result['probability'] = result.pop('logits').softmax(-1)
            for key, value in result.items():
                if isinstance(value, torch.Tensor) and value.ndim and len(value) == len(batch['y']):
                    collected.setdefault(key, []).append(value.cpu().numpy())
    saved = {k: np.concatenate(v) for k, v in collected.items()}
    selected = data['split'] == 'test'
    for key, value in data.items():
        if value.ndim and len(value) == len(selected) and key != 'x':
            saved['target_' + key] = value[selected]
    saved['latency_ms'] = np.array(timings)
    saved['parameters'] = np.array(sum(p.numel() for p in model.parameters()))
    np.savez_compressed(out / 'predictions.npz', **saved)


def evaluate(c, out):
    d = dict(np.load(out / 'predictions.npz', allow_pickle=False))
    y, p = d['target_y'], d['probability']
    predicted = p.argmax(1)
    correct = (predicted == y).astype(float)
    confidence = p.max(1)
    ece = 0.
    for ids in np.array_split(np.argsort(confidence), min(15, len(y))):
        ece += len(ids) / len(y) * abs(correct[ids].mean() - confidence[ids].mean())
    order = np.argsort(-confidence)
    risk = np.cumsum(1 - correct[order]) / np.arange(1, len(y) + 1)
    rows = dict(method=c['variant'], accuracy=accuracy_score(y, predicted),
                balanced_accuracy=balanced_accuracy_score(y, predicted), macro_f1=f1_score(y, predicted, average='macro', zero_division=0),
                weighted_f1=f1_score(y, predicted, average='weighted', zero_division=0),
                nll=log_loss(y, p, labels=np.arange(c['classes'])), ece=ece, aurc=risk.mean(),
                latency_ms=float(d['latency_ms'].mean()), parameters=int(d['parameters']))
    if c['classes'] == 2 and len(np.unique(y)) == 2:
        rows['auroc'] = roc_auc_score(y, p[:, 1])
    rows.update(extra_metrics(d, c))
    pd.DataFrame([rows]).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(dict(coverage=np.arange(1, len(y) + 1) / len(y), risk=risk)).to_csv(out / 'risk_coverage.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'risk_coverage.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d.coverage, d.risk)
    ax.set(xlabel='Coverage', ylabel='Selective classification risk')
    fig.tight_layout()
    fig.savefig(out / 'risk_coverage.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train', 'run', 'evaluate', 'plot'])
    parser.add_argument('--input')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output', default='runs/default')
    parser.add_argument('--variant')
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if args.input is None:
            parser.error('--input is required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
