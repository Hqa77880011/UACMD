import copy

import numpy as np
import torch
from scipy.stats import pearsonr
from torch import nn
from torch.nn import functional as F


class Fusion(nn.Module):
    def __init__(self, c, shape):
        super().__init__()
        modalities, feature_dim = shape
        self.c = c
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, c['width']), nn.ReLU(), nn.Linear(c['width'], c['width']), nn.ReLU()) for _ in range(modalities)])
        self.quality = nn.Linear(c['width'], 1)
        self.uncertainty = nn.Linear(c['width'], 1)
        self.classifier = nn.Sequential(nn.Linear(c['width'], c['width']), nn.ReLU(), nn.Linear(c['width'], c['classes']))
        self.regression = nn.Linear(c['width'], 1)

    def forward(self, x, available):
        features = torch.stack([encoder(x[:, i]) for i, encoder in enumerate(self.encoders)], 1)
        uncertainty = F.softplus(self.uncertainty(features).squeeze(-1)) + 1e-7
        quality = self.quality(features).squeeze(-1).sigmoid()
        scores = quality - self.c['uncertainty_scale'] * uncertainty
        if self.c['variant'] == 'uniform':
            scores = scores * 0
        if not available.any(1).all():
            raise ValueError('At least one modality is required')
        weights = (scores / self.c['temperature']).masked_fill(~available, -torch.inf).softmax(-1)
        fused = (weights[..., None] * features).sum(1)
        return dict(logits=self.classifier(fused), representation=fused, uncertainty=uncertainty,
                    weights=weights, regression=self.regression(fused).squeeze(-1))


class Model(nn.Module):
    def __init__(self, c, shape):
        super().__init__()
        self.c = c
        self.student = Fusion(c, shape)
        self.teacher = copy.deepcopy(self.student)
        if c['variant'] not in ['full', 'uniform', 'no-kd', 'no-feature', 'no-uncertainty', 'no-curriculum']:
            raise ValueError('Unknown variant')

    def forward(self, batch):
        x = batch['x'].float()
        available = batch.get('available', torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)).bool()
        return self.student(x, available)

    def task_loss(self, output, batch):
        loss = F.cross_entropy(output['logits'], batch['y'])
        if 'sentiment' in batch:
            loss = loss + self.c['regression_weight'] * F.l1_loss(output['regression'], batch['sentiment'].float())
        return loss

    def loss(self, batch, c, epoch):
        x = batch['x'].float()
        if epoch < c['teacher_epochs']:
            return self.task_loss(self.teacher(x, torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)), batch)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        probability = c['missing_rate']
        if c['variant'] != 'no-curriculum':
            probability *= min(1, (epoch - c['teacher_epochs'] + 1) / c['curriculum_epochs'])
        available = torch.rand(x.shape[:2], device=x.device) >= probability
        empty = ~available.any(1)
        available[empty, 0] = True
        corrupted = (torch.rand(x.shape[:2], device=x.device) < c['corruption_rate']) & available
        damaged = x + torch.randn_like(x) * c['noise_scale'] * corrupted[..., None]
        student = self.student(damaged, available)
        with torch.no_grad():
            teacher = self.teacher(x, torch.ones_like(available))
        loss = self.task_loss(student, batch)
        if c['variant'] != 'no-kd':
            temperature = c['distillation_temperature']
            loss += c['kd_weight'] * temperature ** 2 * F.kl_div((student['logits'] / temperature).log_softmax(-1),
                                                                              (teacher['logits'] / temperature).softmax(-1), reduction='batchmean')
        if c['variant'] != 'no-feature':
            loss += c['feature_weight'] * (F.normalize(student['representation'], dim=1) - F.normalize(teacher['representation'], dim=1)).square().sum(1).mean()
        if c['variant'] != 'no-uncertainty':
            u = student['uncertainty']
            penalty = corrupted * F.relu(c['uncertainty_margin'] - u).square() + (~corrupted) * u
            loss += c['uncertainty_weight'] * (available * penalty).mean()
        return loss


def extra_metrics(d, c):
    metrics = dict(mean_uncertainty=float(d['uncertainty'].mean()))
    if 'target_sentiment' in d:
        y, p = d['target_sentiment'], d['regression']
        metrics['sentiment_mae'] = float(np.abs(y - p).mean())
        metrics['sentiment_pearson'] = float(pearsonr(y, p).statistic)
    return metrics
