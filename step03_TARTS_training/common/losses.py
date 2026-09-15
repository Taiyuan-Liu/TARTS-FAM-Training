"""Physical regression losses and the author's DARE-GRAM feature loss."""

import math

import numpy as np
import torch
from torch import nn


ARCSEC_PER_MICRON = np.array([
    0.75453564, 0.27113150, 0.27113150, 0.82237164, 0.82237164,
    0.39572461, 0.39572461, 1.68719298, 0.93956571, 0.93956571,
    0.51643078, 0.51643078, 1.76255706, 1.76255706, 1.09112012,
    1.09112012, 0.63498144, 0.63498144, 2.82321385, 1.87588235,
    1.87588235, 1.26437424, 1.26437424, 0.75241174, 0.75241174,
], dtype=np.float32)


def regression_loss(prediction, truth, kind):
    weights = torch.as_tensor(ARCSEC_PER_MICRON, device=prediction.device)
    squared = (((prediction.float() - truth) * weights) ** 2).sum(dim=1)
    return squared.mean() if kind == "wavenet" else (squared + 1e-12).sqrt().mean()


def metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    error = prediction - truth
    correlation = []
    for pred, target in zip(prediction.T, truth.T):
        correlation.append(float(np.corrcoef(pred, target)[0, 1])
                           if pred.std() > 0 and target.std() > 0 else None)
    return dict(samples=len(error), coefficient_rmse_um=float(np.sqrt(np.mean(error**2))),
                mrsse_arcsec=float(np.linalg.norm(error * ARCSEC_PER_MICRON, axis=1).mean()),
                per_mode_rmse_um=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                per_mode_bias_um=error.mean(axis=0).tolist(), per_mode_corr=correlation)


def dare_gram(source, target, threshold=0.9, tradeoff_angle=0.05, tradeoff_scale=0.001):
    """Extracted from the author's two DARE-GRAM methods; use float32 SVD."""
    source, target = source.float(), target.float()
    if source.shape != target.shape or len(source) < 2:
        raise ValueError("DARE requires equal feature shapes and at least two samples")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite DARE features")
    count, width = source.shape
    ones = torch.ones(count, 1, device=source.device)
    a, b = torch.cat([ones, source], 1), torch.cat([ones, target], 1)
    cov_a, cov_b = a.T @ a, b.T @ b
    _, la, _ = torch.linalg.svd(cov_a)
    _, lb, _ = torch.linalg.svd(cov_b)
    if la[0] < 1e-10 or lb[0] < 1e-10:
        return (source.sum() + target.sum()) * 0
    # CPU cumsum is also valid when deterministic CUDA cumsum is unavailable.
    ea = (la.detach().cpu().cumsum(0) / la.detach().cpu().sum()).to(source.device)
    eb = (lb.detach().cpu().cumsum(0) / lb.detach().cpu().sum()).to(source.device)
    def rank(eigen):
        indices = torch.argwhere(eigen <= max(eigen[1].item(), threshold))
        return int(indices[-1, 0]) if len(indices) else 1
    k = min(max(rank(ea), rank(eb)), len(la) - 1, len(lb) - 1)
    pa = torch.linalg.pinv(cov_a, rtol=max((la[k] / la[0]).item(), 1e-6))
    pb = torch.linalg.pinv(cov_b, rtol=max((lb[k] / lb[0]).item(), 1e-6))
    angle = torch.dist(torch.ones(width + 1, device=source.device),
                       nn.functional.cosine_similarity(pa, pb, dim=0, eps=1e-6), p=1) / (width + 1)
    scale = torch.dist(la[:k], lb[:k], p=1) / k
    return (tradeoff_angle * angle.clamp(0, 10) + tradeoff_scale * scale.clamp(0, 100)).clamp(0, 100)


def source_guard_scale(score, baseline, fraction):
    """Exponentially decrease the DARE weight over the configured source-error interval."""
    if score <= baseline:
        return 1.0
    if score >= baseline * (1 + fraction):
        return 0.0
    t = (score - baseline) / (baseline * fraction)
    return 1 - (1 - math.exp(-6 * t)) / (1 - math.exp(-6))
