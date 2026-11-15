"""Labelling directions of the motion site by what they respond to: camera or scene (Part C, D-42).

A direction is a column of clip-level activations: an SAE feature, a raw channel or a random unit
direction, each averaged over the tokens of a clip. Its selectivity for a label is the one-vs-rest
AUROC, two-sided (max(a, 1 - a)): raw channels are signed, and a feature that switches off for one
camera is as informative as one that switches on. Taking the two-sided value for every kind of
direction keeps the comparison between SAE features and the baselines fair.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata

CAMERA_MIN, SCENE_ETA2_MAX, HS1_MIN_FEATURES = 0.9, 0.1, 10


def parse_clip_id(cid: str) -> dict:
    """"<seed>__<scene-slug>__<camera>" (camera_set.CameraClip.id)."""
    seed, scene, camera = cid.split("__")
    return {"clip": cid, "seed": int(seed), "scene": scene, "camera": camera}


def auroc(x: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """One AUROC per column of x (n, m) for a boolean label (n,): P(score of a positive > score of a negative),
    ties counted half. NaN when one class is empty."""
    positive = np.asarray(positive, bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return np.full(x.shape[1], np.nan)
    ranks = rankdata(x, axis=0)
    return (ranks[positive].sum(0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def selectivity(x: np.ndarray, labels) -> tuple[np.ndarray, np.ndarray]:
    """Per column: the best two-sided one-vs-rest AUROC over the classes of `labels`, and that class's index
    in sorted(set(labels))."""
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    a = np.stack([auroc(x, labels == c) for c in classes])        # (classes, m)
    a = np.maximum(a, 1 - a)
    return a.max(0), a.argmax(0)


def eta2(x: np.ndarray, labels) -> np.ndarray:
    """Per column: the share of the variance over clips that lies between the classes of `labels`
    (between-class sum of squares / total). 0 for a constant column."""
    labels = np.asarray(labels)
    mean = x.mean(0)
    total = ((x - mean) ** 2).sum(0)
    between = sum((labels == c).sum() * (x[labels == c].mean(0) - mean) ** 2 for c in set(labels.tolist()))
    return np.where(total > 0, between / np.where(total > 0, total, 1.0), 0.0)


def label_directions(x: np.ndarray, cameras, scenes) -> dict:
    """Camera and scene selectivity of every column, and the HS1 count.

    Camera-only: best one-vs-rest camera AUROC >= 0.9 and scene variance share <= 0.1. The scene side
    is a variance share, not a best-of-classes AUROC: with nine clips per scene the best of eight
    one-vs-rest AUROCs passes 0.6 by chance for almost any direction, so that rule could not be met.
    """
    cam, cam_arg = selectivity(x, cameras)
    sc, _ = selectivity(x, scenes)
    e_cam, e_scene = eta2(x, cameras), eta2(x, scenes)
    hit = (cam >= CAMERA_MIN) & (e_scene <= SCENE_ETA2_MAX)
    return {"camera_auroc": cam, "camera_argmax": cam_arg, "scene_auroc": sc, "camera_eta2": e_cam,
            "scene_eta2": e_scene, "camera_only": hit, "n_camera_only": int(hit.sum()),
            "classes": sorted(set(np.asarray(cameras).tolist()))}


@torch.no_grad()
def clip_means(tokens: dict, encode=None, steps=None, batch: int = 8192, device: str = "cpu") -> tuple[list[str], np.ndarray]:
    """{clip: {step: (n, C)}} -> clip ids and the (clips, m) mean of `encode(tokens)` over a clip's tokens
    (all `steps` pooled; encode=None keeps the raw channels)."""
    ids, rows = sorted(tokens), []
    for cid in ids:
        x = torch.cat([t for s, t in sorted(tokens[cid].items()) if steps is None or s in steps]).float()
        if encode is None:
            rows.append(x.mean(0))
            continue
        total = 0
        for i in range(0, len(x), batch):
            total = total + encode(x[i:i + batch].to(device)).sum(0).cpu()
        rows.append(total / len(x))
    return ids, torch.stack(rows).numpy()
