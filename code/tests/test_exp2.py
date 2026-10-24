"""Experiment 2 analysis primitives on synthetic data with planted answers."""

import numpy as np
import pandas as pd
import pytest

from vdprobe import exp2 as E


def test_sign_flip_exact_and_fwer():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (10, 50))
    X[:, 7] += 3.0                                   # one real effect among 50 cells
    r = E.sign_flip_test(X, n_flip=2 ** 10)
    assert r["n_patterns"] == 1024
    assert r["p_fwer"][7] < 0.01 and r["p"][7] <= 2 / 1024 + 1e-12
    assert np.median(r["p_fwer"][np.arange(50) != 7]) > 0.5
    one = E.sign_flip_test(-np.abs(X[:, :1]) - 0.1, alternative="less")
    assert one["p"][0] == pytest.approx(1 / 1024)    # all subjects negative: the most extreme pattern


def test_anova_fractions_recover_planted_factors():
    rng = np.random.default_rng(1)
    rows = [{"subject": s, "verb": v, "seed": k} for s in range(6) for v in range(4) for k in range(2)]
    d = pd.DataFrame(rows)
    subj, verb = rng.normal(0, 3, (6, 32)), rng.normal(0, 1, (4, 32))
    Z = subj[d.subject] + verb[d.verb] + rng.normal(0, 0.1, (len(d), 32))
    f = E.anova_fractions(Z, d, ["subject", "verb", "seed"], pairs=[("subject", "verb")])
    assert f["subject"] > 0.8 and 0.05 < f["verb"] < 0.2 and f["seed"] < 0.01 and f["subjectxverb"] < 0.02
    assert abs(sum(f.values()) - 1) < 1e-9


def test_flow_rdm_and_descriptor():
    n, P, h = 3, 4, 16
    flows = np.zeros((n, P, h, h, 2), np.float32)
    masks = np.zeros((n, P + 1, h, h), bool)
    masks[:, :, 4:12, 4:12] = True
    flows[0, :, 4:12, 4:12] = (1, 0)
    flows[1, :, 4:12, 4:12] = (1, 0)
    flows[1, :, :4] = (0, 5)                         # clip 1: also a moving background strip
    flows[2, :, 4:12, 4:12] = (-1, 0)
    D = E.flow_rdm(flows, masks)
    assert D[0, 1] == pytest.approx(0, abs=1e-6) and D[0, 2] == pytest.approx(2, abs=1e-6)
    assert E.flow_descriptor(flows[0], masks[0], grid=4).shape == (32,)


def test_decoding_gain_detects_motion_beyond_the_latent():
    rng = np.random.default_rng(2)
    n_prompts, seeds = 60, 2
    pid = np.repeat(np.arange(n_prompts), seeds)
    groups = pid % 6
    Y = rng.normal(0, 1, (len(pid), 5))                         # per-clip generated motion
    lat = rng.normal(0, 1, (len(pid), 20))                      # latent unrelated to it
    layer = np.hstack([Y @ rng.normal(0, 1, (5, 20)), rng.normal(0, 1, (len(pid), 20))])
    g = E.decoding_gain(layer, lat, Y, pid, groups)
    assert g["r2_layer"] > 0.8 and g["r2_latent"] < 0.1 and g["gain"] > 0.7


def test_channel_selectivity():
    rng = np.random.default_rng(3)
    lab = np.repeat(np.arange(4), 10)
    e = rng.normal(0, 1, (40, 8))
    e[:, 2] += lab * 3                                           # channel 2 tracks the verb
    eta = E.channel_selectivity(e, lab)
    assert eta.argmax() == 2 and eta[2] > 0.8
