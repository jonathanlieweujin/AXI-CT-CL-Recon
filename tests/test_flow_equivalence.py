"""
The Quality (TIGRE) and Performance (ASTRA) flows must reconstruct the same
volume from the same projections.

Three geometries, each backed by a synthetic dataset on disk whose projections
were forward-projected from a known phantom:

    CT        inclined, DetTiltX = 90
    CL        inclined, DetTiltX = 51
    Coplanar  coplanar, DetTiltX = 51

Regenerate the data with:  python tests/make_synthetic_data.py
"""
import configparser
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Manager import VxManager
from Manager.Param import VxParam
from Manager.Constants.recon_method import ReconMethodConstants
from Manager.Constants.laminography_method import LaminographyMethodConstants
from Manager.Constants.flow_method import VxFlowMethod

DATASET_ROOT = r"C:\Users\User\Documents\Dataset"

# The two flows use different toolkits, so exact equality is not the bar;
# these are "same volume, same orientation, same features" thresholds.
MIN_FLOW_AGREEMENT = 0.99
MIN_PHANTOM_AGREEMENT = 0.50

CASES = [
    ("Synthetic CT", LaminographyMethodConstants.INCLINED, 90.0),
    ("Synthetic CL", LaminographyMethodConstants.INCLINED, 51.0),
    ("Synthetic Coplanar", LaminographyMethodConstants.COPLANAR, 51.0),
]


def load_config(dataset):
    path = os.path.join(DATASET_ROOT, dataset, "Config", "geometry.config")
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    cfg = dict(parser["VXMPR CONFIG"])
    cfg["projectionangles"] = np.array(
        [float(v) for v in cfg["projectionangles"].strip("[]").split(",")])
    return cfg


def build(dataset, method):
    cfg = load_config(dataset)
    interval = int(cfg["interval"])
    binning = int(cfg["binning"])
    angles = cfg["projectionangles"][::interval]

    param = VxParam(
        sod=float(cfg["sod"]), sdd=float(cfg["sdd"]), angles=angles,
        tilt_x=float(cfg["dettiltx"]), tilt_y=float(cfg["dettilty"]),
        det_width=int(cfg["detu"]), det_height=int(cfg["detv"]),
        det_pitch=float(cfg["detpitch"]),
        offset_u=float(cfg["detoffsetu"]), offset_v=float(cfg["detoffsetv"]),
        binning=binning,
        volume_mid_x=float(cfg["volmidx"]), volume_mid_y=float(cfg["volmidy"]),
        volume_mid_z=float(cfg["volmidz"]),
        dst_x=int(cfg["volx"]), dst_y=int(cfg["voly"]), dst_z=int(cfg["volz"]),
        recon_method=ReconMethodConstants.FDK,
        laminography_method=method,
        iterations=int(cfg["iterations"]),
    )
    manager = VxManager()
    manager.SetParam(param)
    manager.SetFlowMethod(VxFlowMethod.QUALITY)
    images = manager.LoadImages(
        os.path.join(DATASET_ROOT, dataset, "Corrected"), interval=interval)
    phantom = np.load(os.path.join(DATASET_ROOT, dataset, "phantom.npy"))
    return manager, images, phantom, cfg


def corr(a, b, chunk: int = 1 << 24):
    """
    Pearson correlation of two volumes.

    These are ~294 M voxels, so the obvious form is dominated by temporaries:
    a.astype(float64) alone is 2.4 GB and centring makes another. Instead the
    five sums are accumulated in float64 straight off the float32 buffers
    (sum(dtype=...) and einsum(dtype=...) promote per element, not per array),
    which measures 0.9 s against 2.7 s for the naive version and agrees with it
    to ~1e-14. Chunking bounds the working set whatever the volume size.
    """
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.size != b.size:
        raise ValueError(f"size mismatch: {a.size} vs {b.size}")
    n = a.size
    if n == 0:
        return 0.0

    sa = sb = saa = sbb = sab = 0.0
    for i in range(0, n, chunk):
        x = a[i:i + chunk]
        y = b[i:i + chunk]
        sa += float(x.sum(dtype=np.float64))
        sb += float(y.sum(dtype=np.float64))
        saa += float(np.einsum("i,i->", x, x, dtype=np.float64))
        sbb += float(np.einsum("i,i->", y, y, dtype=np.float64))
        sab += float(np.einsum("i,i->", x, y, dtype=np.float64))

    cov = sab - sa * sb / n
    va = saa - sa * sa / n
    vb = sbb - sb * sb / n
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    return float(cov / np.sqrt(va * vb))


@pytest.fixture(scope="module")
def results():
    out = {}
    for dataset, method, _ in CASES:
        manager, images, phantom, _ = build(dataset, method)

        manager.FlowMethod = VxFlowMethod.QUALITY
        manager.Run(images.copy())
        q = manager.Result

        manager.FlowMethod = VxFlowMethod.PERFORMANCE
        manager.Run(images.copy())
        p = manager.Result

        # correlating 294 M voxels is the expensive part of this suite, so each
        # pair is measured once here rather than again in every test
        out[dataset] = (q, p, phantom, {
            "quality_performance": corr(q, p),
            "quality_phantom": corr(q, phantom),
            "performance_phantom": corr(p, phantom),
        })
    return out


@pytest.mark.parametrize("dataset,method,tilt_x", CASES,
                         ids=[c[0].replace("Synthetic ", "") for c in CASES])
def test_config_matches_case(dataset, method, tilt_x):
    cfg = load_config(dataset)
    assert float(cfg["dettiltx"]) == tilt_x
    assert cfg["laminographymethod"] == str(method)


@pytest.mark.parametrize("dataset,method,tilt_x", CASES,
                         ids=[c[0].replace("Synthetic ", "") for c in CASES])
def test_flows_agree(results, dataset, method, tilt_x):
    quality, performance, phantom, scores = results[dataset]

    assert quality.shape == performance.shape, (
        f"{dataset}: shape mismatch {quality.shape} vs {performance.shape}")
    assert quality.shape == phantom.shape, (
        f"{dataset}: recon shape {quality.shape} != phantom {phantom.shape}")

    # a blank volume correlates with nothing; catch it explicitly
    assert quality.std() > 0, f"{dataset}: Quality flow returned a constant volume"
    assert performance.std() > 0, f"{dataset}: Performance flow returned a constant volume"

    c = scores["quality_performance"]
    assert c >= MIN_FLOW_AGREEMENT, f"{dataset}: flows disagree, corr={c:.4f}"


@pytest.mark.parametrize("dataset,method,tilt_x", CASES,
                         ids=[c[0].replace("Synthetic ", "") for c in CASES])
def test_both_flows_recover_phantom(results, dataset, method, tilt_x):
    quality, performance, phantom, scores = results[dataset]
    cq = scores["quality_phantom"]
    cp = scores["performance_phantom"]
    assert cq >= MIN_PHANTOM_AGREEMENT, f"{dataset}: Quality flow vs phantom corr={cq:.4f}"
    assert cp >= MIN_PHANTOM_AGREEMENT, f"{dataset}: Performance flow vs phantom corr={cp:.4f}"


def test_report(results, capsys):
    """Not an assertion - prints the agreement figures for the record."""
    with capsys.disabled():
        print(f"\n{'dataset':22s} {'shape':>16s} {'quality~perf':>13s} "
              f"{'quality~phantom':>16s} {'perf~phantom':>13s}")
        for dataset, _, _ in CASES:
            q, _, _, scores = results[dataset]
            print(f"{dataset:22s} {str(q.shape):>16s} "
                  f"{scores['quality_performance']:13.4f} "
                  f"{scores['quality_phantom']:16.4f} "
                  f"{scores['performance_phantom']:13.4f}")
