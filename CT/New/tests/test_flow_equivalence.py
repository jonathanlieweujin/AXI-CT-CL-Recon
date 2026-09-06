"""
QualityFlow (TIGRE) and PerformanceFlow (ASTRA) must reconstruct the same
volume from the same projections.

Three geometries, each backed by a synthetic dataset on disk whose projections
were forward-projected from a known phantom:

    CT        inclined, DetTiltX = 90
    CL        inclined, DetTiltX = 55
    Coplanar  coplanar, DetTiltX = 55

Regenerate the data with:  python tests/make_synthetic_data.py
"""
import configparser
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Util.param import VxParam
from Util.load_files import FileLoader
from Util.recon_method import ReconMethodConstants
from Util.laminography_method import LaminographyMethodConstants
from QualityFlow.get_structure import VxTool as QualityTool
from PerformanceFlow.get_structure import VxTool as PerformanceTool

DATASET_ROOT = r"C:\Users\User\Documents\Dataset"

# The two flows use different toolkits, so exact equality is not the bar;
# these are "same volume, same orientation, same features" thresholds.
MIN_FLOW_AGREEMENT = 0.99
MIN_PHANTOM_AGREEMENT = 0.50

CASES = [
    ("Synthetic CT", LaminographyMethodConstants.INCLINED, 90.0),
    ("Synthetic CL", LaminographyMethodConstants.INCLINED, 55.0),
    ("Synthetic Coplanar", LaminographyMethodConstants.COPLANAR, 55.0),
]


def load_config(dataset):
    path = os.path.join(DATASET_ROOT, dataset, "Config", "geometry.config")
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    cfg = dict(parser["VXMPR CONFIG"])
    cfg["projectionangles"] = np.array(
        [float(v) for v in cfg["projectionangles"].strip("[]").split(",")])
    return cfg


def build(dataset):
    cfg = load_config(dataset)
    angles = cfg["projectionangles"]
    interval = int(cfg["interval"])
    binning = int(cfg["binning"])
    angles = angles[::interval]

    images = FileLoader.loadImages(
        os.path.join(DATASET_ROOT, dataset, "Corrected"),
        int(cfg["detu"]), int(cfg["detv"]), len(angles),
        interval=interval, binning=binning)

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
        iterations=int(cfg["iterations"]),
    )
    phantom = np.load(os.path.join(DATASET_ROOT, dataset, "phantom.npy"))
    return param, images, phantom, cfg


def corr(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


@pytest.fixture(scope="module")
def results():
    out = {}
    for dataset, method, _ in CASES:
        param, images, phantom, _ = build(dataset)
        q = QualityTool(param, laminography_method=method).run(
            images.copy(), algo=ReconMethodConstants.FDK)
        p = PerformanceTool(param, laminography_method=method).run(
            images.copy(), algo=ReconMethodConstants.FDK)
        out[dataset] = (q, p, phantom)
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
    quality, performance, phantom = results[dataset]

    assert quality.shape == performance.shape, (
        f"{dataset}: shape mismatch {quality.shape} vs {performance.shape}")
    assert quality.shape == phantom.shape, (
        f"{dataset}: recon shape {quality.shape} != phantom {phantom.shape}")

    # a blank volume correlates with nothing; catch it explicitly
    assert quality.std() > 0, f"{dataset}: QualityFlow returned a constant volume"
    assert performance.std() > 0, f"{dataset}: PerformanceFlow returned a constant volume"

    c = corr(quality, performance)
    assert c >= MIN_FLOW_AGREEMENT, f"{dataset}: flows disagree, corr={c:.4f}"


@pytest.mark.parametrize("dataset,method,tilt_x", CASES,
                         ids=[c[0].replace("Synthetic ", "") for c in CASES])
def test_both_flows_recover_phantom(results, dataset, method, tilt_x):
    quality, performance, phantom = results[dataset]
    cq = corr(quality, phantom)
    cp = corr(performance, phantom)
    assert cq >= MIN_PHANTOM_AGREEMENT, f"{dataset}: QualityFlow vs phantom corr={cq:.4f}"
    assert cp >= MIN_PHANTOM_AGREEMENT, f"{dataset}: PerformanceFlow vs phantom corr={cp:.4f}"


def test_report(results, capsys):
    """Not an assertion - prints the agreement figures for the record."""
    with capsys.disabled():
        print(f"\n{'dataset':22s} {'shape':>16s} {'quality~perf':>13s} "
              f"{'quality~phantom':>16s} {'perf~phantom':>13s}")
        for dataset, _, _ in CASES:
            q, p, ph = results[dataset]
            print(f"{dataset:22s} {str(q.shape):>16s} {corr(q, p):13.4f} "
                  f"{corr(q, ph):16.4f} {corr(p, ph):13.4f}")
