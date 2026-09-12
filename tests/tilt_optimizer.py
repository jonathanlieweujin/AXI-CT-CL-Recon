"""
Description:

* Computes inclined laminography tilt (DetTiltX) and detector offset
  (offset_u, offset_v) from projection images, based on known phantom /
  jig design information - ball count, diameter and 3D coordinates (here,
  MICRO_JIG's fitted layout, see
  SyntheticDataGenerator/Structures/micro_jig.py) - rather than the
  scanner's own declared geometry. sod, sdd, det_pitch, det_width/height,
  the rotation angle of every projection and tilt_y are assumed known
  throughout; only tilt_x, offset_u and offset_v are treated as unknown.

Methodology:

* Ball detection: threshold each projection at the 80th percentile of its
  nonzero pixels, then take each compact connected component as one ball's
  blob centroid.
* Frame selection: keep only frames where nearly every ball resolves as
  its own blob. A frame with several balls merged/occluded biases the fit
  rather than just adding noise to it, so fewer clean frames beat more
  noisy ones.
* Hungarian assignment: for a trial (tilt_x, offset_u, offset_v), match
  detected blobs to the known geometry's predicted ball positions via
  optimal (minimum-cost) assignment, rather than hand-tracking a single
  ball's trajectory.
* tilt_x: 1D coarse-then-fine grid search over the matched reprojection
  error.
* offset_u/offset_v: solved in closed form (mean residual of the matched
  pairs, iterated to convergence) and profiled out of the tilt search - a
  detector offset is exactly a constant pixel-space translation of every
  ball's projection, independent of tilt, frame or ball identity, so no
  separate search over it is needed (see FitOffsetAndCost). This pairing -
  tilt angle plus horizontal/vertical shift as a scan's "imaging
  parameters", solved apart from the scanner's fixed sod/sdd "system
  parameters" - follows Xiao et al. 2018's parameter-division grouping.

Validation:

* The trial geometry built for the search starts with default tilt_x at 0
  and offset_u/offset_v at 0.
* The synthetic dataset's own geometry.config still carries the true
  tilt_x/offset_u/offset_v it was generated with (LoadConfigFields reads
  them as tilt_x_truth/offset_u/offset_v). Those are read only after the
  search has produced its estimate, purely to report the error.
* EstimateTilt's printed summary and returned dict (estimated_tilt_x /
  truth_tilt_x, estimated_offset_u / truth_offset_u, etc.) put the two
  side by side so this comparison is explicit rather than implied.
* This only works because the dataset is synthetic and the truth is known
  in advance; against a real, unlabelled scan the estimate would be the
  only number available. 
* Ground truth here exists to confirm the method recovers the geometry
  parameters

Run:
    .venv\\python.exe tests\\tilt_optimizer.py
    .venv\\python.exe tests\\tilt_optimizer.py --dataset output\\Synthetic\\MICRO_JIG

Ref:
- Schwinn, M. (2015). Estimation of geometrical set-ups for computed laminography 3D-reconstruction [Master's thesis, Saarland University]. Mathematical Image Analysis Group.
- Xiao et al. (2018). A parameter division based method for the geometrical calibration of X-ray industrial cone-beam CT. IEEE Access, 6, 48970–48977.
"""

import argparse
import configparser
import glob
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Manager import VxManager
from Manager.Param import VxParam
from Manager.Constants.laminography_method import LaminographyMethodConstants
from SyntheticDataGenerator.Structures.micro_jig import MicroJigStructure


# ----------------------------------------------------------------------
# geometry.config: read the raw fields we need with plain configparser,
# same INI section VxManager itself writes/reads ("VXMPR CONFIG").
# ----------------------------------------------------------------------
def LoadConfigFields(config_path: str) -> dict:
    cp = configparser.ConfigParser()
    cp.read(config_path)
    sec = cp[cp.sections()[0]]
    angles = np.array([float(x) for x in sec["projectionangles"].strip("[]").split(",")])
    return dict(
        det_width=int(sec["detu"]), det_height=int(sec["detv"]),
        det_pitch=float(sec["detpitch"]), sod=float(sec["sod"]), sdd=float(sec["sdd"]),
        tilt_x_truth=float(sec["dettiltx"]), tilt_y=float(sec["dettilty"]),
        offset_u=float(sec["detoffsetu"]), offset_v=float(sec["detoffsetv"]),
        binning=int(sec["binning"]), angles=angles,
        laminography_method=(LaminographyMethodConstants.COPLANAR
                              if sec["isdetectorparalleltofov"].strip().lower() == "true"
                              else LaminographyMethodConstants.INCLINED),
    )


def BuildParam(fields: dict, tilt_x_guess: float = 0.0) -> VxParam:
    """
    Build once with the *known* fields and binning as given by the config
    (constructed exactly once - trial tilt_x values are then applied by
    mutating `.tilt_x` directly, never by re-invoking the constructor,
    since VxParam's constructor divides det_pitch/offsets by `binning`
    every time it runs; re-constructing per trial would silently re-apply
    that scaling on every iteration).
    """
    return VxParam(
        sod=fields["sod"], sdd=fields["sdd"], angles=fields["angles"],
        tilt_x=tilt_x_guess, tilt_y=fields["tilt_y"],
        det_width=fields["det_width"], det_height=fields["det_height"],
        det_pitch=fields["det_pitch"],
        # offset_u/offset_v start at 0, not fields["offset_u"/"offset_v"]: those
        # are exactly the "shifting" imaging parameters this module estimates,
        # so seeding the trial geometry with the true value would make
        # FitOffsetAndCost's fitted offset trivially converge to ~0 rather
        # than actually recovering it from the images. fields["offset_u"/"v"]
        # are read only so EstimateTilt can report ground truth at the end.
        offset_u=0.0, offset_v=0.0,
        binning=fields["binning"],
        dst_x=1, dst_y=1, dst_z=1,  # reconstruction volume is irrelevant here
        laminography_method=fields["laminography_method"],
    )


# ----------------------------------------------------------------------
# Ball detection in the raw float32 projections
# ----------------------------------------------------------------------
def DetectBallBlobs(frame: np.ndarray, min_area: int = 80, max_area: int = 3000) -> np.ndarray:
    """
    (N, 2) array of (u_pixel, v_pixel) blob centroids. Ball hits are the
    locally brightest, most compact regions (ruby's higher attenuation
    coefficient than the quartz pillars/base plate, and a ball is the only
    compact near-circular feature of this size - the plate can be brighter
    still at grazing angles, but shows up as a large or elongated region,
    filtered out by the area cap rather than by intensity).
    """
    nz = frame[frame > 0]
    if nz.size == 0:
        return np.empty((0, 2))
    threshold = np.percentile(nz, 80)
    mask = (frame > threshold).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = (areas >= min_area) & (areas <= max_area)
    return centroids[1:][keep]


# ----------------------------------------------------------------------
# Forward projection: known 3D ball centres -> predicted detector pixels,
# via the same source/detector/U/V cone_vec convention GetScanGeometry()
# returns (object static at the origin; ray-plane intersection then
# expressed in U/V pixel units, following the same plane-intersection
# algebra as SyntheticDataGenerator's own solveCoverage_Internal).
# ----------------------------------------------------------------------
def ProjectPoints(vec_row: np.ndarray, points_mm: np.ndarray,
                   det_width: int, det_height: int) -> np.ndarray:
    S, D, U, V = vec_row[0:3], vec_row[3:6], vec_row[6:9], vec_row[9:12]
    n = np.cross(U, V)
    d = points_mm - S                       # (N,3)
    denom = d @ n                           # (N,)
    t = np.dot(D - S, n) / denom
    hit = S + t[:, None] * d                # (N,3)
    a = ((hit - D) @ U) / (U @ U)
    b = ((hit - D) @ V) / (V @ V)
    u_pixel = det_width / 2.0 + a
    v_pixel = det_height / 2.0 + b
    return np.stack([u_pixel, v_pixel], axis=1)


def FitOffsetAndCost(tilt_x: float, param: VxParam, frame_indices: list[int],
                      detections: dict[int, np.ndarray], balls_mm: np.ndarray,
                      det_width: int, det_height: int, max_gate_px: float = 25.0,
                      offset_iters: int = 3) -> tuple[float, float, float]:
    """
    For a trial tilt_x, jointly resolve the detector's "shifting" imaging
    parameters (Xiao et al. 2018's term - detector centre offset along U
    and V, i.e. offset_u/offset_v) rather than leaving them at 0.

    A detector offset shifts D by `offset_u*U + offset_v*V`, and U/V are
    orthogonal with |U|=|V|=det_pitch, so its effect on every ball's
    predicted pixel position is *exactly* the constant translation
    `(-offset_u, -offset_v)` - independent of tilt, frame, or which ball
    (verified numerically against the real VxParam offset mechanism: a
    known (offset_u, offset_v) reproduces that exact shift to float
    precision). That means, for any trial tilt_x, the best-fit offset given
    a correspondence is just the mean residual of the matched pairs - no
    grid search over offset needed, only alternating between (a) matching
    detections to the current offset-corrected prediction and (b)
    re-centring the offset on the residual, which converges in a couple of
    iterations. This profiles offset_u/offset_v out of the outer tilt_x
    search entirely.

    Returns (mean squared pixel error, offset_u, offset_v) at convergence.
    """
    param.tilt_x = tilt_x
    manager = VxManager()
    manager.SetParam(param)
    vecs = manager.GetScanGeometry()
    preds0 = {i: ProjectPoints(vecs[i], balls_mm, det_width, det_height) for i in frame_indices}

    def match_all(offset):
        total_sq, total_n, resid = 0.0, 0, []
        for i in frame_indices:
            det = detections[i]
            if det.shape[0] == 0:
                continue
            pred = preds0[i] - offset
            cost = np.sum((det[:, None, :] - pred[None, :, :]) ** 2, axis=2)
            rows, cols = linear_sum_assignment(cost)
            d2 = cost[rows, cols]
            keep = d2 < max_gate_px ** 2
            if not np.any(keep):
                continue
            resid.append(preds0[i][cols[keep]] - det[rows[keep]])
            total_sq += d2[keep].sum()
            total_n += int(keep.sum())
        return total_sq, total_n, resid

    offset = np.zeros(2)
    for _ in range(offset_iters):
        _, _, resid = match_all(offset)
        if resid:
            offset = np.concatenate(resid, axis=0).mean(axis=0)

    total_sq, total_n, _ = match_all(offset)
    mse = total_sq / total_n if total_n else np.inf
    return mse, float(offset[0]), float(offset[1])


def SelectGoodFrames(detections: dict[int, np.ndarray], n_balls: int = 22,
                      min_good_frames: int = 8, verbose: bool = True) -> list[int]:
    """
    Keep only frames where nearly every ball was detected as its own blob.

    A frame with several balls merged/occluded in projection isn't just
    noisier - the Hungarian assignment on it is more likely to match a
    detected blob to the *wrong* nearby ball outright, which biases the
    fit rather than just adding scatter to it. Empirically (this dataset),
    loosening the threshold to sweep in more frames pulls the recovered
    tilt further from the truth even as the frame count goes up by 10x -
    fewer, cleaner frames beat more, noisier ones. So: start strict and
    only relax the count threshold enough to reach `min_good_frames`.
    """
    counts = {i: d.shape[0] for i, d in detections.items()}
    for min_count in range(n_balls, 0, -1):
        good = [i for i, c in counts.items() if c >= min_count]
        if len(good) >= min_good_frames:
            if verbose:
                print(f"Frame quality filter: keeping {len(good)} frames with "
                      f">={min_count}/{n_balls} balls detected "
                      f"(of {len(counts)} sampled)")
            return good
    return list(counts.keys())


def EstimateTilt(dataset_root: str, frame_stride: int = 2,
                  coarse_range=(-10.0, 110.0), coarse_step=1.0,
                  fine_half_width=2.0, fine_step=0.02,
                  min_good_frames: int = 8, verbose=True) -> dict:
    config_path = os.path.join(dataset_root, "Config", "geometry.config")
    corrected_dir = os.path.join(dataset_root, "Corrected")
    fields = LoadConfigFields(config_path)
    param = BuildParam(fields)
    balls_mm = MicroJigStructure.GetJigBallPosition()

    files = sorted(glob.glob(os.path.join(corrected_dir, "proj_*.tif")))
    sampled_indices = list(range(0, len(files), frame_stride))
    if verbose:
        print(f"Dataset: {dataset_root}  ({len(files)} projections, "
              f"sampling every {frame_stride} -> {len(sampled_indices)} frames)")

    detections = {}
    for i in sampled_indices:
        img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
        detections[i] = DetectBallBlobs(img)
    counts = [detections[i].shape[0] for i in sampled_indices]
    if verbose:
        print(f"Detected blobs per frame: min={min(counts)} median={int(np.median(counts))} "
              f"max={max(counts)} (22 balls expected)")

    frame_indices = SelectGoodFrames(detections, min_good_frames=min_good_frames, verbose=verbose)

    def cost_only(t):
        mse, _, _ = FitOffsetAndCost(t, param, frame_indices, detections, balls_mm,
                                      fields["det_width"], fields["det_height"])
        return mse

    t0 = time.time()
    lo, hi = coarse_range
    coarse_grid = np.arange(lo, hi + coarse_step, coarse_step)
    coarse_cost = np.array([cost_only(t) for t in coarse_grid])
    coarse_best = float(coarse_grid[np.argmin(coarse_cost)])
    if verbose:
        print(f"Coarse search: best tilt_x = {coarse_best:.2f} deg "
              f"(cost={coarse_cost.min():.3f} px^2), {time.time()-t0:.1f}s")

    fine_grid = np.arange(coarse_best - fine_half_width, coarse_best + fine_half_width + fine_step, fine_step)
    fine_cost = np.array([cost_only(t) for t in fine_grid])
    fine_best = float(fine_grid[np.argmin(fine_cost)])

    # final pass at the converged tilt to report the matching offset_u/offset_v
    final_mse, offset_u, offset_v = FitOffsetAndCost(
        fine_best, param, frame_indices, detections, balls_mm,
        fields["det_width"], fields["det_height"])

    if verbose:
        print(f"Fine search:   best tilt_x = {fine_best:.3f} deg "
              f"(cost={final_mse:.4f} px^2), total {time.time()-t0:.1f}s")
        print(f"Fitted detector offset: offset_u = {offset_u:+.3f} px, "
              f"offset_v = {offset_v:+.3f} px")
        print(f"Ground truth (from geometry.config, not used in the search): "
              f"tilt_x = {fields['tilt_x_truth']:.3f} deg, "
              f"offset_u = {fields['offset_u']:+.3f} px, offset_v = {fields['offset_v']:+.3f} px")
        print(f"Error: tilt_x {fine_best - fields['tilt_x_truth']:+.3f} deg, "
              f"offset_u {offset_u - fields['offset_u']:+.3f} px, "
              f"offset_v {offset_v - fields['offset_v']:+.3f} px")

    return dict(estimated_tilt_x=fine_best, truth_tilt_x=fields["tilt_x_truth"],
                estimated_offset_u=offset_u, estimated_offset_v=offset_v,
                truth_offset_u=fields["offset_u"], truth_offset_v=fields["offset_v"],
                coarse_grid=coarse_grid, coarse_cost=coarse_cost,
                fine_grid=fine_grid, fine_cost=fine_cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=r"output\Synthetic\MICRO_JIG",
                         help="Dataset root containing Config/geometry.config and Corrected/*.tif")
    parser.add_argument("--stride", type=int, default=2,
                         help="Scan every Nth projection frame for ball blobs before quality-filtering")
    parser.add_argument("--min-good-frames", type=int, default=8,
                         help="Minimum near-complete-detection frames to keep for the fit")
    args = parser.parse_args()

    result = EstimateTilt(args.dataset, frame_stride=args.stride,
                           min_good_frames=args.min_good_frames)