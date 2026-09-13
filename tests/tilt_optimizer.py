"""
Description:

* Computes inclined laminography tilt (DetTiltX) and detector offset
  (offset_u, offset_v) from projection images, based on known phantom /
  jig design information - ball count, diameter and 3D coordinates (here,
  MICRO_JIG's fitted layout, 
* Pre-requisites: system geometry information must be known: 
  detector dimension, resolution, SOD, SDD, projection angles.
* Detector tilt y is currently assumed to be 0 (no roll).

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
  parameters".

Validation:

* The trial geometry built for the search starts with default tilt_x at 0
  and offset_u/offset_v at 0.
* The synthetic dataset's own geometry.config still carries the true
  tilt_x/offset_u/offset_v it was generated with (LoadTruthAndBuildParam
  reads them via VxManager.LoadConfig into `truth`, before zeroing them on
  the `param` the search actually uses). Those are read only after the
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
from SyntheticDataGenerator.Structures.micro_jig import MicroJigStructure

def LoadTruthAndBuildParam(config_path: str) -> tuple[VxParam, dict]:
    """
    Load the dataset's geometry via VxManager.LoadConfig (toLoadImages=False,
    geometry only - no need to touch the projection stack here) rather than
    re-parsing geometry.config by hand: this is the exact same parser the
    real reconstruction path uses, so field names, units and binning can't
    silently drift out of sync with it the way a second, hand-rolled parser
    could.

    Returns (param, truth): `param` is the loaded VxParam with tilt_x,
    offset_u and offset_v all reset to 0 - the unknowns this module
    estimates, deliberately blinded rather than left at the file's real
    values - and `truth` is what those three actually were, read once here
    and not touched again until EstimateTilt reports the final error
    against it.
    """
    param = VxManager().LoadConfig(config_path, toLoadImages=False)
    truth = dict(tilt_x=param.tilt_x, offset_u=param.offset_u, offset_v=param.offset_v)
    param.tilt_x, param.offset_u, param.offset_v = 0.0, 0.0, 0.0
    return param, truth

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

    Air is assumed to be the majority of the frame (true for a small part
    on a much larger detector), so its level is estimated as the frame's
    own median rather than assumed to be an exact 0. A real detector's air
    reading carries a noisy dark/offset floor - a handful of counts, not a
    clean zero - and a hardcoded ">0" test would then treat nearly every
    pixel as foreground, so the percentile below would be computed over
    background noise instead of the object, and the threshold would no
    longer separate ball hits from air at all.
    """
    background = np.median(frame)
    fg = frame[frame > background]
    if fg.size == 0:
        return np.empty((0, 2))
    threshold = np.percentile(fg, 80)
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


def Plot(dataset_root: str, result: dict, param: VxParam,
         balls_mm: np.ndarray, detections: dict[int, np.ndarray],
         frame_indices: list[int], files: list[str],
         max_gate_px: float = 25.0, save_path: str = None) -> None:
    """
    Visualise the optimisation rather than just printing its result: the
    coarse and fine cost-vs-tilt_x curves the grid search actually walked
    (with the estimated and ground-truth tilt marked on both), one sample
    frame with detected ball blobs overlaid against the fitted geometry's
    predicted ball positions, and the distribution of final matched pixel
    residuals across every kept frame - the same numbers EstimateTilt
    prints, shown so the fit can be seen rather than taken on faith.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"{dataset_root}  -  tilt_x / offset_u / offset_v estimation")

    est_t, truth_t = result["estimated_tilt_x"], result["truth_tilt_x"]

    ax = axes[0, 0]
    ax.plot(result["coarse_grid"], result["coarse_cost"], ".-", ms=3, color="tab:blue")
    ax.axvline(est_t, color="tab:green", ls="--", label=f"estimated {est_t:.2f} deg")
    ax.axvline(truth_t, color="tab:red", ls=":", label=f"truth {truth_t:.2f} deg")
    ax.set_xlabel("trial tilt_x (deg)"); ax.set_ylabel("mean matched error (px^2)")
    ax.set_title("Coarse grid search"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(result["fine_grid"], result["fine_cost"], ".-", ms=3, color="tab:blue")
    ax.axvline(est_t, color="tab:green", ls="--")
    ax.axvline(truth_t, color="tab:red", ls=":")
    ax.set_xlabel("trial tilt_x (deg)")
    ax.set_title("Fine grid search (zoom)")

    # Re-run the match at the converged (tilt, offset) to (a) overlay one
    # representative frame's detections against the fitted prediction and
    # (b) collect every kept frame's matched residuals for the histogram -
    # the same computation FitOffsetAndCost's last pass did, just kept
    # around here since that function only returns the summary MSE.
    param.tilt_x = est_t
    manager = VxManager()
    manager.SetParam(param)
    vecs = manager.GetScanGeometry()
    offset = np.array([result["estimated_offset_u"], result["estimated_offset_v"]])

    ax = axes[1, 0]
    sample_i = frame_indices[len(frame_indices) // 2]
    img = cv2.imread(files[sample_i], cv2.IMREAD_UNCHANGED)
    ax.imshow(img, cmap="gray")
    det = detections[sample_i]
    ax.scatter(det[:, 0], det[:, 1], s=70, facecolors="none",
               edgecolors="lime", linewidths=1.5, label="detected")

    all_resid_px = []
    for i in frame_indices:
        d = detections[i]
        if d.shape[0] == 0:
            continue
        pred = ProjectPoints(vecs[i], balls_mm, param.det_width, param.det_height) - offset
        cost = np.sum((d[:, None, :] - pred[None, :, :]) ** 2, axis=2)
        rows, cols = linear_sum_assignment(cost)
        d2 = cost[rows, cols]
        keep = d2 < max_gate_px ** 2
        all_resid_px.append(np.sqrt(d2[keep]))
        if i == sample_i:
            ax.scatter(pred[cols[keep], 0], pred[cols[keep], 1], marker="x", s=50,
                       c="red", linewidths=1.5, label="predicted (fitted)")

    ax.set_title(f"Frame {sample_i}: detected vs. fitted prediction")
    ax.set_xlim(0, param.det_width); ax.set_ylim(param.det_height, 0)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    resid = np.concatenate(all_resid_px) if all_resid_px else np.array([])
    if resid.size:
        ax.hist(resid, bins=20, color="tab:blue")
        ax.set_title(f"Final matched residuals, n={resid.size} "
                     f"(RMS={np.sqrt(np.mean(resid ** 2)):.2f} px)")
    else:
        ax.set_title("No matched residuals")
    ax.set_xlabel("matched pixel residual (px)"); ax.set_ylabel("count")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved diagnostic plot: {save_path}")
    else:
        plt.show()


def EstimateTilt(dataset_root: str, frame_stride: int = 2,
                  coarse_range=(-10.0, 110.0), coarse_step=1.0,
                  fine_half_width=2.0, fine_step=0.02,
                  min_good_frames: int = 8, verbose=True,
                  plot: bool = False, save_plot: str = None) -> dict:
    config_path = os.path.join(dataset_root, "Config", "geometry.config")
    corrected_dir = os.path.join(dataset_root, "Corrected")
    param, truth = LoadTruthAndBuildParam(config_path)
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
                                      param.det_width, param.det_height)
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
        param.det_width, param.det_height)

    if verbose:
        print(f"Fine search:   best tilt_x = {fine_best:.3f} deg "
              f"(cost={final_mse:.4f} px^2), total {time.time()-t0:.1f}s")
        print(f"Fitted detector offset: offset_u = {offset_u:+.3f} px, "
              f"offset_v = {offset_v:+.3f} px")
        print(f"Ground truth (from geometry.config, not used in the search): "
              f"tilt_x = {truth['tilt_x']:.3f} deg, "
              f"offset_u = {truth['offset_u']:+.3f} px, offset_v = {truth['offset_v']:+.3f} px")
        print(f"Error: tilt_x {fine_best - truth['tilt_x']:+.3f} deg, "
              f"offset_u {offset_u - truth['offset_u']:+.3f} px, "
              f"offset_v {offset_v - truth['offset_v']:+.3f} px")

    result = dict(estimated_tilt_x=fine_best, truth_tilt_x=truth["tilt_x"],
                  estimated_offset_u=offset_u, estimated_offset_v=offset_v,
                  truth_offset_u=truth["offset_u"], truth_offset_v=truth["offset_v"],
                  coarse_grid=coarse_grid, coarse_cost=coarse_cost,
                  fine_grid=fine_grid, fine_cost=fine_cost)

    if plot or save_plot:
        Plot(dataset_root, result, param, balls_mm,
             detections, frame_indices, files, save_path=save_plot)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=r"output\Synthetic\MICRO_JIG",
                         help="Dataset root containing Config/geometry.config and Corrected/*.tif")
    parser.add_argument("--stride", type=int, default=2,
                         help="Scan every Nth projection frame for ball blobs before quality-filtering")
    parser.add_argument("--min-good-frames", type=int, default=8,
                         help="Minimum near-complete-detection frames to keep for the fit")
    parser.add_argument("--plot", action="store_true",
                         help="Show a diagnostic matplotlib figure of the search + fit")
    parser.add_argument("--save-plot", default=None,
                         help="Save the diagnostic figure to this path instead of showing it")
    args = parser.parse_args()

    result = EstimateTilt(args.dataset, frame_stride=args.stride,
                           min_good_frames=args.min_good_frames,
                           plot=args.plot, save_plot=args.save_plot)