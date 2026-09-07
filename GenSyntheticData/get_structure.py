import os
from enum import Enum

import cv2
import numpy as np

from Manager import VxManager
from Manager.Param import VxParam
# from Manager.Constants.laminography_method import LaminographyMethodConstants


class VxPhantomConstants(str, Enum):
    """Phantom shapes available to the synthetic data generator."""

    SOLID = "SOLID"
    SLAB = "SLAB"

    def __str__(self) -> str:
        return self.value


class VxSyntheticData:
    """
    Builds a synthetic dataset from a VxParam: a phantom, the projections that
    geometry would produce, and a Corrected/ + Config/ folder pair matching the
    layout of a real acquisition.

    Projections are forward-projected with the same geometry the recon flows
    rebuild, so a correct backend reconstructs the phantom that produced them.
    """

    def __init__(
        self,
        param: VxParam,
        phantom: VxPhantomConstants = VxPhantomConstants.SOLID,
        acquisition_param: VxParam = None,
    ):
        """
        param            the reconstruction geometry (post-binning), and the
                         volume grid the phantom lives on.
        acquisition_param optional unbinned geometry describing the detector the
                         projections are STORED at. Give this when the dataset
                         should be written full-resolution and binned on load,
                         as a real acquisition is. The volume grid always comes
                         from `param`; only the detector differs.

        The laminography method is read from the params, so the geometry the
        data is projected through cannot drift from the one it declares.
        """
        self._param = param
        self._acq = acquisition_param if acquisition_param is not None else param
        if self._acq.laminography_method != param.laminography_method:
            raise ValueError(
                "acquisition_param and param disagree on laminography_method: "
                f"{self._acq.laminography_method} vs {param.laminography_method}")
        self._phantom_kind = phantom
        self.phantom = None
        # kept in ASTRA order (det_v, n_angles, det_u): the stack is far too
        # large at full detector resolution to also hold a transposed copy.
        self._sinogram = None

    @property
    def voxel_size(self) -> float:
        p = self._param
        return p.det_pitch * p.sod / p.sdd

    def build_phantom(self) -> np.ndarray:
        """Phantom in ASTRA (z, y, x) order."""
        p = self._param
        vol_x, vol_y, vol_z = int(p.dst_x), int(p.dst_y), int(p.dst_z)
        vol = np.zeros((vol_z, vol_y, vol_x), dtype=np.float32)
        z, y, x = np.meshgrid(np.arange(vol_z), np.arange(vol_y), np.arange(vol_x), indexing="ij")
        cz, cy, cx = vol_z / 2.0, vol_y / 2.0, vol_x / 2.0

        if self._phantom_kind == VxPhantomConstants.SOLID:
            r = min(vol_x, vol_y, vol_z) * 0.34
            vol[(z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 < r ** 2] = 1.0
            # off-centre rod, so a mirrored reconstruction cannot score as a match
            r2 = min(vol_x, vol_y) * 0.12
            vol[(y - cy) ** 2 + (x - cx - vol_x * 0.16) ** 2 < r2 ** 2] = 2.0
            vol[int(cz - vol_z * 0.30):int(cz - vol_z * 0.18),
                int(cy - vol_y * 0.22):int(cy + vol_y * 0.22),
                int(cx - vol_x * 0.30):int(cx - vol_x * 0.10)] = 1.6
        else:
            # thin slab with in-plane structure: the laminography use case
            z0, z1 = int(cz - vol_z * 0.22), int(cz + vol_z * 0.22)
            vol[z0:z1, int(vol_y * 0.18):int(vol_y * 0.82),
                int(vol_x * 0.18):int(vol_x * 0.82)] = 0.6
            for k in range(4):
                yk = int(vol_y * (0.28 + 0.14 * k))
                vol[z0:z1, yk:yk + 4, int(vol_x * 0.24):int(vol_x * 0.76)] = 1.8
            r = min(vol_x, vol_y) * 0.07
            disc = (y - cy) ** 2 + (x - cx + vol_x * 0.22) ** 2 < r ** 2
            disc[:z0] = False
            disc[z1:] = False
            vol[disc] = 2.4

        self.phantom = vol
        return vol

    @property
    def n_projections(self) -> int:
        return self._param.num_of_imgs

    def _get_vectors(self) -> np.ndarray:
        # Built from the acquisition geometry: the U/V vectors carry the
        # detector pitch the projections are stored at.
        return VxManager(self._acq).GetScanGeometry()

    def project(self, phantom: np.ndarray = None) -> np.ndarray:
        """Forward project. Returns the sinogram in ASTRA (det_v, angles, det_u) order."""
        import astra

        p = self._param
        acq = self._acq
        if phantom is None:
            phantom = self.phantom if self.phantom is not None else self.build_phantom()

        voxel = self.voxel_size
        half_x = p.dst_x * voxel / 2.0
        half_y = p.dst_y * voxel / 2.0
        half_z = p.dst_z * voxel / 2.0
        vol_geom = astra.create_vol_geom(
            p.dst_y, p.dst_x, p.dst_z,
            -half_x + p.volume_mid_x, half_x + p.volume_mid_x,
            -half_y + p.volume_mid_y, half_y + p.volume_mid_y,
            -half_z + p.volume_mid_z, half_z + p.volume_mid_z,
        )
        proj_geom = astra.create_proj_geom(
            'cone_vec', int(acq.det_height), int(acq.det_width), self._get_vectors())

        proj_id, proj = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom)
        astra.data3d.delete(proj_id)

        self._sinogram = proj
        return self._sinogram

    def build_config(self) -> str:
        """
        geometry.config describing this dataset.

        The detector fields describe the resolution the projections are STORED
        at, paired with the binning that reproduces `param` when the config is
        read back. Without an acquisition_param the stored resolution is already
        binned, so Binning is 1 - writing the original factor there would apply
        it a second time.
        """
        p = self._param
        acq = self._acq
        binning = p.binning if acq is not p else 1
        angles = ", ".join(f"{a:g}" for a in np.asarray(p.angles))
        return f"""[VXMPR CONFIG]

DetU = {int(acq.det_width)}
DetV = {int(acq.det_height)}
DetPitch = {acq.det_pitch:g}
ProjectionImages = {p.num_of_imgs}

VolScale = 1
VolX = {int(p.dst_x)}
VolY = {int(p.dst_y)}
VolZ = {int(p.dst_z)}
DstPixelFormat = U8C1
VolMidX = {p.volume_mid_x:g}
VolMidY = {p.volume_mid_y:g}
VolMidZ = {p.volume_mid_z:g}

Interval = 1
Binning = {binning}
ProjectionAngles = [{angles}]

SOD = {p.sod:g}
SDD = {p.sdd:g}

DetTiltX = {p.tilt_x:g}
DetTiltY = {p.tilt_y:g}

DetOffsetU = {acq.offset_u:g}
DetOffsetV = {acq.offset_v:g}

LeftPad = 0
RightPad = 0

Filter = SheppLogan
ReconType = FDK
Iterations = {p.iterations}
"""

    def write(self, root: str) -> str:
        """Write Corrected/, Config/geometry.config and phantom.npy under root."""
        if self._sinogram is None:
            self.project()

        corrected = os.path.join(root, "Corrected")
        config_dir = os.path.join(root, "Config")
        os.makedirs(corrected, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)

        # slice per angle out of the ASTRA-order stack rather than transposing
        # the whole thing, which would double peak memory
        for i in range(self.n_projections):
            frame = np.ascontiguousarray(self._sinogram[:, i, :], dtype=np.float32)
            cv2.imwrite(os.path.join(corrected, f"proj_{i:04d}.tif"), frame)

        with open(os.path.join(config_dir, "geometry.config"), "w", encoding="utf-8") as f:
            f.write(self.build_config())

        np.save(os.path.join(root, "phantom.npy"), self.phantom)
        return root
