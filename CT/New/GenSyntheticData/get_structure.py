import os
from enum import Enum

import cv2
import numpy as np

from Util.param import VxParam
from Util.compute_geometry import VxComputeGeometry
from Util.laminography_method import LaminographyMethodConstants


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
        laminography_method: LaminographyMethodConstants = LaminographyMethodConstants.INCLINED,
        phantom: VxPhantomConstants = VxPhantomConstants.SOLID,
    ):
        self._param = param
        self._laminography_method = laminography_method
        self._phantom_kind = phantom
        self.phantom = None
        self.projections = None

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

    def _get_vectors(self) -> np.ndarray:
        if self._laminography_method == LaminographyMethodConstants.COPLANAR:
            return VxComputeGeometry.computeCoplanarTranslationalLaminographyGeometry(
                self._param).reshape(-1, 12)
        return VxComputeGeometry.computeDefaultInclinedLaminographyGeometry(
            self._param).reshape(-1, 12)

    def project(self, phantom: np.ndarray = None) -> np.ndarray:
        """Forward project to (angles, det_v, det_u), the layout VxTool.run expects."""
        import astra

        p = self._param
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
            'cone_vec', p.det_height, p.det_width, self._get_vectors())

        proj_id, proj = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom)
        astra.data3d.delete(proj_id)

        self.projections = np.ascontiguousarray(
            np.transpose(proj, (1, 0, 2)).astype(np.float32))
        return self.projections

    def build_config(self) -> str:
        """
        geometry.config describing this dataset.

        VxParam already divided the detector geometry by its binning factor, so
        the config is written in those post-binned terms with Binning = 1.
        Writing the original binning back would apply it a second time when the
        config is read into a fresh VxParam.
        """
        p = self._param
        angles = ", ".join(f"{a:g}" for a in np.asarray(p.angles))
        return f"""[VXMPR CONFIG]

DetU = {int(p.det_width)}
DetV = {int(p.det_height)}
DetPitch = {p.det_pitch:g}
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
Binning = 1
ProjectionAngles = [{angles}]

SOD = {p.sod:g}
SDD = {p.sdd:g}

DetTiltX = {p.tilt_x:g}
DetTiltY = {p.tilt_y:g}

DetOffsetU = {p.offset_u:g}
DetOffsetV = {p.offset_v:g}

LeftPad = 0
RightPad = 0

Filter = SheppLogan
ReconType = FDK
Iterations = {p.iterations}
LaminographyMethod = {self._laminography_method}
"""

    def write(self, root: str) -> str:
        """Write Corrected/, Config/geometry.config and phantom.npy under root."""
        if self.projections is None:
            self.project()

        corrected = os.path.join(root, "Corrected")
        config_dir = os.path.join(root, "Config")
        os.makedirs(corrected, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)

        for i in range(self.projections.shape[0]):
            cv2.imwrite(os.path.join(corrected, f"proj_{i:04d}.tif"), self.projections[i])

        with open(os.path.join(config_dir, "geometry.config"), "w", encoding="utf-8") as f:
            f.write(self.build_config())

        np.save(os.path.join(root, "phantom.npy"), self.phantom)
        return root
