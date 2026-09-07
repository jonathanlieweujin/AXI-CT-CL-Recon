import os
from enum import Enum

import cv2
import numpy as np

from Manager import VxManager
from Manager.Param import VxParam
from Manager.Constants.laminography_method import LaminographyMethodConstants


class VxPhantomConstants(str, Enum):
    """Phantom shapes available to the synthetic data generator."""

    SOLID = "SOLID"
    SLAB  = "SLAB"
    HBM   = "HBM"    # HBM high-mag board cross-section (μbumps / C4 / BGA)

    def __str__(self) -> str:
        return self.value


class VxSyntheticDataGenerator:
    """
    Single entry point for generating a synthetic dataset: builds the parameter
    pair, the phantom, the projections that geometry would produce, and a
    Corrected/ + Config/ folder pair matching the layout of a real acquisition.

    Projections are forward-projected with the same geometry the recon flows
    rebuild, so a correct backend reconstructs the phantom that produced them.

    The mirror of VxManager on the reconstruction side - construct, set the
    params, generate, save - so a dataset is produced without the caller
    assembling VxParam objects or knowing the step order.
    """

    # Constructor
    def __init__(
        self,
        phantom: VxPhantomConstants = VxPhantomConstants.SOLID,
    ):
        self.PhantomKind = phantom
        self.Param = None
        self.AcquisitionParam = None
        self.Volume = None
        # kept in ASTRA order (det_v, n_angles, det_u): the stack is far too
        # large at full detector resolution to also hold a transposed copy.
        self.Sinogram = None

    # Public
    def SetParams(
        self,
        angles: np.ndarray,
        sod: float,
        sdd: float,
        det_width: int,
        det_height: int,
        det_pitch: float,
        volume: tuple[int, int, int],
        tilt_x: float = 90.0,
        tilt_y: float = 0.0,
        binning: int = 1,
        offset_u: float = 0.0,
        offset_v: float = 0.0,
        volume_mid: tuple[float, float, float] = (0.0, 0.0, 0.0),
        laminography_method: LaminographyMethodConstants = LaminographyMethodConstants.INCLINED,
        iterations: int = 1,
    ) -> VxParam:
        """
        Build the geometry pair the dataset needs and return the reconstruction
        param.

        det_width/det_height/det_pitch describe the UNBINNED detector the
        projections are stored at. Two params are derived from them: the
        reconstruction geometry (binned, and the volume grid the phantom lives
        on) and the acquisition geometry (unbinned, the stored detector), so a
        dataset is written full resolution and binned on load, as a real
        acquisition is. With binning=1 the two are the same object.
        """
        vol_x, vol_y, vol_z = volume
        mid_x, mid_y, mid_z = volume_mid

        def make(b):
            return VxParam(
                sod=sod, sdd=sdd, angles=angles,
                tilt_x=tilt_x, tilt_y=tilt_y,
                det_width=det_width, det_height=det_height, det_pitch=det_pitch,
                offset_u=offset_u, offset_v=offset_v, binning=b,
                volume_mid_x=mid_x, volume_mid_y=mid_y, volume_mid_z=mid_z,
                dst_x=vol_x, dst_y=vol_y, dst_z=vol_z,
                laminography_method=laminography_method, iterations=iterations,
            )

        self.Param = make(binning)
        # binning=1 must still yield ONE object, not two equal ones: the config
        # keys "has binning already been applied" off `acq is not param`.
        self.AcquisitionParam = self.Param if binning == 1 else make(1)
        self.Volume = None
        self.Sinogram = None
        return self.Param

    def Run(self) -> np.ndarray:
        """
        Build the phantom and forward project it.

        The phantom is left on Volume and the projections on Sinogram, in ASTRA
        (det_v, angles, det_u) order.
        """
        self.requireParams_Internal()
        self.Volume = self.buildPhantom_Internal()
        self.Sinogram = self.project_Internal(self.Volume)
        return self.Sinogram

    def Save(self, root: str) -> str:
        """
        Write Corrected/, Config/geometry.config and phantom.npy under root.
        Projects first if Run has not been called.
        """
        self.requireParams_Internal()
        if self.Sinogram is None:
            self.Run()

        corrected = os.path.join(root, "Corrected")
        config_dir = os.path.join(root, "Config")
        os.makedirs(corrected, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)

        # slice per angle out of the ASTRA-order stack rather than transposing
        # the whole thing, which would double peak memory
        for i in range(self.NumProjections):
            frame = np.ascontiguousarray(self.Sinogram[:, i, :], dtype=np.float32)
            cv2.imwrite(os.path.join(corrected, f"proj_{i:04d}.tif"), frame)

        with open(os.path.join(config_dir, "geometry.config"), "w", encoding="utf-8") as f:
            f.write(self.buildConfig_Internal())

        np.save(os.path.join(root, "phantom.npy"), self.Volume)
        return root

    def Generate(self, root: str) -> str:
        """Run and Save in one call; returns root."""
        self.Run()
        return self.Save(root)

    @property
    def VoxelSize(self) -> float:
        p = self.requireParams_Internal()
        return p.det_pitch * p.sod / p.sdd

    @property
    def NumProjections(self) -> int:
        return self.requireParams_Internal().num_of_imgs

    def Describe(self) -> str:
        """One-line summary of what was generated, for progress output."""
        p = self.requireParams_Internal()
        acq = self.AcquisitionParam
        voxel = self.VoxelSize
        vol = self.Volume.shape if self.Volume is not None else "-"
        extent = (f"{p.dst_x * voxel:.2f}x{p.dst_y * voxel:.2f}x{p.dst_z * voxel:.2f}mm"
                  if self.Volume is not None else "-")
        rng = (f"[{self.Sinogram.min():.3f}, {self.Sinogram.max():.3f}]"
               if self.Sinogram is not None else "-")
        return (f"tilt={p.tilt_x:g} {str(p.laminography_method):9s} "
                f"det {int(acq.det_width)}x{int(acq.det_height)} -> "
                f"{int(p.det_width)}x{int(p.det_height)} binned  "
                f"voxel={voxel:.4f}mm  vol={vol} ({extent})  proj range={rng}")

    # private / internal
    def requireParams_Internal(self) -> VxParam:
        if self.Param is None:
            raise ValueError("No parameters: call SetParams first.")
        return self.Param

    def buildPhantom_Internal(self) -> np.ndarray:
        """Phantom in ASTRA (z, y, x) order."""
        p = self.Param
        vol_x, vol_y, vol_z = int(p.dst_x), int(p.dst_y), int(p.dst_z)
        shape = (vol_z, vol_y, vol_x)
        vs = self.VoxelSize

        if self.PhantomKind == VxPhantomConstants.HBM:
            from SyntheticDataGenerator.Structures.hbm import HBMStructure
            return HBMStructure.GetStructure(shape, vs)
        if self.PhantomKind == VxPhantomConstants.SOLID:
            from SyntheticDataGenerator.Structures.solid import SolidStructure
            return SolidStructure.GetStructure(shape, vs)
        from SyntheticDataGenerator.Structures.slab import SlabStructure
        return SlabStructure.GetStructure(shape, vs)

    def getVectors_Internal(self) -> np.ndarray:
        # Built from the acquisition geometry: the U/V vectors carry the
        # detector pitch the projections are stored at.
        return VxManager(self.AcquisitionParam).GetScanGeometry()

    def project_Internal(self, phantom: np.ndarray) -> np.ndarray:
        """Forward project. Returns the sinogram in ASTRA (det_v, angles, det_u) order."""
        import astra

        p = self.Param
        acq = self.AcquisitionParam

        voxel = self.VoxelSize
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
            'cone_vec', int(acq.det_height), int(acq.det_width), self.getVectors_Internal())

        proj_id, proj = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom)
        astra.data3d.delete(proj_id)
        return proj

    def buildConfig_Internal(self) -> str:
        """
        geometry.config describing this dataset.

        The detector fields describe the resolution the projections are STORED
        at, paired with the binning that reproduces `Param` when the config is
        read back. Without a separate acquisition geometry the stored resolution
        is already binned, so Binning is 1 - writing the original factor there
        would apply it a second time.
        """
        p = self.Param
        acq = self.AcquisitionParam
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
