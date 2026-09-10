import itertools
import json
import os
import warnings
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
    BGA   = "BGA"    # die-on-substrate BGA joints with IPC-7095 void defects
    WLCSP = "WLCSP"  # wafer-level CSP: balls straight onto the die, no substrate
    PCB_PANEL = "PCB_PANEL"  # PCB panel matching the FID_2 reference (200um bumps, PTH vias)

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

    # ~12 GB of float32 before ASTRA allocates the sinogram; past this the
    # phantom is clamped and a warning is raised rather than the box silently
    # growing until the machine swaps.
    _MAX_PHANTOM_VOXELS = 3_000_000_000

    # Constructor
    def __init__(
        self,
        phantom: VxPhantomConstants = VxPhantomConstants.SOLID,
    ):
        self.PhantomKind = phantom
        self.Param = None
        self.AcquisitionParam = None
        self.PhantomParam = None
        self.PhantomSizeOverrideMm = None
        self.MaxPhantomVoxels = self._MAX_PHANTOM_VOXELS
        self.Volume = None
        self.Manifest = None
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
        phantom_size_mm: tuple[float, float, float] = None,
        max_phantom_voxels: int = None,
    ) -> VxParam:
        """
        Build the geometry the dataset needs and return the reconstruction
        param.

        det_width/det_height/det_pitch describe the UNBINNED detector the
        projections are stored at. Two params are derived from them: the
        reconstruction geometry (binned) and the acquisition geometry
        (unbinned, the stored detector), so a dataset is written full
        resolution and binned on load, as a real acquisition is. With
        binning=1 the two are the same object.

        `volume` is the RECONSTRUCTION volume. It is written to the config and
        used to crop the saved phantom, but it does not size the object being
        projected: a real board extends well past the reconstructed field of
        view, and a phantom that stops at it puts its own box wall in the
        projections. The phantom box is derived instead - see PhantomSizeMm:
        thickness from the phantom class, lateral extent solved from this
        geometry so the object covers the detector at every angle.

        phantom_size_mm overrides that solve when you want a specific extent
        (deliberate truncation, or a memory ceiling). max_phantom_voxels caps
        the solved size; past it the box is clamped and a warning raised.
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
        self.PhantomParam = None
        self.PhantomSizeOverrideMm = phantom_size_mm
        if max_phantom_voxels is not None:
            self.MaxPhantomVoxels = int(max_phantom_voxels)
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
        Write Corrected/, Config/geometry.config and phantom.npy under root,
        plus ground_truth.json when the phantom seeded defects.
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

        # phantom.npy is the RECONSTRUCTION field of view cut out of the
        # phantom, so it can be compared with a reconstruction directly. The
        # full box is larger by design - it holds the material outside the FOV
        # that the rays still pass through.
        np.save(os.path.join(root, "phantom.npy"), self.cropToFov_Internal())

        # phantoms that carry seeded defects also write what they seeded, so a
        # detector can be scored against ground truth rather than eyeballed
        if self.Manifest is not None:
            with open(os.path.join(root, "ground_truth.json"), "w", encoding="utf-8") as f:
                json.dump(self.Manifest, f, indent=2)
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
    def PhantomVoxelSize(self) -> float:
        """
        Voxel the phantom is built on: the ACQUISITION (unbinned) one.

        VoxelSize is the reconstruction voxel, which binning coarsens. Building
        the object at that scale and then projecting it onto the full-resolution
        detector throws away exactly the detail binning was meant to preserve on
        the way back, so the phantom is built at the detector's own scale.
        """
        acq = self.AcquisitionParam
        if acq is None:
            raise ValueError("No parameters: call SetParams first.")
        return acq.det_pitch * acq.sod / acq.sdd

    @property
    def PhantomSizeMm(self) -> tuple[float, float, float]:
        """
        Physical box the phantom is built in, (width, length, thickness) mm.

        Thickness comes from the phantom class - it is a property of the part.
        Lateral extent is solved from the geometry so the object covers the
        detector at every angle. Phantoms with no declared physical size
        (SOLID, SLAB - their features are fractions of the array) fall back to
        the reconstruction volume, which is the only size they have.
        """
        cls = self.phantomClass_Internal()
        default = getattr(cls, "DEFAULT_SIZE_MM", None) if cls else None
        if isinstance(default, dict):
            default = default[self.phantomMode_Internal()]

        if self.PhantomSizeOverrideMm is not None:
            return tuple(float(v) for v in self.PhantomSizeOverrideMm)

        p = self.Param
        if default is None:
            v = self.VoxelSize
            return (p.dst_x * v, p.dst_y * v, p.dst_z * v)

        thickness = float(default[2])
        lateral = self.solveCoverage_Internal(thickness)
        if lateral is None:
            warnings.warn(
                "No phantom size covers the detector at this geometry (a flat "
                "object goes edge-on in CT geometry); falling back to the "
                "phantom's default extent. Expect the box edge in the "
                "projections.", stacklevel=2)
            lateral = float(default[0])

        voxel = self.PhantomVoxelSize
        count = (lateral / voxel) ** 2 * (thickness / voxel)
        if count > self.MaxPhantomVoxels:
            capped = (self.MaxPhantomVoxels * voxel ** 3 / thickness) ** 0.5
            warnings.warn(
                f"Phantom needs {lateral:.2f} mm laterally to cover the "
                f"detector ({count:,.0f} voxels) but max_phantom_voxels allows "
                f"{self.MaxPhantomVoxels:,}; clamping to {capped:.2f} mm. The "
                f"object edge will appear in the projections.", stacklevel=2)
            lateral = capped
        return (lateral, lateral, thickness)

    @property
    def PhantomVolume(self) -> tuple[int, int, int]:
        """Phantom grid in voxels, (nz, ny, nx) - ASTRA order."""
        w, l, t = self.PhantomSizeMm
        v = self.PhantomVoxelSize
        return (max(1, round(t / v)), max(1, round(l / v)), max(1, round(w / v)))

    def MinimumPhantomVolume(self) -> tuple[float, float, float]:
        """
        Smallest phantom box that keeps its own edge out of every projection,
        (width, length, thickness) mm. None when no size achieves it.
        """
        cls = self.phantomClass_Internal()
        default = getattr(cls, "DEFAULT_SIZE_MM", None) if cls else None
        if isinstance(default, dict):
            default = default[self.phantomMode_Internal()]
        p = self.Param
        thickness = float(default[2]) if default else p.dst_z * self.VoxelSize
        lateral = self.solveCoverage_Internal(thickness)
        return None if lateral is None else (lateral, lateral, thickness)

    @property
    def NumProjections(self) -> int:
        return self.requireParams_Internal().num_of_imgs

    def Describe(self) -> str:
        """One-line summary of what was generated, for progress output."""
        p = self.requireParams_Internal()
        acq = self.AcquisitionParam
        voxel = self.VoxelSize
        vol = self.Volume.shape if self.Volume is not None else "-"
        box = "x".join(f"{v:.2f}" for v in self.PhantomSizeMm) + "mm"
        extent = (f"{p.dst_x * voxel:.2f}x{p.dst_y * voxel:.2f}x{p.dst_z * voxel:.2f}mm"
                  if self.Volume is not None else "-")
        rng = (f"[{self.Sinogram.min():.3f}, {self.Sinogram.max():.3f}]"
               if self.Sinogram is not None else "-")
        return (f"tilt={p.tilt_x:g} {str(p.laminography_method):9s} "
                f"det {int(acq.det_width)}x{int(acq.det_height)} -> "
                f"{int(p.det_width)}x{int(p.det_height)} binned  "
                f"voxel={voxel:.4f}mm  phantom={vol} ({box})  "
                f"fov={extent}  proj range={rng}")

    # private / internal
    def requireParams_Internal(self) -> VxParam:
        if self.Param is None:
            raise ValueError("No parameters: call SetParams first.")
        return self.Param

    def cropToFov_Internal(self) -> np.ndarray:
        """
        The reconstruction volume cut out of the phantom box, at the
        reconstruction voxel. Both are centred on the origin, so the crop is
        centred too, offset by volume_mid; binning is applied by block-mean so
        the result matches the shape a reconstruction produces.
        """
        p = self.Param
        vol = self.Volume
        nz, ny, nx = vol.shape
        pv, rv = self.PhantomVoxelSize, self.VoxelSize
        ratio = max(1, int(round(rv / pv)))

        out = []
        for n, dst, mid in ((nx, p.dst_x, p.volume_mid_x),
                            (ny, p.dst_y, p.volume_mid_y),
                            (nz, p.dst_z, p.volume_mid_z)):
            want = int(round(dst)) * ratio
            start = int(round(n / 2.0 + (mid - dst * rv / 2.0) / pv))
            start = max(0, min(start, max(0, n - want)))
            out.append((start, min(n, start + want)))
        (x0, x1), (y0, y1), (z0, z1) = out
        cut = vol[z0:z1, y0:y1, x0:x1]

        if ratio > 1:
            kz, ky, kx = (d // ratio * ratio for d in cut.shape)
            cut = (cut[:kz, :ky, :kx]
                   .reshape(kz // ratio, ratio, ky // ratio, ratio, kx // ratio, ratio)
                   .mean(axis=(1, 3, 5)))
        return np.ascontiguousarray(cut, dtype=np.float32)

    def phantomMode_Internal(self) -> str:
        return "WLCSP" if self.PhantomKind == VxPhantomConstants.WLCSP else "FCBGA"

    def phantomClass_Internal(self):
        """The Structures class backing PhantomKind, or None for the relative
        phantoms (SOLID / SLAB) that have no physical size of their own."""
        k = self.PhantomKind
        if k == VxPhantomConstants.PCB_PANEL:
            from SyntheticDataGenerator.Structures.pcb_panel import PcbPanelStructure
            return PcbPanelStructure
        if k in (VxPhantomConstants.BGA, VxPhantomConstants.WLCSP):
            from SyntheticDataGenerator.Structures.bga import BgaStructure
            return BgaStructure
        if k == VxPhantomConstants.HBM:
            from SyntheticDataGenerator.Structures.hbm import HBMStructure
            return HBMStructure
        return None

    def solveCoverage_Internal(self, thickness_mm: float):
        """
        Smallest lateral extent (mm) whose box silhouette covers the whole
        detector at every projection angle, by bisection on the scan vectors.

        None when no size does: with an upright detector (CT geometry) a flat
        object turns edge-on and its projection collapses to a line, so no
        finite board ever fills the frame.
        """
        from scipy.spatial import ConvexHull

        acq = self.AcquisitionParam
        vecs = np.asarray(self.getVectors_Internal(), dtype=np.float64).reshape(-1, 12)
        hu, hv = acq.det_width / 2.0, acq.det_height / 2.0
        det = np.array([[-hu, -hv], [hu, -hv], [hu, hv], [-hu, hv]])
        hz = thickness_mm / 2.0

        def covers(half_xy: float) -> bool:
            corners = np.array(list(itertools.product(
                (-half_xy, half_xy), (-half_xy, half_xy), (-hz, hz))))
            for row in vecs:
                S, D, U, V = row[0:3], row[3:6], row[6:9], row[9:12]
                n = np.cross(U, V)
                d = corners - S
                den = d @ n
                if np.any(np.abs(den) < 1e-12):
                    return False
                t = np.dot(D - S, n) / den
                w = S + t[:, None] * d - D
                pts = np.stack([(w @ U) / (U @ U), (w @ V) / (V @ V)], axis=1)
                eq = ConvexHull(pts).equations          # A.x + b <= 0 inside
                if np.any(eq[:, :2] @ det.T + eq[:, 2:3] > 1e-9):
                    return False
            return True

        voxel = self.PhantomVoxelSize
        hi = max(acq.det_width, acq.det_height) * voxel * 2.0
        if not covers(hi):
            return None
        lo = voxel
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if covers(mid):
                hi = mid
            else:
                lo = mid
        return 2.0 * hi

    def buildPhantom_Internal(self) -> np.ndarray:
        """Phantom in ASTRA (z, y, x) order, on the derived phantom grid."""
        shape = self.PhantomVolume
        vs = self.PhantomVoxelSize

        if self.PhantomKind == VxPhantomConstants.PCB_PANEL:
            from SyntheticDataGenerator.Structures.pcb_panel import PcbPanelStructure
            vol, self.Manifest = PcbPanelStructure.Build(shape, vs)
            return vol
        if self.PhantomKind in (VxPhantomConstants.BGA, VxPhantomConstants.WLCSP):
            from SyntheticDataGenerator.Structures.bga import BgaStructure
            mode = "FCBGA" if self.PhantomKind == VxPhantomConstants.BGA else "WLCSP"
            vol, self.Manifest = BgaStructure.Build(shape, vs, mode=mode)
            return vol
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

        acq = self.AcquisitionParam

        # The phantom box is centred on the origin, not on volume_mid: the
        # object sits where it sits, and volume_mid only says which part of it
        # the reconstruction later looks at.
        nz, ny, nx = self.PhantomVolume
        voxel = self.PhantomVoxelSize
        half_x = nx * voxel / 2.0
        half_y = ny * voxel / 2.0
        half_z = nz * voxel / 2.0
        vol_geom = astra.create_vol_geom(
            ny, nx, nz,
            -half_x, half_x, -half_y, half_y, -half_z, half_z,
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
VolX = {int(acq.det_width)}
VolY = {int(acq.det_width)}
VolZ = {int(acq.det_height)}
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
ReconType = FDK_Performance
Iterations = {p.iterations}
IsLaminoStaticSourceGeometry = False
IsDetectorParallelToFOV = {p.laminography_method == LaminographyMethodConstants.COPLANAR}
"""
