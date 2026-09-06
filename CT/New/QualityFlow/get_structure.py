import copy

import numpy as np
import tigre
import tigre.algorithms as algs
from tigre.utilities.common_geometry import ArbitrarySourceDetMoveGeo
from Util.param import VxParam
from Util.recon_method import ReconMethodConstants
from Util.laminography_method import LaminographyMethodConstants
from Util.compute_geometry import VxComputeGeometry

def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

# Maps the Astra canonical detector frame (u=ex, v=-ey, source=-ez) onto
# TIGRE's beam frame (u=+y, v=+z, source=+x): ex->ey, ey->-ez, ez->-ex.
_B = np.array([
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
])

class VxGeom:
    """Builds TIGRE geometry from VxParam, mirroring ComputeAnglesStruct2."""

    def __init__(self, param: VxParam):
        self.param = param

    def _rig_rotation(self) -> np.ndarray:
        """
        (N, 3, 3) rig rotation stack, identical rows to PerformanceFlow's
        computeDefaultInclinedLaminographyGeometry: R = Rx(-tilt_x) · Ry(tilt_y) · Rz(phi),
        phi = -deg2rad(angles).
        """
        p = self.param
        phi = -np.deg2rad(p.angles)
        sin_tx, cos_tx = np.sin(np.deg2rad(p.tilt_x)), np.cos(np.deg2rad(p.tilt_x))
        sin_ty, cos_ty = np.sin(np.deg2rad(p.tilt_y)), np.cos(np.deg2rad(p.tilt_y))
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)

        R = np.empty((p.num_of_imgs, 3, 3))
        R[:, 0, 0] = cos_ty * cos_phi
        R[:, 0, 1] = cos_ty * sin_phi
        R[:, 0, 2] = -sin_ty
        R[:, 1, 0] = sin_tx * sin_ty * cos_phi - cos_tx * sin_phi
        R[:, 1, 1] = sin_tx * sin_ty * sin_phi + cos_tx * cos_phi
        R[:, 1, 2] = sin_tx * cos_ty
        R[:, 2, 0] = cos_tx * sin_ty * cos_phi + sin_tx * sin_phi
        R[:, 2, 1] = cos_tx * sin_ty * sin_phi - sin_tx * cos_phi
        R[:, 2, 2] = cos_tx * cos_ty
        return R

    def _zyz_angles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Per-projection ZYZ Euler angles (a0, a1, a2) such that TIGRE's rig
        rotation Rz(-a2)·Ry(-a1)·Rz(-a0) equals B·R_rig exactly, for any
        tilt_x / tilt_y. Generalizes the closed form
        [-(angles+90), tilt_x+90, tilt_y], which is only exact for tilt_y=0.

        Decomposition of D = Rz(C)·Ry(-b)·Rz(A), b = arccos(D22) in [0, pi]
        (branch chosen so a1 stays positive, matching the old convention);
        then (a0, a1, a2) = (-A, b, -C).
        """
        D = _B @ self._rig_rotation()          # (N, 3, 3)
        b = np.arccos(np.clip(D[:, 2, 2], -1.0, 1.0))
        regular = np.sin(b) > 1e-9
        A = np.where(regular, np.arctan2(-D[:, 2, 1], D[:, 2, 0]), 0.0)
        C = np.where(regular,
                     np.arctan2(-D[:, 1, 2], -D[:, 0, 2]),
                     # gimbal lock (tilt_x ~ -90): D is a pure z-rotation
                     np.arctan2(D[:, 1, 0], D[:, 0, 0]))
        return -A, b, -C

    # def get_default_angles(self) -> np.ndarray:
    #     a0, a1, a2 = self._zyz_angles()
    #     return np.column_stack([a0, a1, a2]).astype(np.float32)

    def _make_geometry(self, planar_axis_order: bool = False) -> tigre.geometry:
        p = self.param
        voxel_size = p.det_pitch * p.sod / p.sdd
        off_det_u_mm = p.offset_u * p.det_pitch
        off_det_v_mm = p.offset_v * p.det_pitch

        geo = tigre.geometry()
        geo.DSD = p.sdd
        geo.DSO = p.sod

        geo.nDetector = np.array([p.det_height, p.det_width], dtype=np.int32)
        geo.dDetector = np.array([p.det_pitch, p.det_pitch], dtype=np.float32)
        geo.sDetector = geo.nDetector * geo.dDetector

        if p.dst_x > 0 and p.dst_y > 0 and p.dst_z > 0:
            if planar_axis_order:
                # ASTRA planar (x,y,z) is represented in TIGRE as (y,z,x).
                # TIGRE arrays are [Tz, Ty, Tx], so allocate [Ay, Ax, Az].
                geo.nVoxel = np.array([p.dst_y, p.dst_x, p.dst_z], dtype=np.int32)
            else:
                geo.nVoxel = np.array([p.dst_z, p.dst_y, p.dst_x], dtype=np.int32)
        else:
            n = int(round(p.det_width * p.sod / p.sdd))
            geo.nVoxel = np.array([n, n, n], dtype=np.int32)

        geo.dVoxel = np.array([voxel_size, voxel_size, voxel_size], dtype=np.float32)
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        if planar_axis_order:
            geo.offOrigin = np.array([p.volume_mid_y, p.volume_mid_x, p.volume_mid_z], dtype=np.float32)
        else:
            geo.offOrigin = np.array([p.volume_mid_z, p.volume_mid_y, p.volume_mid_x], dtype=np.float32)
        geo.offDetector = np.array([off_det_v_mm, off_det_u_mm], dtype=np.float32)
        geo.accuracy = 0.5
        geo.mode = 'cone'
        return geo

    # def get_default_geometry(self) -> tigre.geometry:
    #     return self._make_geometry(planar_axis_order=False)

    @staticmethod
    def _astra_to_tigre_points(points: np.ndarray) -> np.ndarray:
        # ASTRA planar axes: U=+x, V=+y, source=+z. TIGRE: source=+x, U=+y, V=+z.
        return np.column_stack([points[:, 2], points[:, 0], points[:, 1]])

    @staticmethod
    def _flat_detector_rotation(angles: np.ndarray) -> np.ndarray:
        desired_u = np.array([0.0, 1.0, 0.0])
        desired_v = np.array([0.0, 0.0, 1.0])
        rot = np.zeros((angles.shape[0], 3), dtype=np.float64)
        for i, (a0, a1, a2) in enumerate(angles):
            E = _rot_z(-a2) @ _rot_y(-a1) @ _rot_z(-a0)
            u = E @ desired_u
            v = E @ desired_v
            Wt = np.column_stack([np.cross(u, v), u, v])
            rot[i, 2] = np.arctan2(Wt[1, 0], Wt[0, 0])
            rot[i, 1] = np.arcsin(np.clip(-Wt[2, 0], -1.0, 1.0))
            rot[i, 0] = np.arctan2(Wt[2, 1], Wt[2, 2])
        return rot.astype(np.float32)

    def get_planar_geometry_and_angles(self, laminography_method: LaminographyMethodConstants) -> tuple[tigre.geometry, np.ndarray]:
        if laminography_method == LaminographyMethodConstants.COPLANAR:
            vecs = VxComputeGeometry.computeCoplanarTranslationalLaminographyGeometry(self.param).reshape(-1, 12)
        else:
            vecs = VxComputeGeometry.computeDefaultInclinedLaminographyGeometry(self.param).reshape(-1, 12)
        
        source = self._astra_to_tigre_points(vecs[:, 0:3])
        detector = self._astra_to_tigre_points(vecs[:, 3:6])

        geo = self._make_geometry(planar_axis_order=True)
        geo.offDetector = np.zeros((self.param.num_of_imgs, 2), dtype=np.float32)
        geo.rotDetector = np.zeros((self.param.num_of_imgs, 3), dtype=np.float32)
        geo = ArbitrarySourceDetMoveGeo(geo, source, detector)
        geo.offDetector[np.abs(geo.offDetector) < 1e-6] = 0.0
        geo.rotDetector = self._flat_detector_rotation(geo.angles)
        geo.COR = np.zeros(self.param.num_of_imgs, dtype=np.float32)

        return geo, geo.angles.astype(np.float32)

    def get_planar_angles(self) -> np.ndarray:
        return self.get_planar_geometry_and_angles()[1]

    def get_planar_geometry(self) -> tigre.geometry:
        return self.get_planar_geometry_and_angles()[0]

class VxTool:
    """Runs a TIGRE reconstruction algorithm given projections and a VxParam."""

    _ITERATIVE_ALGORITHMS = {
        # Gradient / ART family
        ReconMethodConstants.SART: algs.sart,
        ReconMethodConstants.OS_SART: algs.ossart,
        ReconMethodConstants.OSSART: algs.ossart,
        ReconMethodConstants.SIRT: algs.sirt,
        ReconMethodConstants.ASD_POCS: algs.asd_pocs,
        ReconMethodConstants.OS_ASD_POCS: algs.os_asd_pocs,
        ReconMethodConstants.B_ASD_POCS_BETA: algs.asd_pocs,
        ReconMethodConstants.AW_ASD_POCS: algs.awasd_pocs,
        ReconMethodConstants.AWASD_POCS: algs.awasd_pocs,
        ReconMethodConstants.OS_AW_ASD_POCS: algs.os_awasd_pocs,
        ReconMethodConstants.OS_AWASD_POCS: algs.os_awasd_pocs,
        ReconMethodConstants.PCSD: algs.pcsd,
        ReconMethodConstants.OS_PCSD: algs.os_pcsd,
        ReconMethodConstants.AW_PCSD: algs.aw_pcsd,
        ReconMethodConstants.AWPCSD: algs.aw_pcsd,
        ReconMethodConstants.OS_AW_PCSD: algs.os_aw_pcsd,
        ReconMethodConstants.OS_AWPCSD: algs.os_aw_pcsd,
        # Krylov subspace family
        ReconMethodConstants.CGLS: algs.cgls,
        ReconMethodConstants.LSQR: algs.lsqr,
        ReconMethodConstants.HYBRID_LSQR: algs.hybrid_lsqr,
        ReconMethodConstants.H_LSQR: algs.hybrid_lsqr,
        ReconMethodConstants.LSMR: algs.lsmr,
        ReconMethodConstants.IRN_TV_CGLS: algs.irn_tv_cgls,
        ReconMethodConstants.HYBRID_FLSQR_TV: algs.hybrid_flsqr_tv,
        ReconMethodConstants.H_FLSQR_TV: algs.hybrid_flsqr_tv,
        ReconMethodConstants.AB_GMRES: algs.ab_gmres,
        ReconMethodConstants.BA_GMRES: algs.ba_gmres,
        ReconMethodConstants.AB_BA_GMRES: algs.ab_gmres,
        # Statistical / variational
        ReconMethodConstants.MLEM: algs.mlem,
        ReconMethodConstants.FISTA: algs.fista,
        ReconMethodConstants.ISTA: algs.ista,
        ReconMethodConstants.SART_TV: algs.sart_tv,
        ReconMethodConstants.OSSART_TV: algs.ossart_tv,
        ReconMethodConstants.OS_SART_TV: algs.ossart_tv,
    }

    def __init__(self, param: VxParam, laminography_method: LaminographyMethodConstants = LaminographyMethodConstants.INCLINED, left_pad: int = 0, right_pad: int = 0):
        self._param = param
        self._geom  = VxGeom(param)
        self._laminography_method = laminography_method
        self._left_pad = int(left_pad)
        self._right_pad = int(right_pad)
        self.result = None

    @staticmethod
    def _normalise_algo_name(algo: str) -> str:
        name = algo.strip()
        name = name.replace("\u03b2", "BETA").replace("\u0392", "BETA")
        name = name.upper()
        name = name.replace("?", "BETA")
        name = name.replace("-", "_").replace(" ", "_").replace("/", "_")
        while "__" in name:
            name = name.replace("__", "_")
        return name

    def _extrapolate(self, projections: np.ndarray, geo: tigre.geometry) -> tuple[np.ndarray, tigre.geometry]:
        left_pad = max(0, self._left_pad)
        right_pad = max(0, self._right_pad)
        if left_pad == 0 and right_pad == 0:
            return projections, geo

        # TIGRE projection stack is (angles, detector_v, detector_u).
        projections = np.pad(
            projections,
            ((0, 0), (0, 0), (left_pad, right_pad)),
            mode='edge',
        )

        geo = copy.deepcopy(geo)
        geo.nDetector = geo.nDetector.copy()
        geo.sDetector = geo.sDetector.copy()
        geo.nDetector[1] = projections.shape[2]
        geo.sDetector[1] = geo.nDetector[1] * geo.dDetector[1]

        center_shift_mm = ((right_pad - left_pad) / 2.0) * geo.dDetector[1]
        if center_shift_mm != 0.0:
            geo.offDetector = np.array(geo.offDetector, copy=True)
            if geo.offDetector.ndim == 1:
                geo.offDetector[1] += center_shift_mm
            else:
                geo.offDetector[:, 1] += center_shift_mm

        return projections, geo

    def run(self, projections: np.ndarray, algo: str = ReconMethodConstants.FDK, **kwargs) -> np.ndarray:
        is_planar = self._laminography_method == LaminographyMethodConstants.COPLANAR
        geo, angles = self._geom.get_planar_geometry_and_angles(self._laminography_method)

        projections, geo = self._extrapolate(projections, geo)

        name = self._normalise_algo_name(algo)
        if name == ReconMethodConstants.FDK:
            if is_planar and "dowang" not in kwargs:
                kwargs["dowang"] = False
            self.result = algs.fdk(
                projections,
                geo,
                angles,
                filter=kwargs.pop("filter", "shepp_logan"),
                **kwargs,
            )
            if is_planar:
                self.result = np.transpose(self.result, (2, 0, 1))
            return self.result

        fn = self._ITERATIVE_ALGORITHMS.get(name)
        if fn is None:
            known = ", ".join([ReconMethodConstants.FDK, *sorted(self._ITERATIVE_ALGORITHMS)])
            raise ValueError(f"Unknown algorithm: '{algo}'. Known algorithms: {known}")

        niter = kwargs.pop("niter", kwargs.pop("iterations", self._param.iterations))
        self.result = fn(projections, geo, angles, int(niter), **kwargs)
        if is_planar:
            if isinstance(self.result, tuple):
                recon, *extra = self.result
                self.result = (np.transpose(recon, (2, 0, 1)), *extra)
            else:
                self.result = np.transpose(self.result, (2, 0, 1))
        return self.result
