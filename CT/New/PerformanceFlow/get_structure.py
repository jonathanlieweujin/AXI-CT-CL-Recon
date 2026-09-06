import numpy as np
from Util.param import VxParam
from Util.recon_method import ReconMethodConstants
from Util.laminography_method import LaminographyMethodConstants
from Util.compute_geometry import VxComputeGeometry

class VxTool:
    """Runs an ASTRA reconstruction algorithm given projections and a VxParam."""

    def __init__(self, param: VxParam, laminography_method: LaminographyMethodConstants = LaminographyMethodConstants.INCLINED, left_pad: int = 0, right_pad: int = 0):
        self._param = param
        self._laminography_method = laminography_method
        self._left_pad = int(left_pad)
        self._right_pad = int(right_pad)
        self.result = None

    def run(self, projections: np.ndarray, algo: str = ReconMethodConstants.FDK,
            min_constraint: float | None = None, max_constraint: float | None = None) -> np.ndarray:
        match algo.upper():
            case ReconMethodConstants.FDK:
                self.result = self._run_fdk(projections)
            case ReconMethodConstants.SIRT | "SIRT3D_CUDA":
                self.result = self._run_iterative(
                    projections,
                    astra_algo="SIRT3D_CUDA",
                    iterations=self._param.iterations,
                    min_constraint=min_constraint,
                    max_constraint=max_constraint,
                )
            case ReconMethodConstants.CGLS | "CGLS3D_CUDA":
                self.result = self._run_iterative(
                    projections,
                    astra_algo="CGLS3D_CUDA",
                    iterations=self._param.iterations,
                    min_constraint=min_constraint,
                    max_constraint=max_constraint,
                )
            case _:
                raise NotImplementedError(f"Algorithm '{algo}' is not implemented in PerformanceFlow.")

        return self.result

    def _get_vectors(self) -> np.ndarray:
        if self._laminography_method == LaminographyMethodConstants.COPLANAR:
            return VxComputeGeometry.computeCoplanarTranslationalLaminographyGeometry(self._param).reshape(-1, 12)
        return VxComputeGeometry.computeDefaultInclinedLaminographyGeometry(self._param).reshape(-1, 12)

    def _get_padded_vectors(self, vecs: np.ndarray) -> np.ndarray:
        left_pad = max(0, self._left_pad)
        right_pad = max(0, self._right_pad)
        if left_pad == 0 and right_pad == 0:
            return vecs

        padded = vecs.copy()
        center_shift_pixels = (right_pad - left_pad) / 2.0
        padded[:, 3:6] += center_shift_pixels * padded[:, 6:9]
        return padded

    def _extrapolate(self, projections: np.ndarray) -> np.ndarray:
        left_pad = max(0, self._left_pad)
        right_pad = max(0, self._right_pad)
        if left_pad == 0 and right_pad == 0:
            return projections

        return np.pad(
            projections,
            ((0, 0), (0, 0), (left_pad, right_pad)),
            mode='edge',
        )

    def _apply_redundancy_weight(
        self,
        sino,
        offsetX,
        srcPitch,
        sod,
        odd):
        """
        A weighting function using offset centre-of-rotation weights, for which the weighting function is angularly symmetric 
        about the centre of rotation. This function is used in this script for the offset cone-beam geometry.

        Weighting functions based on the work of G. Belotti, G. Fattori, G. Baroni, and S. Rit, “Extension of the cone-beam ct field-of-view
        using two complementary short scans,” Medical Physics, 2023.

        Input:
        sino: sinogram, assumed to be 2D array of shape (num_projections, num_columns)
        offset: offset of the centre of rotation from the centre of the detector
        detector_pixel_size: size of detector pixel
        distance_source_origin: distance from source to centre of rotation
        distance_origin_detector: distance from centre of rotation to detector

        Output:
        sino: weighted sinogram of same shape as input
        y: weighting function applied to sinogram

        """

        num_u = sino.shape[-1]

        # -------------------------------------------------
        # Step 1: Normalization 
        # -------------------------------------------------
        magnification = (sod + odd) / sod
        effective_pixel_size = srcPitch / magnification

        normOdd = odd / effective_pixel_size
        normSod = sod / effective_pixel_size

        # -------------------------------------------------
        # Step 2: Geometry angles
        # -------------------------------------------------
        tau = np.arctan(offsetX / normSod)

        alpha = np.arctan((num_u / 2.0) / normOdd)
        denom = alpha - np.abs(tau)

        # -------------------------------------------------
        # Step 3: Flat-region cutoff (matches C++)
        # -------------------------------------------------
        weight_end = int(np.round(
            abs(num_u / 2.0 + normOdd * np.tan(alpha - 2.0 * abs(tau)))
        ))

        # -------------------------------------------------
        # Step 4: Detector u coordinates (pixel centers!)
        # -------------------------------------------------
        u = np.arange(num_u, dtype=np.float64)
        t = u - num_u / 2.0 + 0.5

        # -------------------------------------------------
        # Step 5: Wang weights
        # -------------------------------------------------
        num = np.arctan(t / normOdd) + tau

        y = 0.5 * (
            np.sign(tau) * np.sin((np.pi / 2.0) * num / denom) + 1.0
        )
        y = 2.0 * y

        # -------------------------------------------------
        # Step 6: Flat region (constant W = 2)
        # -------------------------------------------------
        if tau > 0.0:
            y[weight_end:] = 2.0
        elif tau < 0.0:
            y[:num_u - weight_end] = 2.0

        # -------------------------------------------------
        # Step 7: Apply weights (broadcast over v, z, angles)
        # -------------------------------------------------
        sino *= y[np.newaxis, np.newaxis, :]

        return sino

    def plot_geometry(self, angle_index: int | None = None, show: bool = True, save_path: str | None = None):
        """Plot the exact ASTRA cone_vec geometry from S, D, U, V vectors."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        p = self._param
        vecs = self._get_padded_vectors(self._get_vectors())
        n_views = vecs.shape[0]
        max_idx = n_views - 1
        initial_idx = 0 if angle_index is None else int(np.clip(angle_index, 0, max_idx))
        interactive = angle_index is None and n_views > 1

        det_width = p.det_width + max(0, self._left_pad) + max(0, self._right_pad)
        half_u_px = det_width / 2.0
        half_v_px = p.det_height / 2.0
        voxel_size = p.det_pitch * p.sod / p.sdd
        half_x = p.dst_x * voxel_size / 2.0
        half_y = p.dst_y * voxel_size / 2.0
        half_z = p.dst_z * voxel_size / 2.0
        volume_faces = [
            np.array([[-half_x, -half_y, -half_z], [ half_x, -half_y, -half_z], [ half_x,  half_y, -half_z], [-half_x,  half_y, -half_z]]),
            np.array([[-half_x, -half_y,  half_z], [ half_x, -half_y,  half_z], [ half_x,  half_y,  half_z], [-half_x,  half_y,  half_z]]),
        ]

        def view_geometry(idx: int):
            row = vecs[idx]
            source = row[0:3]
            detector = row[3:6]
            u = row[6:9]
            v = row[9:12]
            corners = np.array([
                detector - half_u_px * u - half_v_px * v,
                detector + half_u_px * u - half_v_px * v,
                detector + half_u_px * u + half_v_px * v,
                detector - half_u_px * u + half_v_px * v,
            ])
            return source, detector, u, v, corners

        all_points = volume_faces[:]
        for idx in range(n_views):
            source, detector, _, _, corners = view_geometry(idx)
            all_points.extend([source[None, :], detector[None, :], corners])
        pts_all = np.vstack(all_points)
        center = pts_all.mean(axis=0)
        radius = np.max(np.linalg.norm(pts_all - center, axis=1))

        fig = plt.figure(figsize=(7, 7))
        if interactive:
            fig.subplots_adjust(bottom=0.16)
        ax = fig.add_subplot(111, projection="3d")

        def draw(idx: int):
            source, detector, u, v, corners = view_geometry(idx)
            unit_u = u / np.linalg.norm(u)
            unit_v = v / np.linalg.norm(v)
            ax.clear()
            ax.set_title(f"ASTRA cone_vec geometry, view {idx}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.scatter(*source, color="steelblue", s=80, label="Source")
            ax.scatter(*detector, color="maroon", s=35, label="Detector center")
            ax.add_collection3d(Poly3DCollection([corners], facecolor="maroon", alpha=0.35, edgecolor="black"))
            for face in volume_faces:
                ax.add_collection3d(Poly3DCollection([face], facecolor="black", alpha=0.08, edgecolor="black"))
            for corner in corners:
                ax.plot([source[0], corner[0]], [source[1], corner[1]], [source[2], corner[2]], color="black", lw=0.8)
            ax.plot([source[0], detector[0]], [source[1], detector[1]], [source[2], detector[2]], color="orange", lw=1.5)
            ax.quiver(detector[0], detector[1], detector[2], unit_u[0], unit_u[1], unit_u[2], length=np.linalg.norm(u) * half_u_px * 0.35, color="red")
            ax.quiver(detector[0], detector[1], detector[2], unit_v[0], unit_v[1], unit_v[2], length=np.linalg.norm(v) * half_v_px * 0.35, color="green")
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.legend(loc="upper right")

        draw(initial_idx)
        if interactive:
            slider_ax = fig.add_axes([0.20, 0.05, 0.60, 0.03])
            slider = Slider(slider_ax, "View", 0, max_idx, valinit=initial_idx, valstep=1)

            def update(value):
                draw(int(value))
                fig.canvas.draw_idle()

            slider.on_changed(update)
            fig._vx_angle_slider = slider

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def _prepare_astra_data(self, astra, projections: np.ndarray):
        p = self._param
        vecs = self._get_padded_vectors(self._get_vectors())

        # Geometry vectors are in mm. ASTRA vol_geom must use the same unit.
        sdd = p.sdd
        voxel_size = p.det_pitch * p.sod / sdd  # divide by Scale too if Scale != 1

        half_x = p.dst_x * voxel_size / 2.0
        half_y = p.dst_y * voxel_size / 2.0
        half_z = p.dst_z * voxel_size / 2.0
        vol_geom = astra.create_vol_geom(
            p.dst_y, p.dst_x, p.dst_z,
            -half_x + p.volume_mid_x,  half_x + p.volume_mid_x,
            -half_y + p.volume_mid_y,  half_y + p.volume_mid_y,
            -half_z + p.volume_mid_z,  half_z + p.volume_mid_z,
        )

        projections = np.transpose(projections, (1, 0, 2))
        projections = self._extrapolate(projections)
        projections = self._apply_redundancy_weight(projections,
                                   p.offset_u,
                                   p.det_pitch,
                                   p.sod,
                                   p.sdd - p.sod)

        proj_geom = astra.create_proj_geom(
            'cone_vec',
            p.det_height,
            projections.shape[2],
            vecs,
        )

        proj_id = astra.data3d.create('-proj3d', proj_geom, projections)
        vol_id = astra.data3d.create('-vol', vol_geom)
        return proj_id, vol_id

    def _run_fdk(self, projections: np.ndarray) -> np.ndarray:
        import astra

        proj_id, vol_id = self._prepare_astra_data(astra, projections)

        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ProjectionDataId'] = proj_id
        cfg['ReconstructionDataId'] = vol_id
        cfg['FilterType'] = 'shepp-logan'

        alg_id = None
        try:
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            return astra.data3d.get(vol_id)
        finally:
            if alg_id is not None:
                astra.algorithm.delete(alg_id)
            astra.data3d.delete(proj_id)
            astra.data3d.delete(vol_id)

    def _run_iterative(self, 
                       projections: np.ndarray, 
                       astra_algo: str, 
                       iterations: int,
                       min_constraint: float | None, max_constraint: float | None) -> np.ndarray:
        import astra

        proj_id, vol_id = self._prepare_astra_data(astra, projections)

        cfg = astra.astra_dict(astra_algo)
        cfg['ProjectionDataId'] = proj_id
        cfg['ReconstructionDataId'] = vol_id
        options = {}
        if astra_algo == "SIRT3D_CUDA":
            if min_constraint is not None:
                options['MinConstraint'] = min_constraint
            if max_constraint is not None:
                options['MaxConstraint'] = max_constraint
        elif min_constraint is not None or max_constraint is not None:
            raise ValueError(f"{astra_algo} does not support MinConstraint/MaxConstraint")
        if options:
            cfg['option'] = options

        alg_id = None
        try:
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, iterations=int(iterations))
            return astra.data3d.get(vol_id)
        finally:
            if alg_id is not None:
                astra.algorithm.delete(alg_id)
            astra.data3d.delete(proj_id)
            astra.data3d.delete(vol_id)
