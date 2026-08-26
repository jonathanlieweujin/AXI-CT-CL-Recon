import numpy as np
from Util.param import VxParam


class VxGeom:
    """
    Builds one 3x4 vector geometry matrix per projection angle,
    matching the ComputeSliceContributionToVectorStack convention:

        R = Rx(-tiltX) Â· Ry(tiltY) Â· Rz(-phi)

    Each row of the returned array is the 12-element flat vector:
        [Sx, Sy, Sz,  Dx, Dy, Dz,  Ux, Uy, Uz,  Vx, Vy, Vz]

    reshaped to (num_angles, 3, 4):
        col 0 â†’ D (detector origin)
        col 1 â†’ U (detector u-axis, row direction)
        col 2 â†’ V (detector v-axis, col direction)
        col 3 â†’ S (source position)

    which is the ASTRA cone_vec / parallel_vec layout.
    """

    def __init__(self, param: VxParam):
        self.param = param

    def computeGimballLockGeom(self) -> np.ndarray:
        """Build a Z-orbit geometry with a world-X-stabilized detector.

        Source and detector centres follow the inclined orbit used by the
        default geometry. Unlike the default rigid rotation, detector U is
        obtained by projecting world +X onto the detector plane. This removes
        detector roll: the edge that starts on the right remains the right
        edge throughout the orbit. V completes the orthonormal detector basis.
        """
        p = self.param
        tx = np.deg2rad(p.tilt_x)
        ty = np.deg2rad(p.tilt_y)
        phi = -np.deg2rad(p.angles)
        sx, cx = np.sin(tx), np.cos(tx)
        sy, cy = np.sin(ty), np.cos(ty)
        sp, cp = np.sin(phi), np.cos(phi)

        # Inclined detector normal at angle zero; its orbit is around world Z.
        n0 = np.array([cx * sy, -sx, cx * cy])

        def rotate_about_z(vector):
            return np.column_stack((
                cp * vector[0] - sp * vector[1],
                sp * vector[0] + cp * vector[1],
                np.full_like(phi, vector[2]),
            ))

        normal = rotate_about_z(n0)

        # Project fixed world-right into each detector plane. This no-roll
        # constraint keeps the same physical detector edge on the right.
        world_right = np.array([1.0, 0.0, 0.0])
        u = world_right - normal * (normal @ world_right)[:, None]
        u_norm = np.linalg.norm(u, axis=1, keepdims=True)
        if np.any(u_norm < 1e-12):
            raise ValueError("Gimbal U axis is singular when detector normal aligns with world X")
        u /= u_norm
        v = np.cross(u, normal)
        vecs = np.zeros((p.num_of_imgs, 12), dtype=np.float64)
        vecs[:, 0:3] = -p.sod * normal
        vecs[:, 3:6] = (p.sdd - p.sod) * normal
        vecs[:, 6:9] = p.det_pitch * u
        vecs[:, 9:12] = p.det_pitch * v
        vecs[:, 3:6] += p.offset_u * vecs[:, 6:9] + p.offset_v * vecs[:, 9:12]
        return vecs.reshape(p.num_of_imgs, 3, 4)

    def computePlanarGeom(self) -> np.ndarray:
        """
        Build ASTRA cone_vec geometry for the planar laminography flow.

        This planar flow does not use the old rotation-matrix extraction:
            U = row0 * pixel, V = -row1 * pixel,
            S = -row2 * SOD, D = row2 * ODD.

        Instead, SOD and ODD are interpreted as axial distances along Z.
        The laminography tilt creates only an XY orbit shift. For a point at
        axial distance z from the object plane, the in-plane shift is:
            r = z * tan(tilt_x)

        With phi = -deg2rad(angle), the source and detector centers are:
            Sx = -SOD * tan(tilt_x) * cos(phi)
            Sy = -SOD * tan(tilt_x) * sin(phi)
            Sz = -SOD

            Dx =  ODD * tan(tilt_x) * cos(phi)
            Dy =  ODD * tan(tilt_x) * sin(phi)
            Dz =  ODD

        The detector plane is kept flat in the world frame, so the pixel axes
        are constant for all projections:
            U = (det_pitch, 0, 0)
            V = (0, +det_pitch, 0)

        Detector offsets move D along those U/V axes:
            D = D + offset_u * U + offset_v * V

        ASTRA cone_vec row layout is:
            [Sx Sy Sz  Dx Dy Dz  Ux Uy Uz  Vx Vy Vz]
        """
        p = self.param

        tilt_x_rad = -np.deg2rad(p.tilt_x) # tilt x negated
        angles_rad = -np.deg2rad(p.angles)

        eff_sod = p.sod
        eff_odd = p.sdd - p.sod
        eff_pixel = p.det_pitch

        tan_tx = np.tan(tilt_x_rad)
        cos_phi = np.cos(angles_rad)
        sin_phi = np.sin(angles_rad)

        N = p.num_of_imgs
        vecs = np.zeros((N, 12), dtype=np.float64)

        # Source center: axial SOD plus tilt-induced XY shift.
        vecs[:, 0] = eff_sod * tan_tx * cos_phi
        vecs[:, 1] = eff_sod * tan_tx * sin_phi
        vecs[:, 2] = eff_sod

        # Detector center: axial ODD plus matching tilt-induced XY shift.
        vecs[:, 3] = -eff_odd * tan_tx * cos_phi
        vecs[:, 4] = -eff_odd * tan_tx * sin_phi
        vecs[:, 5] = -eff_odd

        # Flat detector basis in world coordinates.
        vecs[:, 6] = eff_pixel
        vecs[:, 7] = 0.0
        vecs[:, 8] = 0.0

        vecs[:,  9] = 0.0
        vecs[:, 10] = +eff_pixel
        vecs[:, 11] = 0.0

        # Detector offsets
        vecs[:, 3] += p.offset_u * vecs[:, 6] + p.offset_v * vecs[:, 9]
        vecs[:, 4] += p.offset_u * vecs[:, 7] + p.offset_v * vecs[:, 10]
        vecs[:, 5] += p.offset_u * vecs[:, 8] + p.offset_v * vecs[:, 11]

        return vecs.reshape(N, 3, 4)

    # ref: https:# doi.org/10.1088/1361-6501/aafcae
    # Laminography in the lab: imaging planar objects using a conventional x-ray CT scanner
    # Î¸ = detTiltXRad,  Î² = detTiltYRad,  Ï† = angleRadContribution
    # 
    # M1 = Rx(-Î¸) = [ 1      0       0   ]
    #               [ 0    cosÎ¸    sinÎ¸   ]
    #               [ 0   -sinÎ¸    cosÎ¸   ]
    # 
    # M2 = Ry(Î²)  = [ cosÎ²    0   -sinÎ²  ]
    #               [  0      1     0    ]
    #               [ sinÎ²    0    cosÎ²  ]
    # 
    # M3 = Rz(-Ï†) = [ cosÏ†   sinÏ†    0   ]
    #               [-sinÏ†   cosÏ†    0   ]
    #               [  0      0      1   ]
    # 
    # Step 1 â€” A = M1Â·M2:
    # 
    #   A[0][0] = 1Â·cosÎ²  + 0Â·0 + 0Â·sinÎ²   =  cosÎ²
    #   A[0][1] = 1Â·0     + 0Â·1 + 0Â·0      =  0
    #   A[0][2] = 1Â·(-sinÎ²) + 0Â·0 + 0Â·cosÎ² =  -sinÎ²
    # 
    #   A[1][0] = 0Â·cosÎ²  + cosÎ¸Â·0 + sinÎ¸Â·sinÎ²  =  sinÎ¸ sinÎ²
    #   A[1][1] = 0Â·0     + cosÎ¸Â·1 + sinÎ¸Â·0     =  cosÎ¸
    #   A[1][2] = 0Â·(-sinÎ²) + cosÎ¸Â·0 + sinÎ¸Â·cosÎ² =  sinÎ¸ cosÎ²
    # 
    #   A[2][0] = 0Â·cosÎ²  + (-sinÎ¸)Â·0 + cosÎ¸Â·sinÎ²  =  cosÎ¸ sinÎ²
    #   A[2][1] = 0Â·0     + (-sinÎ¸)Â·1 + cosÎ¸Â·0     = -sinÎ¸
    #   A[2][2] = 0Â·(-sinÎ²) + (-sinÎ¸)Â·0 + cosÎ¸Â·cosÎ² =  cosÎ¸ cosÎ²
    # 
    #        [  cosÎ²      0       -sinÎ²  ]
    # A  =   [  sinÎ¸ sinÎ²  cosÎ¸   sinÎ¸ cosÎ² ]
    #        [  cosÎ¸ sinÎ² -sinÎ¸   cosÎ¸ cosÎ² ]
    # 
    # Step 2 â€” R = AÂ·M3:
    # 
    #   R[0][0] = cosÎ²Â·cosÏ†   + 0Â·(-sinÏ†)      + (-sinÎ²)Â·0   =  cosÎ² cosÏ†
    #   R[0][1] = cosÎ²Â·sinÏ†   + 0Â·cosÏ†         + (-sinÎ²)Â·0   =  cosÎ² sinÏ†
    #   R[0][2] = cosÎ²Â·0      + 0Â·0            + (-sinÎ²)Â·1   = -sinÎ²
    # 
    #   R[1][0] = sinÎ¸ sinÎ²Â·cosÏ† + cosÎ¸Â·(-sinÏ†) + sinÎ¸ cosÎ²Â·0  =  sinÎ¸ sinÎ² cosÏ† - cosÎ¸ sinÏ†
    #   R[1][1] = sinÎ¸ sinÎ²Â·sinÏ† + cosÎ¸Â·cosÏ†   + sinÎ¸ cosÎ²Â·0  =  sinÎ¸ sinÎ² sinÏ† + cosÎ¸ cosÏ†
    #   R[1][2] = sinÎ¸ sinÎ²Â·0   + cosÎ¸Â·0       + sinÎ¸ cosÎ²Â·1  =  sinÎ¸ cosÎ²
    # 
    #   R[2][0] = cosÎ¸ sinÎ²Â·cosÏ† + (-sinÎ¸)Â·(-sinÏ†) + cosÎ¸ cosÎ²Â·0  =  cosÎ¸ sinÎ² cosÏ† + sinÎ¸ sinÏ†
    #   R[2][1] = cosÎ¸ sinÎ²Â·sinÏ† + (-sinÎ¸)Â·cosÏ†   + cosÎ¸ cosÎ²Â·0  =  cosÎ¸ sinÎ² sinÏ† - sinÎ¸ cosÏ†
    #   R[2][2] = cosÎ¸ sinÎ²Â·0   + (-sinÎ¸)Â·0       + cosÎ¸ cosÎ²Â·1  =  cosÎ¸ cosÎ²
    # 
    #        [  cosÎ² cosÏ†                  cosÎ² sinÏ†                 -sinÎ²      ]  <- row0
    # R  =   [  sinÎ¸ sinÎ² cosÏ† - cosÎ¸ sinÏ†   sinÎ¸ sinÎ² sinÏ† + cosÎ¸ cosÏ†   sinÎ¸ cosÎ²  ]  <- row1
    #        [  cosÎ¸ sinÎ² cosÏ† + sinÎ¸ sinÏ†   cosÎ¸ sinÎ² sinÏ† - sinÎ¸ cosÏ†   cosÎ¸ cosÎ²  ]  <- row2
    # 
    # Extraction:
    #   U  = +row0 Â· effPixelSize
    #   V  = -row1 Â· effPixelSize
    #   S  = -row2 Â· effSOD
    #   D  = +row2 Â· effODD
    def computeDefaultGeom(self) -> np.ndarray:
        p = self.param

        tilt_x_rad = np.deg2rad(p.tilt_x)   # Î¸
        tilt_y_rad = np.deg2rad(p.tilt_y)   # Î²

        angles_rad = -np.deg2rad(p.angles)  # negated: matches -angleDeg[i] * PI / 180

        # Effective geometry (voxel domain, voxel_size = 1)
        eff_sod = p.sod
        eff_odd = p.sdd - p.sod
        eff_pixel = p.det_pitch

        sin_tx = np.sin(tilt_x_rad)   # sinÎ¸
        cos_tx = np.cos(tilt_x_rad)   # cosÎ¸
        sin_ty = np.sin(tilt_y_rad)   # sinÎ²
        cos_ty = np.cos(tilt_y_rad)   # cosÎ²

        cos_phi = np.cos(angles_rad)   # (N,)
        sin_phi = np.sin(angles_rad)   # (N,)

        # R = Rx(-Î¸) Â· Ry(Î²) Â· Rz(-Ï†)  (broadcast over N angles)
        # row0
        r00 =  cos_ty * cos_phi
        r01 =  cos_ty * sin_phi
        r02 = -sin_ty * np.ones_like(cos_phi)

        # row1
        r10 =  sin_tx * sin_ty * cos_phi - cos_tx * sin_phi
        r11 =  sin_tx * sin_ty * sin_phi + cos_tx * cos_phi
        r12 =  sin_tx * cos_ty * np.ones_like(cos_phi)

        # row2
        r20 =  cos_tx * sin_ty * cos_phi + sin_tx * sin_phi
        r21 =  cos_tx * sin_ty * sin_phi - sin_tx * cos_phi
        r22 =  cos_tx * cos_ty * np.ones_like(cos_phi)

        N = p.num_of_imgs
        vecs = np.zeros((N, 12), dtype=np.float64)

        # S = -row2 Â· eff_sod
        vecs[:, 0] = -eff_sod * r20
        vecs[:, 1] = -eff_sod * r21
        vecs[:, 2] = -eff_sod * r22

        # D = +row2 Â· eff_odd
        vecs[:, 3] = eff_odd * r20
        vecs[:, 4] = eff_odd * r21
        vecs[:, 5] = eff_odd * r22

        # U = +row0 Â· eff_pixel
        vecs[:, 6] = eff_pixel * r00
        vecs[:, 7] = eff_pixel * r01
        vecs[:, 8] = eff_pixel * r02

        # V = -row1 Â· eff_pixel
        vecs[:, 9]  = -eff_pixel * r10
        vecs[:, 10] = -eff_pixel * r11
        vecs[:, 11] = -eff_pixel * r12

        # Apply detector offsets along U and V axes
        vecs[:, 3] += p.offset_u * vecs[:, 6] + p.offset_v * vecs[:, 9]
        vecs[:, 4] += p.offset_u * vecs[:, 7] + p.offset_v * vecs[:, 10]
        vecs[:, 5] += p.offset_u * vecs[:, 8] + p.offset_v * vecs[:, 11]

        # Return as (N, 3, 4): each matrix is [D | U | V | S]
        # ASTRA cone_vec row layout: [Sx Sy Sz  Dx Dy Dz  Ux Uy Uz  Vx Vy Vz]
        return vecs.reshape(N, 3, 4)

class VxTool:
    """Runs an ASTRA reconstruction algorithm given projections and a VxParam."""

    def __init__(self, param: VxParam, detector_parallel: bool = False, left_pad: int = 0, right_pad: int = 0):
        self._geom = VxGeom(param)
        self._param = param
        self._detector_parallel = detector_parallel
        self._left_pad = int(left_pad)
        self._right_pad = int(right_pad)
        self.result = None

    def run(self, projections: np.ndarray, algo: str = "FDK", iterations: int = 5,
            min_constraint: float | None = None, max_constraint: float | None = None) -> np.ndarray:
        match algo.upper():
            case "fdk" | "FDK":
                self.result = self._run_fdk(projections)
            case "sirt" | "SIRT" | "SIRT3D_CUDA":
                self.result = self._run_iterative(
                    projections,
                    astra_algo="SIRT3D_CUDA",
                    iterations=iterations,
                    min_constraint=min_constraint,
                    max_constraint=max_constraint,
                )
            case "cgls" | "CGLS" | "CGLS3D_CUDA":
                self.result = self._run_iterative(
                    projections,
                    astra_algo="CGLS3D_CUDA",
                    iterations=iterations,
                    min_constraint=min_constraint,
                    max_constraint=max_constraint,
                )
            case _:
                raise NotImplementedError(f"Algorithm '{algo}' is not implemented in AstraLib.")

        return self.result

    def _get_vectors(self) -> np.ndarray:
        if self._detector_parallel:
            return self._geom.computePlanarGeom().reshape(-1, 12)
        return self._geom.computeDefaultGeom().reshape(-1, 12)

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
