import numpy as np

from SyntheticDataGenerator.Structures.materials import MU_QUARTZ, MU_RUBY


class MicroJigStructure:
    """
    MICRO_JIG: a 22-ball calibration/verification test piece sized for a
    micro-CT / small-field-of-view cone-beam setup, per VDI/VDE 2630-1.3.
    This is a different scale of artefact from JigStructure - that one is a
    full-size, ~76 mm industrial-CT jig; this one targets a ~5 mm field of
    view, matching a rig with a source-to-object distance of only a few mm
    (published spec: contained in a cylindrical volume ~4 mm diameter x
    ~1.8 mm height).

    Unlike the first cut of this phantom (a Fibonacci-sphere hedgehog, which
    was a plausible-but-invented 3D shape), this geometry is fit directly to
    published measurement data for the real artefact: a peer-reviewed paper
    on this exact check piece lists 35 calibrated sphere-centre-to-centre
    distances, grouped into 5 nominal lengths, together covering
    "at least five different distances ... in a total of seven different
    spatial directions" as VDI/VDE 2630-1.3 requires. The 22 balls (0.3 mm
    ruby, on individual conical quartz-glass pillars rising from a quartz
    base, per the source) split cleanly by id parity/grouping into 3 families
    of 7 plus 1 centre ball:

      OUTER  (odd ids 1,3,..,13)   - distance to centre id 22 = 2.343 mm
      INNER  (ids 15..21)          - distance to centre id 22 = 1.210 mm
      UPPER  (even ids 2,4,..,14)  - distance to same-index INNER id = 1.134 mm
      CENTRE (id 22)

    and two more measured lengths cross those families at a fixed index
    shift of 3 (out of 7): OUTER[k] to INNER[k+3] = 2.850 mm, and OUTER[k] to
    UPPER[k+3] = 3.600 mm. Assuming 7-fold rotational symmetry (matching the
    7 families of 7) and that OUTER sits exactly opposite INNER/UPPER's
    (k+3)th member (the shift that makes the 5 equations solvable with a
    clean 180 deg relative phase - the fit converges to within a fraction of
    a degree of exactly 180 regardless of starting point), these 5 equations
    plus the published ~4 mm / ~1.8 mm envelope pin down a 3D layout - one
    radius and one height per family - solved numerically (see below) and
    verified to reproduce all 5 published lengths to 4-5 significant figures
    while landing within ~3-5% of the stated envelope (4.11 mm diameter x
    1.89 mm height here). This is fitted geometry, not the vendor's CAD -
    the real part is individually calibrated per serial number - but it
    reproduces real, published metrology rather than an invented shape.

    Linear attenuation coefficients (mm^-1, 60 keV scheme - see
    materials.py): ruby 0.115, quartz glass 0.059.
    """

    DEFAULT_SIZE_MM = (5.0, 5.0, 2.4)   # (width, length, height) mm, with margin

    N_PER_FAMILY = 7
    BALL_DIA_MM  = 0.3

    # Solved from the published sphere-pair distances (see docstring). Each
    # family sits at a fixed (radius, height); id -> (family, k) below.
    # Heights are zero-centred (mean of the lowest feature - the base plate
    # underside - and the highest - the tallest ball's top - is at z=0):
    # every other Structure's Build() allocates a voxel grid centred on the
    # origin, so a part built off-centre runs off the top of its own box
    # and gets silently clipped in projection, however large DEFAULT_SIZE_MM
    # is set - centring is what actually keeps the part inside the box.
    OUTER_RADIUS_MM  = 1.905711
    OUTER_HEIGHT_MM  = -0.722042
    INNER_RADIUS_MM  = 0.891678
    INNER_HEIGHT_MM  = -0.176919
    UPPER_RADIUS_MM  = 1.322120
    UPPER_HEIGHT_MM  = 0.872042
    CENTER_HEIGHT_MM = 0.641011

    # OUTER[k] sits at 180 deg (opposite) relative to INNER[k+3]/UPPER[k+3]
    # in the shared 7-fold angular indexing - see docstring.
    OUTER_PHASE_DEG = 180.0
    OUTER_INDEX_SHIFT = 3

    BASE_PLATE_RADIUS_MM = OUTER_RADIUS_MM + 0.3
    BASE_PLATE_THICKNESS_MM = 0.15

    PILLAR_BASE_RADIUS_MM = 0.18
    PILLAR_TIP_RADIUS_MM  = 0.07

    MU_BASE = MU_QUARTZ
    MU_BALL = MU_RUBY

    @staticmethod
    def GetStructure(vol_shape: tuple[int, int, int], voxel_size_mm: float) -> np.ndarray:
        """Phantom only, in ASTRA (z, y, x) order - the Structures interface."""
        vol, _ = MicroJigStructure.Build(vol_shape, voxel_size_mm)
        return vol

    @staticmethod
    def GetJigBallPosition() -> np.ndarray:
        """
        (22, 3) array of nominal ball centres, mm, in the same world frame
        Build() paints into - i.e. with no position_noise_mm jitter and no
        dependence on a voxel grid at all. For a consumer that only needs
        the known geometry (e.g. a reprojection-based geometry-calibration
        optimizer matching detected balls against where they *should*
        project to), building the whole volume just to read back
        `nominal_center_mm` off the manifest is unnecessary - this is the
        same computation Build() does, factored out so both call it once
        rather than keeping two copies of the ball layout in sync by hand.
        """
        H = MicroJigStructure
        layout = H.ballLayout_Internal()
        r = np.array([radius for _, _, radius, _, _ in layout])
        theta = np.deg2rad([angle for _, _, _, angle, _ in layout])
        z = np.array([height for _, _, _, _, height in layout])
        return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

    @staticmethod
    def Build(
        vol_shape: tuple[int, int, int],
        voxel_size_mm: float,
        seed: int = 0,
        position_noise_mm: float = 0.0,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Build the phantom and the ground-truth manifest.

        Args:
            vol_shape:          (vol_z, vol_y, vol_x) in voxels.
            voxel_size_mm:      isotropic voxel edge, mm.
            seed:                RNG seed for position_noise_mm.
            position_noise_mm:  std, mm, of isotropic Gaussian jitter applied
                                 to each ball centre - a stand-in for the
                                 sphere-centre-distance error SD this test
                                 piece exists to monitor. 0 (default) places
                                 every ball at its nominal (fitted) centre.

        Returns:
            (volume float32 (z, y, x), manifest: one dict per ball with id,
            family, diameter_mm, nominal_center_mm, actual_center_mm).
        """
        H = MicroJigStructure
        vol_z, vol_y, vol_x = vol_shape
        vs = float(voxel_size_mm)
        vol = np.zeros((vol_z, vol_y, vol_x), dtype=np.float32)
        rng = np.random.default_rng(seed)

        x1d = (np.arange(vol_x, dtype=np.float64) - vol_x / 2.0) * vs
        y1d = (np.arange(vol_y, dtype=np.float64) - vol_y / 2.0) * vs
        z1d = (np.arange(vol_z, dtype=np.float64) - vol_z / 2.0) * vs

        # base plate sits just under the shortest (OUTER) pillar stub
        plate_top = H.OUTER_HEIGHT_MM - 0.15
        H.paintDisc_Internal(
            vol, x1d, y1d, z1d,
            plate_top - H.BASE_PLATE_THICKNESS_MM, plate_top,
            H.BASE_PLATE_RADIUS_MM, H.MU_BASE)

        balls = H.ballLayout_Internal()

        manifest = []
        for ball_id, family, radius, angle_deg, height in balls:
            theta = np.deg2rad(angle_deg)
            nominal = (float(radius * np.cos(theta)), float(radius * np.sin(theta)), float(height))
            cx, cy, cz = nominal
            if position_noise_mm > 0:
                jx, jy, jz = rng.normal(scale=position_noise_mm, size=3)
                cx, cy, cz = cx + jx, cy + jy, cz + jz
            actual = (float(cx), float(cy), float(cz))

            H.paintCone_Internal(
                vol, x1d, y1d, z1d, cx, cy, plate_top, cz - H.BALL_DIA_MM * 0.15,
                H.PILLAR_BASE_RADIUS_MM, H.PILLAR_TIP_RADIUS_MM, H.MU_BASE)

            manifest.append(dict(
                id=ball_id, family=family, diameter_mm=H.BALL_DIA_MM,
                nominal_center_mm=nominal, actual_center_mm=actual,
            ))

        # balls painted after every pillar so they cleanly cap their tips
        for m in manifest:
            H.paintSphere_Internal(
                vol, x1d, y1d, z1d, m["actual_center_mm"],
                H.BALL_DIA_MM / 2.0, H.MU_BALL)

        return vol, manifest

    # private / internal
    @staticmethod
    def ballLayout_Internal() -> list[tuple[int, str, float, float, float]]:
        """(ball_id, family, radius_mm, angle_deg, height_mm) for all 22 balls."""
        H = MicroJigStructure
        step = 360.0 / H.N_PER_FAMILY
        balls = []
        for k in range(H.N_PER_FAMILY):
            balls.append((2 * k + 1, "OUTER", H.OUTER_RADIUS_MM,
                          (k + H.OUTER_INDEX_SHIFT) * step + H.OUTER_PHASE_DEG,
                          H.OUTER_HEIGHT_MM))
            balls.append((15 + k, "INNER", H.INNER_RADIUS_MM, k * step, H.INNER_HEIGHT_MM))
            balls.append((2 * k + 2, "UPPER", H.UPPER_RADIUS_MM, k * step, H.UPPER_HEIGHT_MM))
        balls.append((22, "CENTER", 0.0, 0.0, H.CENTER_HEIGHT_MM))
        return balls

    @staticmethod
    def paintDisc_Internal(vol, x1d, y1d, z1d, z0, z1, radius, mu) -> None:
        """Paint a flat disc (the base plate) centred on the z-axis."""
        vol_z, vol_y, vol_x = vol.shape
        k0 = max(0, int(np.searchsorted(z1d, z0)))
        k1 = min(vol_z, int(np.searchsorted(z1d, z1)) + 1)
        if k1 <= k0:
            return
        r2 = x1d[None, :] ** 2 + y1d[:, None] ** 2
        vol[k0:k1, r2 < radius ** 2] = mu

    @staticmethod
    def paintSphere_Internal(vol, x1d, y1d, z1d, centre, radius, mu) -> None:
        """Paint a solid ball into `vol` in place."""
        vol_z, vol_y, vol_x = vol.shape
        cx, cy, cz = centre

        i0 = max(0, int(np.searchsorted(x1d, cx - radius)))
        i1 = min(vol_x, int(np.searchsorted(x1d, cx + radius)) + 1)
        j0 = max(0, int(np.searchsorted(y1d, cy - radius)))
        j1 = min(vol_y, int(np.searchsorted(y1d, cy + radius)) + 1)
        k0 = max(0, int(np.searchsorted(z1d, cz - radius)))
        k1 = min(vol_z, int(np.searchsorted(z1d, cz + radius)) + 1)
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            return

        xl = (x1d[i0:i1] - cx)[None, None, :]
        yl = (y1d[j0:j1] - cy)[None, :, None]
        zl = (z1d[k0:k1] - cz)[:, None, None]
        sub = vol[k0:k1, j0:j1, i0:i1]
        sub[(xl ** 2 + yl ** 2 + zl ** 2) < radius ** 2] = mu

    @staticmethod
    def paintCone_Internal(vol, x1d, y1d, z1d, cx, cy, z0, z1, r0, r1, mu) -> None:
        """
        Paint a vertical conical pillar (frustum) at fixed (cx, cy), tapering
        from radius r0 at z0 to radius r1 at z1. Every ball sits directly
        above its own pillar (same x, y; only height and radius differ by
        family), so unlike JigStructure's pins or the old hedgehog spokes,
        this stays axis-aligned - just with a per-slice radius instead of a
        constant one.
        """
        vol_z, vol_y, vol_x = vol.shape
        if z1 <= z0:
            return
        reach = max(r0, r1)

        i0 = max(0, int(np.searchsorted(x1d, cx - reach)))
        i1 = min(vol_x, int(np.searchsorted(x1d, cx + reach)) + 1)
        j0 = max(0, int(np.searchsorted(y1d, cy - reach)))
        j1 = min(vol_y, int(np.searchsorted(y1d, cy + reach)) + 1)
        k0 = max(0, int(np.searchsorted(z1d, z0)))
        k1 = min(vol_z, int(np.searchsorted(z1d, z1)) + 1)
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            return

        xl = (x1d[i0:i1] - cx)[None, None, :]
        yl = (y1d[j0:j1] - cy)[None, :, None]
        t = np.clip((z1d[k0:k1] - z0) / (z1 - z0), 0.0, 1.0)
        r_at_z = (r0 + (r1 - r0) * t)[:, None, None]

        sub = vol[k0:k1, j0:j1, i0:i1]
        sub[(xl ** 2 + yl ** 2) < r_at_z ** 2] = mu
