import numpy as np

from SyntheticDataGenerator.Structures.materials import (
    MU_ACRYLIC, MU_CERAMIC, MU_INVAR, MU_RUBY,
)


class MicroJigStructure:
    """
    MICRO_JIG: a 22-sphere calibration/verification test piece for
    monitoring the sphere-centre-distance error SD (VDI/VDE 2617 p.13,
    VDI/VDE 2630 p.1.3) and, paired with a length-standard artefact, the
    length measuring error E of a CT scanner.

    The geometry here is a physically-motivated approximation, not a
    calibrated CAD: a real unit's sphere coordinates are individually
    calibrated and certified per serial number, so there is no single
    "correct" layout to copy. What follows is built from the properties
    that are intrinsic to this class of test piece:

      22 precision spheres   one larger datum sphere (4 mm dia) used for
                             CMM/CT base alignment, the other 21 at 3 mm
      base body              Invar, rho 8.1 g/cm^3, alpha 1.3e-6 /K
      ball shaft             ceramic
      protective cover       acrylic glass
      max measuring length   56.5 mm (sphere-centre to sphere-centre)
      assembly weight        1.55 kg incl. mounting pallet
      22 spheres positioned to evaluate ~25 measuring lengths for
      VDI/VDE 2630 compliance

    A length-measurement report on this class of artefact names its
    characteristics as sphere-id pairs (SD 1-15, SD 1-22, SD 3-16, SD 3-22,
    SD 5-17, ...) that always pair an odd sphere id with another odd id or
    with an even id, never in a way consistent with a single ring - so the
    22 spheres are modelled as two interleaved 11-point rings: odd ids
    (1, 3, .., 21) on the outer ring, even ids (2, 4, .., 22) on the inner
    ring, the inner ring rotated by half a pitch (16.36 deg) so no sphere
    sits radially behind another. The outer-ring radius is chosen so the
    longest chord on 11 points (5 steps apart, the closest an odd count
    gets to opposite) comes out at the 56.5 mm measuring length above.

    In practice a scan is framed so the reconstruction volume sees only the
    spheres and shafts, not the base - so the base sits low in the phantom
    box here, outside where a caller would normally put the reconstruction
    volume, but is still built because rays still cross it off-axis. That
    is exactly the distinction PhantomSizeMm (the full phantom box) draws
    against the recon `volume` elsewhere in this generator.

    Linear attenuation coefficients (mm^-1, 60 keV scheme - see
    materials.py): Invar 1.215, ruby 0.115, ceramic (Al2O3 shaft) 0.101,
    acrylic 0.035.
    """

    # Bounding box the phantom is designed for: 2x(base radius + cover gap +
    # cover wall) laterally, base height + tallest pin/sphere + cover
    # clearance + lid thickness in z, each padded ~4 mm so nothing touches
    # the box wall.
    DEFAULT_SIZE_MM = (76.0, 76.0, 55.0)

    # sphere layout: two interleaved 11-point rings (odd ids outer, even
    # ids inner), radius picked so the longest outer-ring chord is ~56.5 mm
    N_PER_RING          = 11
    OUTER_RADIUS_MM     = 28.5
    INNER_RADIUS_MM     = 18.0
    INNER_ROTATION_DEG  = 180.0 / N_PER_RING

    DATUM_SPHERE_DIA_MM = 4.0   # sphere 1 - the base-alignment datum
    SPHERE_DIA_MM       = 3.0   # spheres 2..22
    PIN_DIA_MM          = 1.2   # ceramic shaft diameter

    OUTER_PIN_LEN_MM = 26.0   # base top -> outer-ring sphere centre
    INNER_PIN_LEN_MM = 16.0   # base top -> inner-ring sphere centre

    BASE_RADIUS_MM = 31.0     # > OUTER_RADIUS_MM + PIN_DIA_MM/2, with margin
    BASE_HEIGHT_MM = 18.0
    BOSS_RADIUS_MM = 8.0      # central raised boss on the base, cosmetic
    BOSS_HEIGHT_MM = 6.0

    COVER_GAP_MM  = 3.0    # air gap: base rim -> acrylic cover inner wall
    COVER_WALL_MM = 2.0
    COVER_TOP_CLEARANCE_MM = 3.0   # tallest sphere top -> underside of the lid

    MU_BASE   = MU_INVAR
    MU_PIN    = MU_CERAMIC
    MU_SPHERE = MU_RUBY
    MU_COVER  = MU_ACRYLIC

    @staticmethod
    def GetStructure(vol_shape: tuple[int, int, int], voxel_size_mm: float) -> np.ndarray:
        """Phantom only, in ASTRA (z, y, x) order - the Structures interface."""
        vol, _ = MicroJigStructure.Build(vol_shape, voxel_size_mm)
        return vol

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
                                 to each sphere centre - a stand-in for the
                                 sphere-centre-distance error SD this test
                                 piece exists to monitor. 0 (default) places
                                 every sphere at its nominal centre.

        Returns:
            (volume float32 (z, y, x), manifest: one dict per sphere with
            id, ring, diameter_mm, nominal_center_mm, actual_center_mm).
        """
        H = MicroJigStructure
        vol_z, vol_y, vol_x = vol_shape
        vs = float(voxel_size_mm)
        vol = np.zeros((vol_z, vol_y, vol_x), dtype=np.float32)
        rng = np.random.default_rng(seed)

        x1d = (np.arange(vol_x, dtype=np.float64) - vol_x / 2.0) * vs
        y1d = (np.arange(vol_y, dtype=np.float64) - vol_y / 2.0) * vs
        z1d = (np.arange(vol_z, dtype=np.float64) - vol_z / 2.0) * vs

        tallest_top = (H.BASE_HEIGHT_MM + H.OUTER_PIN_LEN_MM
                       + H.DATUM_SPHERE_DIA_MM / 2.0)
        cover_top = tallest_top + H.COVER_TOP_CLEARANCE_MM
        total_h = cover_top + H.COVER_WALL_MM
        z_bottom = -total_h / 2.0
        z_base_top = z_bottom + H.BASE_HEIGHT_MM

        def zrange(lo, hi):
            k0 = max(0, min(int(np.searchsorted(z1d, lo)), vol_z))
            k1 = max(0, min(int(np.searchsorted(z1d, hi)), vol_z))
            return k0, k1

        r2 = x1d[None, :] ** 2 + y1d[:, None] ** 2   # (vol_y, vol_x)

        # Invar base cylinder + central boss
        k0, k1 = zrange(z_bottom, z_base_top)
        vol[k0:k1, r2 < H.BASE_RADIUS_MM ** 2] = H.MU_BASE

        k0, k1 = zrange(z_base_top, z_base_top + H.BOSS_HEIGHT_MM)
        vol[k0:k1, r2 < H.BOSS_RADIUS_MM ** 2] = H.MU_BASE

        # acrylic cover: cylindrical wall + top lid, kept clear of the base
        cover_inner = H.BASE_RADIUS_MM + H.COVER_GAP_MM
        cover_outer = cover_inner + H.COVER_WALL_MM
        wall_mask = (r2 >= cover_inner ** 2) & (r2 < cover_outer ** 2)
        k0, k1 = zrange(z_base_top, z_bottom + cover_top)
        vol[k0:k1, wall_mask] = H.MU_COVER

        lid_mask = r2 < cover_outer ** 2
        k0, k1 = zrange(z_bottom + cover_top, z_bottom + cover_top + H.COVER_WALL_MM)
        vol[k0:k1, lid_mask] = H.MU_COVER

        # 22 spheres on ceramic pins, two interleaved 11-point rings
        manifest = []
        for ring, radius, pin_len in (
            ("outer", H.OUTER_RADIUS_MM, H.OUTER_PIN_LEN_MM),
            ("inner", H.INNER_RADIUS_MM, H.INNER_PIN_LEN_MM),
        ):
            rot = 0.0 if ring == "outer" else H.INNER_ROTATION_DEG
            for k in range(H.N_PER_RING):
                sid = 2 * k + 1 if ring == "outer" else 2 * k + 2
                angle = np.deg2rad(k * 360.0 / H.N_PER_RING + rot)
                cx0, cy0 = radius * np.cos(angle), radius * np.sin(angle)
                cz0 = z_base_top + pin_len
                dia = H.DATUM_SPHERE_DIA_MM if sid == 1 else H.SPHERE_DIA_MM

                nominal = (float(cx0), float(cy0), float(cz0))
                cx, cy, cz = cx0, cy0, cz0
                if position_noise_mm > 0:
                    jx, jy, jz = rng.normal(scale=position_noise_mm, size=3)
                    cx, cy, cz = cx0 + jx, cy0 + jy, cz0 + jz
                actual = (float(cx), float(cy), float(cz))

                H.paintPinAndSphere_Internal(
                    vol, x1d, y1d, z1d, cx, cy, z_base_top, cz, dia)
                manifest.append(dict(
                    id=sid, ring=ring, diameter_mm=dia,
                    nominal_center_mm=nominal, actual_center_mm=actual,
                ))

        return vol, manifest

    # private / internal
    @staticmethod
    def paintPinAndSphere_Internal(vol, x1d, y1d, z1d, cx, cy, z_pin_bot,
                                    cz, dia) -> None:
        """Paint one ceramic shaft and its ruby ball into `vol` in place."""
        H = MicroJigStructure
        vol_z, vol_y, vol_x = vol.shape
        pin_r = H.PIN_DIA_MM / 2.0
        sph_r = dia / 2.0
        step = x1d[1] - x1d[0] if len(x1d) > 1 else 1.0
        reach = max(pin_r, sph_r) + 2.0 * step

        i0 = max(0, int(np.searchsorted(x1d, cx - reach)))
        i1 = min(vol_x, int(np.searchsorted(x1d, cx + reach)) + 1)
        j0 = max(0, int(np.searchsorted(y1d, cy - reach)))
        j1 = min(vol_y, int(np.searchsorted(y1d, cy + reach)) + 1)
        if i0 >= i1 or j0 >= j1:
            return

        xl = (x1d[i0:i1] - cx)[None, None, :]
        yl = (y1d[j0:j1] - cy)[None, :, None]

        # shaft: from the base top to just under the sphere centre. The
        # radial mask has no z dependence, so it is applied to every layer
        # of the slab explicitly rather than relying on boolean broadcasting
        # (which numpy does not do for fancy indexing).
        k0 = max(0, int(np.searchsorted(z1d, z_pin_bot)))
        k1 = min(vol_z, int(np.searchsorted(z1d, cz)) + 1)
        if k1 > k0:
            pin_mask = ((xl ** 2 + yl ** 2) < pin_r ** 2)[0]
            sub = vol[k0:k1, j0:j1, i0:i1]
            sub[:, pin_mask] = H.MU_PIN

        # ball: painted after the shaft so it cleanly overwrites the tip
        k0 = max(0, int(np.searchsorted(z1d, cz - sph_r)))
        k1 = min(vol_z, int(np.searchsorted(z1d, cz + sph_r)) + 1)
        if k1 > k0:
            zl = (z1d[k0:k1] - cz)[:, None, None]
            sub = vol[k0:k1, j0:j1, i0:i1]
            sub[(xl ** 2 + yl ** 2 + zl ** 2) < sph_r ** 2] = H.MU_SPHERE
