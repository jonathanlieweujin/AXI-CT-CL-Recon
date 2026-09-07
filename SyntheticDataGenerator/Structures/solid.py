import numpy as np


class SolidStructure:
    @staticmethod
    def GetStructure(vol_shape: tuple[int, int, int], voxel_size_mm: float) -> np.ndarray:
        vol_z, vol_y, vol_x = vol_shape
        vol = np.zeros((vol_z, vol_y, vol_x), dtype=np.float32)
        cz, cy, cx = vol_z / 2.0, vol_y / 2.0, vol_x / 2.0

        y1d = np.arange(vol_y, dtype=np.float32) - cy
        x1d = np.arange(vol_x, dtype=np.float32) - cx
        yx_sq = y1d[:, None] ** 2 + x1d[None, :] ** 2

        r = min(vol_x, vol_y, vol_z) * 0.34
        r_sq = float(r ** 2)
        for iz in range(max(0, int(cz - r)), min(vol_z, int(cz + r) + 1)):
            rem = r_sq - float((iz - cz) ** 2)
            if rem > 0:
                vol[iz, yx_sq < rem] = 1.0

        r2 = min(vol_x, vol_y) * 0.12
        rod_sq = y1d[:, None] ** 2 + (x1d[None, :] - vol_x * 0.16) ** 2
        vol[:, rod_sq < r2 ** 2] = 2.0

        vol[int(cz - vol_z * 0.30):int(cz - vol_z * 0.18),
            int(cy - vol_y * 0.22):int(cy + vol_y * 0.22),
            int(cx - vol_x * 0.30):int(cx - vol_x * 0.10)] = 1.6

        return vol
