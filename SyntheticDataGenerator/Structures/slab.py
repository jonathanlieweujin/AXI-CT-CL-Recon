import numpy as np


class SlabStructure:
    @staticmethod
    def GetStructure(vol_shape: tuple[int, int, int], voxel_size_mm: float) -> np.ndarray:
        vol_z, vol_y, vol_x = vol_shape
        vol = np.zeros((vol_z, vol_y, vol_x), dtype=np.float32)
        cz = vol_z / 2.0

        y1d = np.arange(vol_y, dtype=np.float32) - vol_y / 2.0
        x1d = np.arange(vol_x, dtype=np.float32) - vol_x / 2.0

        z0, z1 = int(cz - vol_z * 0.22), int(cz + vol_z * 0.22)
        vol[z0:z1, int(vol_y * 0.18):int(vol_y * 0.82),
            int(vol_x * 0.18):int(vol_x * 0.82)] = 0.6

        for k in range(4):
            yk = int(vol_y * (0.28 + 0.14 * k))
            vol[z0:z1, yk:yk + 4, int(vol_x * 0.24):int(vol_x * 0.76)] = 1.8

        r = min(vol_x, vol_y) * 0.07
        disc_2d = y1d[:, None] ** 2 + (x1d[None, :] + vol_x * 0.22) ** 2 < r ** 2
        vol[z0:z1, disc_2d] = 2.4

        return vol
