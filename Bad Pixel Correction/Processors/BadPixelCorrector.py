import numpy as np
from scipy.ndimage import binary_dilation

# ref : CIL
class BadPixelCorrector:
    r'''Corrects bad pixels in a stack of projections, by replacing each with the weighted mean value
    of unmasked nearest neighbours (including diagonals) in the projection.

    The mask is shared by every projection in the stack (e.g. a fixed sensor defect map). Bad-pixel
    clusters are corrected in "onion-peel" passes: pixels touching already-good data are corrected
    first, then the next ring inward reuses those newly-corrected values, and so on until every bad
    pixel has been resolved.

    Parameters
    ----------
    mask : numpy.ndarray
        A boolean 2D array with the same shape as one projection, where 'False' represents bad
        (masked) pixels. Can be generated using 'MaskGenerator'.
    '''

    def __init__(self, mask):
        if mask is None:
            raise ValueError('Please provide a mask.')
        self.mask = np.asarray(mask, dtype=bool)

    def process(self, projections):
        r'''Correct bad pixels in a stack of projections.

        :param projections: array of shape (num_proj, H, W) or a single projection (H, W)
        :type projections: numpy.ndarray
        :return: corrected array, same shape as input
        :rtype: numpy.ndarray
        '''

        projections = np.asarray(projections)
        single = projections.ndim == 2
        stack = projections[np.newaxis, ...] if single else projections.copy()

        if stack.shape[1:] != self.mask.shape:
            raise ValueError(f"Projection and mask shapes do not match: {stack.shape[1:]} != {self.mask.shape}")

        out = stack.astype(np.float64, copy=True)
        kernels, coords_x, coords_y = self._get_details_of_passes(self.mask)

        print("Number of passes: ", len(kernels))

        self._bad_pix_correct_pixelwise(out, coords_x, coords_y, kernels)

        out = out.astype(projections.dtype, copy=False)
        return out[0] if single else out

    def _get_details_of_passes(self, mask):
        r'''
        Build, for each onion-peel pass, the pixel coordinates to correct and a per-pixel bit-mask
        recording which of its 8 neighbours are usable (i.e. already-good in the previous pass).
        '''
        kernels = []
        coords_x = []
        coords_y = []
        pix_v, pix_h = mask.shape

        prev_erosion = mask
        eroded_mask = np.zeros_like(mask)

        while not np.all(eroded_mask):
            eroded_mask = binary_dilation(prev_erosion)
            current_pixel_map = ~prev_erosion & eroded_mask

            y_coords, x_coords = np.where(current_pixel_map)

            current_kernels = np.zeros(len(x_coords), dtype=np.uint8)

            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                bit_mask = 255

                if x == 0:
                    bit_mask = bit_mask & 107
                if x == pix_h - 1:
                    bit_mask = bit_mask & 214
                if y == 0:
                    bit_mask = bit_mask & 31
                if y == pix_v - 1:
                    bit_mask = bit_mask & 248

                # Drop neighbours that are still bad pixels not yet corrected in a previous pass.
                if (bit_mask & 1 << 7) and not prev_erosion[y - 1][x - 1]:
                    bit_mask = bit_mask & ~(1 << 7)
                if (bit_mask & 1 << 6) and not prev_erosion[y - 1][x]:
                    bit_mask = bit_mask & ~(1 << 6)
                if (bit_mask & 1 << 5) and not prev_erosion[y - 1][x + 1]:
                    bit_mask = bit_mask & ~(1 << 5)
                if (bit_mask & 1 << 4) and not prev_erosion[y][x - 1]:
                    bit_mask = bit_mask & ~(1 << 4)
                if (bit_mask & 1 << 3) and not prev_erosion[y][x + 1]:
                    bit_mask = bit_mask & ~(1 << 3)
                if (bit_mask & 1 << 2) and not prev_erosion[y + 1][x - 1]:
                    bit_mask = bit_mask & ~(1 << 2)
                if (bit_mask & 1 << 1) and not prev_erosion[y + 1][x]:
                    bit_mask = bit_mask & ~(1 << 1)
                if (bit_mask & 1) and not prev_erosion[y + 1][x + 1]:
                    bit_mask = bit_mask & ~1

                current_kernels[i] = bit_mask

            coords_x.append(x_coords)
            coords_y.append(y_coords)
            kernels.append(current_kernels)

            prev_erosion = eroded_mask

        return kernels, coords_x, coords_y

    def _bad_pix_correct_pixelwise(self, stack, coords_x, coords_y, kernels):
        diag_weight = 1 / np.sqrt(2)
        num_proj = stack.shape[0]

        for k in range(num_proj):
            current_proj = stack[k]
            for j in range(len(kernels)):
                for i in range(len(kernels[j])):
                    kernel = kernels[j][i]
                    x = coords_x[j][i]
                    y = coords_y[j][i]

                    accum_num = 0.0
                    accum_denom = 0.0

                    if kernel & 1 << 7:
                        accum_num += current_proj[y - 1][x - 1]
                        accum_denom += 1
                    if kernel & 1 << 5:
                        accum_num += current_proj[y - 1][x + 1]
                        accum_denom += 1
                    if kernel & 1 << 2:
                        accum_num += current_proj[y + 1][x - 1]
                        accum_denom += 1
                    if kernel & 1:
                        accum_num += current_proj[y + 1][x + 1]
                        accum_denom += 1

                    accum_denom *= diag_weight
                    accum_num *= diag_weight

                    if kernel & 1 << 6:
                        accum_num += current_proj[y - 1][x]
                        accum_denom += 1
                    if kernel & 1 << 4:
                        accum_num += current_proj[y][x - 1]
                        accum_denom += 1
                    if kernel & 1 << 3:
                        accum_num += current_proj[y][x + 1]
                        accum_denom += 1
                    if kernel & 1 << 1:
                        accum_num += current_proj[y + 1][x]
                        accum_denom += 1

                    current_proj[y][x] = accum_num / accum_denom
