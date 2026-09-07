import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from Manager.Param import VxParam
from Manager.Constants.flow_method import VxFlowMethod
from Manager.Constants.laminography_method import LaminographyMethodConstants
from Manager.Util.compute_geometry import VxComputeGeometry
from Manager.Util.load_images_internal import FileLoader
from Manager.Tool.Performance import VxTool as VxPerformanceTool
from Manager.Tool.Quality import VxTool as VxQualityTool


class VxManager:
    """
    Single entry point for a reconstruction: holds the parameters, both backend
    tools, and the resulting volume.

    The flow method chooses the backend (Quality = TIGRE, Performance = ASTRA);
    the recon method on VxParam chooses the algorithm (FDK, SIRT, CGLS, ...).
    """

    def __init__(
        self,
        param: VxParam,
        flow_method: VxFlowMethod = VxFlowMethod.QUALITY,
        left_pad: int = 0,
        right_pad: int = 0,
    ):
        self.Param = param
        self.FlowMethod = flow_method
        self.ReconMethod = param.recon_method
        self.Images = None
        self.Result = np.ndarray(0, dtype=np.float32)

        self.QualityTool = VxQualityTool(
            param, laminography_method=param.laminography_method,
            left_pad=left_pad, right_pad=right_pad)
        self.PerformanceTool = VxPerformanceTool(
            param, laminography_method=param.laminography_method,
            left_pad=left_pad, right_pad=right_pad)

    # ------------------------------------------------------------------ public

    def LoadImages(self, input_folder: str, interval: int = 1) -> np.ndarray:
        """
        Load the projection stack described by Param from input_folder.

        Param already holds post-binning detector dimensions, so the source
        dimensions are restored here before handing them to the loader.
        """
        p = self.Param
        src_width = int(p.det_width) * p.binning
        src_height = int(p.det_height) * p.binning

        self.Images = FileLoader.loadImages(
            input_folder,
            src_width,
            src_height,
            p.num_of_imgs,
            interval=interval,
            binning=p.binning,
        )
        return self.Images

    def Run(self, projections: np.ndarray = None, **kwargs) -> np.ndarray:
        """Reconstruct with the backend selected by FlowMethod."""
        if projections is None:
            if self.Images is None:
                raise ValueError("No projections: call LoadImages first or pass them to Run.")
            projections = self.Images

        tool = self.activeTool_Internal()
        self.Result = tool.run_internal(projections, algo=self.ReconMethod, **kwargs)
        return self.Result

    def GetScanGeometry(self) -> np.ndarray:
        """
        (n, 12) ASTRA cone_vec rows [Sx Sy Sz  Dx Dy Dz  Ux Uy Uz  Vx Vy Vz]
        for the current parameters - the same vectors the Performance flow
        reconstructs from, and what synthetic data must be projected through.
        """
        p = self.Param
        if p.laminography_method == LaminographyMethodConstants.COPLANAR:
            vecs = VxComputeGeometry.computeCoplanarTranslationalLaminographyGeometry(p)
        else:
            vecs = VxComputeGeometry.computeDefaultInclinedLaminographyGeometry(p)
        return vecs.reshape(-1, 12)

    def NormaliseProjections(self,
                             srcPath: str,
                             dstPath: str,
                             toApplyLogTransform : bool = True,
                             savePad: str = None) -> np.ndarray:
        """
        Normalise the projection stack in srcPath and write it to dstPath.

        The whole stack is scaled to [0, 1] with a single global min/max so the
        projections stay comparable to each other, then optionally turned into
        attenuation values with -log(x + eps). Saved as float32 .tif files named
        slice_0000.tif ... in stack order, and the stack is returned.

        savePad is the format spec for the index in those names. Left as None it
        is derived from the stack length (at least "04d", wider for stacks past
        9999) so the names always sort in stack order; pass e.g. "06d" to force a
        width.
        """

        # epsilon constant
        eps: float = 1e-5
        
        # guard srcPath and dstPath
        if not os.path.isdir(srcPath):
            raise NotADirectoryError(f"Source folder does not exist: {srcPath}")
        if os.path.abspath(srcPath) == os.path.abspath(dstPath):
            raise ValueError("dstPath must differ from srcPath.")
        os.makedirs(dstPath, exist_ok=True)

        # load images LoadImages
        stack = self.LoadImages(srcPath)
        if stack.size == 0:
            raise RuntimeError(f"No projections loaded from: {srcPath}")

        # normalise based on global min max to [0,1]
        g_min = float(stack.min())
        g_max = float(stack.max())
        denom = g_max - g_min if g_max != g_min else 1.0
        out = (stack - g_min) / denom

        # log tranform (Beer-Lambert)
        if toApplyLogTransform:
            out = -np.log(out + eps)
        out = out.astype(np.float32)

        # save - pad the index wide enough that the names sort in stack order
        if savePad is None:
            savePad = f"0{max(4, len(str(len(out) - 1)))}d"

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.save_Internal, dstPath, savePad, i, img)
                       for i, img in enumerate(out)]
            for future in as_completed(futures):
                future.result()

        self.Images = out
        return out

    # ----------------------------------------------------------------- private

    def activeTool_Internal(self):
        """Pick the backend that matches FlowMethod."""
        if self.FlowMethod == VxFlowMethod.PERFORMANCE:
            return self.PerformanceTool
        return self.QualityTool

    def save_Internal(self, dstPath: str, savePad: str, idx: int, img: np.ndarray):
        """Write one projection as slice_<idx>.tif under dstPath."""
        out_path = os.path.join(dstPath, f"slice_{idx:{savePad}}.tif")
        if not cv2.imwrite(out_path, img):
            raise RuntimeError(f"Failed to save: {out_path}")
