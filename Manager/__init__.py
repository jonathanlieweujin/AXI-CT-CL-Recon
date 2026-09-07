import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from Manager.Param import VxParam
from Manager.Constants.flow_method import VxFlowMethod
from Manager.Constants.laminography_method import LaminographyMethodConstants
from Manager.Constants.recon_method import ReconMethodConstants
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

    # Constructor
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

    # Public
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

    def SaveImages(self,
                   output_folder: str,
                   images: np.ndarray = None,
                   savePad: str = None) -> str:
        """
        Write a projection stack to output_folder as slice_XXXX.tif files, one
        per frame, in stack order and at the stack's own dtype. Defaults to the
        cached Images. Returns output_folder.

        Writes are threaded; the index in the name carries the order, not the
        completion order. savePad is the format spec for that index - left as
        None it is derived from the stack length (at least "04d", wider for
        stacks past 9999) so the names always sort in stack order.
        """
        if images is None:
            if self.Images is None:
                raise ValueError("No projections: call LoadImages first or pass them to SaveImages.")
            images = self.Images

        os.makedirs(output_folder, exist_ok=True)
        if savePad is None:
            savePad = f"0{max(4, len(str(max(1, len(images)) - 1)))}d"

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.save_Internal, output_folder, savePad, i, img)
                       for i, img in enumerate(images)]
            for future in as_completed(futures):
                future.result()

        return output_folder

    def Run(self, projections: np.ndarray = None, **kwargs):
        """
        Reconstruct with the backend selected by FlowMethod.

        The volume is left on Result rather than returned.
        """
        if projections is None:
            if self.Images is None:
                raise ValueError("No projections: call LoadImages first or pass them to Run.")
            projections = self.Images

        p = self.Param

        if p.dst_pixel_format not in ("u8", "u16"):
            raise ValueError("Invalid result pixel format")

        # source stack: at least two projections to reconstruct from
        if p.num_of_imgs <= 1:
            raise ValueError("Invalid number of projections")

        # destination volume: at least one slice
        if p.dst_z < 1:
            raise ValueError("Invalid number of volume slices")

        # detector and volume must have a real footprint
        if p.det_width < 1 or p.det_height < 1 or p.dst_x < 1 or p.dst_y < 1:
            raise ValueError("Invalid image dimension input")

        # geometry: magnification and voxel size divide by these
        if p.sod <= 0 or p.sdd <= p.sod:
            raise ValueError("Invalid source/detector distances")
        if p.det_pitch <= 0:
            raise ValueError("Invalid detector pitch")

        # the stack has to match the detector and angles it is reconstructed with
        if projections.ndim != 3:
            raise ValueError("Projections must be a 3D (angles, height, width) stack")
        n, h, w = projections.shape
        if (h, w) != (p.det_height, p.det_width):
            raise ValueError(
                f"Projection size {w}x{h} does not match detector {p.det_width}x{p.det_height}")
        if n != p.num_of_imgs:
            raise ValueError(
                f"Projection count {n} does not match {p.num_of_imgs} angles")

        # iterative algorithms need a positive iteration count
        if self.ReconMethod != ReconMethodConstants.FDK and p.iterations < 1:
            raise ValueError("Invalid iteration count")

        tool = self.activeTool_Internal()
        self.Result = tool.run_internal(projections, algo=self.ReconMethod, **kwargs)

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

        savePad is passed through to SaveImages, which derives it from the stack
        length when left as None.
        """

        # epsilon constant
        eps: float = 1e-5
        
        # guard srcPath and dstPath
        if not os.path.isdir(srcPath):
            raise NotADirectoryError(f"Source folder does not exist: {srcPath}")
        if os.path.abspath(srcPath) == os.path.abspath(dstPath):
            raise ValueError("dstPath must differ from srcPath.")

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

        # save
        self.SaveImages(dstPath, out, savePad=savePad)

        self.Images = out
        return out

    # private / internal

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
