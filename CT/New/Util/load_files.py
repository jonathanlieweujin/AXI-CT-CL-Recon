import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class FileLoader:
    @staticmethod
    def loadImages(input_folder,
                   srcWidth,
                   srcHeight,
                   numberOfImg,
                   interval: int = 1,
                   binning: int = 1):
        image_files = sorted([
            f for f in os.listdir(input_folder)
            if f.lower().endswith('.tif')
        ])

        image_files = image_files[::interval]

        if len(image_files) >= numberOfImg:
            selected_indices = np.linspace(0, len(image_files) - 1, numberOfImg, dtype=int)
            selected_files = [image_files[i] for i in selected_indices]
        else:
            print("Insufficient images — loading all available.")
            selected_files = image_files

        b = max(1, int(binning))
        out_w = srcWidth // b
        out_h = srcHeight // b
        images = np.zeros((len(selected_files), out_h, out_w), dtype=np.float32)

        def _load(idx, filename):
            path = os.path.join(input_folder, filename)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Failed to load image: {filename}")
                return idx, None
            if b > 1:
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            return idx, img

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_load, i, f): i
                for i, f in enumerate(selected_files)
            }
            for future in as_completed(futures):
                idx, img = future.result()
                if img is not None:
                    images[idx] = img

        return images
