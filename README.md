# CT / CL AXI Reconstruction Module

Cone-beam CT (computed tomography) and CL (computed laminography) reconstruction
driven from a single parameter object, with two interchangeable GPU backends —
TIGRE ("Quality") and the ASTRA Toolbox ("Performance")

The two flows are held to the same result: `tests/test_flow_equivalence.py`
reconstructs synthetic datasets of known phantoms through both and requires the
volumes to agree. Everything the module needs beyond reconstruction — projection
loading, normalisation, geometry vectors, synthetic data generation — lives in
the package rather than in separate scripts.

## Layout

| Path | Purpose |
| --- | --- |
| `reconTest.py` | Entry point — sets geometry/volume parameters, loads projections, reconstructs, and displays slices with a slider. |
| `Manager/__init__.py` | `VxManager` — the single entry point: owns the parameters, both backends, the projections and the result. `LoadImages` / `SaveImages` read and write a projection stack, `NormaliseProjections` conditions one, `Run` reconstructs. See [Manager](#manager). |
| `Manager/Param/__init__.py` | `VxParam` — the shared geometry/acquisition parameter container. Applies binning to the detector size, pitch and offsets on construction. |
| `Manager/Constants/flow_method.py` | `VxFlowMethod` — backend selector enum (`QUALITY` / `PERFORMANCE`). |
| `Manager/Constants/laminography_method.py` | `LaminographyMethodConstants` — geometry selector enum (`INCLINED` / `COPLANAR`). |
| `Manager/Constants/recon_method.py` | `ReconMethodConstants` — reconstruction algorithm identifiers (`FDK`, `CGLS`, `SIRT`, …). |
| `Manager/Util/compute_geometry.py` | `VxComputeGeometry` — the shared coplanar/inclined ASTRA `cone_vec` geometry builders used by both backends. |
| `Manager/Util/load_images_internal.py` | `FileLoader` — reads raw projection stacks with interval and binning, threaded across the stack. Backs `VxManager.LoadImages`; the matching writes live in `VxManager.SaveImages`. |
| `Manager/Util/redundancy_weighting.py` | `to_apply_redundancy_weighting` — decides when Wang redundancy weights apply (offset detector, upright CT geometry only). |
| `Manager/Tool/Quality/__init__.py` | TIGRE backend (`VxTool`, `VxGeom`) — builds the TIGRE geometry and runs reconstruction. |
| `Manager/Tool/Performance/__init__.py` | ASTRA backend (`VxTool`) — same interface, ASTRA toolbox, plus `plot_geometry` for inspecting the scan vectors. |
| `GenSyntheticData/get_structure.py` | `VxSyntheticData` — builds a phantom, forward-projects it through the declared geometry, and writes a `Corrected/` + `Config/` dataset matching the layout of a real acquisition. |
| `tests/` | Pytest suite — flow equivalence and redundancy weighting, plus the synthetic dataset generator they run against. |

## CT vs CL

Both modes share the same code path; the geometry decides which one you get:

- **CT** — detector upright (`DetTiltX = 90`), full circular scan.
- **CL** — detector tilted away from upright (`DetTiltX` e.g. `51`), laminographic
  geometry. Set `LaminographyMethod = LaminographyMethodConstants.INCLINED` for a
  genuinely tilted detector, or `.COPLANAR` to keep a flat detector while the
  source/detector pair traverses the tilted orbit.

Switch backend with `Mode = VxFlowMethod.QUALITY` (TIGRE) or
`Mode = VxFlowMethod.PERFORMANCE` (ASTRA) in `reconTest.py`.

## Manager

`VxManager` (`Manager/__init__.py`) is the single entry point for a
reconstruction. It owns the parameters, both backends, the loaded projections and
the resulting volume, so a run is construct → load → run:

```python
from Manager import VxManager
from Manager.Param import VxParam
from Manager.Constants.flow_method import VxFlowMethod

manager = VxManager(param, flow_method=VxFlowMethod.QUALITY, left_pad=0, right_pad=0)
manager.LoadImages(SRC, interval=1)
volume = manager.Run()
```

### Capabilities

| Method | What it does |
| --- | --- |
| `LoadImages(input_folder, interval=1)` | Loads the `.tif` projection stack described by `Param` — restores the pre-binning detector size, applies `interval` sub-sampling and `binning`, and resamples to exactly `Param.num_of_imgs` frames. Returns and caches the stack as `Images`. |
| `SaveImages(output_folder, images=None, savePad=None)` | Writes a stack to `output_folder` as `slice_XXXX.tif`, one file per frame, threaded. Defaults to the cached `Images`. The index width is derived from the stack length unless `savePad` forces one, so the names always sort in stack order. |
| `Run(projections=None, **kwargs)` | Reconstructs with the backend selected by `FlowMethod` and the algorithm on `Param.recon_method`. Falls back to the cached `Images` when no stack is passed; extra `kwargs` (`filter`, `niter`, …) go straight to the backend algorithm. Returns and caches `Result`. |
| `GetScanGeometry()` | Returns the `(n, 12)` ASTRA `cone_vec` rows `[Sx Sy Sz  Dx Dy Dz  Ux Uy Uz  Vx Vy Vz]` for the current parameters — the same vectors the Performance flow reconstructs from, and what synthetic phantom data must be forward-projected through. |
| `NormaliseProjections(srcPath, dstPath, toApplyLogTransform=True, savePad=None)` | Loads a raw stack, scales it to `[0, 1]` on a single global min/max, optionally applies the Beer–Lambert `-log(x + eps)` transform, and hands the result to `SaveImages` for `dstPath`. Returns the normalised stack and caches it as `Images`. |

### State

| Attribute | Contents |
| --- | --- |
| `Param` | The `VxParam` geometry/acquisition container. |
| `FlowMethod` | `QUALITY` (TIGRE) or `PERFORMANCE` (ASTRA) — chooses the backend. |
| `ReconMethod` | Algorithm identifier taken from `Param.recon_method` at construction. |
| `Images` | Last loaded/normalised projection stack, `(n, v, u)` float32. |
| `Result` | Last reconstructed volume. |
| `QualityTool` / `PerformanceTool` | Both backends, constructed up front with the same `Param`, laminography method and padding — switching `FlowMethod` needs no rebuild. |

Padding (`left_pad` / `right_pad`) is edge-extrapolated onto the detector by the
backend and the detector centre is shifted to match, which suppresses truncation
artefacts without changing the caller's geometry.

### Geometry

`Param.laminography_method` picks which builder in `VxComputeGeometry` produces
the scan vectors. Both return the same `(n, 12)` ASTRA `cone_vec` rows, so
`GetScanGeometry()` and both backends stay geometry-agnostic — only the vectors
differ.

| | `INCLINED` (default) | `COPLANAR` |
| --- | --- | --- |
| Detector | Tilted, rotates with the rig — pixel axes differ per projection | Stays flat in the world frame — `U = (pitch, 0, 0)`, `V = (0, pitch, 0)` for every projection |
| Motion | Source and detector orbit on a rotation `R = Rx(-θ)·Ry(β)·Rz(-φ)`, θ from `tilt_x`, β from `tilt_y` | Source and detector translate on an XY orbit at fixed axial Z; the tilt only shifts them in-plane by `z·tan(tilt_x)` |
| `tilt_y` | Honoured | Ignored — `tilt_x` alone defines the orbit |
| Covers CT | Yes — CT is the `tilt_x = 90` case of this builder | No |
| Wang redundancy weighting | Available (offset detector, `tilt_x ≈ 90` only) | Never applied |

**Inclined** is the general case: a genuinely tilted detector carried around the
rotation, with the source and detector placed along opposite ends of the rig axis
and the pixel axes taken from the same rotation. Because a `tilt_x` of 90° is
just an untilted detector, ordinary circular CT falls out of this builder rather
than needing a path of its own.
Ref: Van Aarle, W., Palenstijn, W. J., Cant, J., Janssens, E., Bleichrodt, F., Dabravolski, A., De Beenhouwer, J., Batenburg, K. J., & Sijbers, J. (2016). Fast and flexible X-ray tomography using the ASTRA toolbox. Optics Express, 24(22), 25129–25145. https://doi.org/10.1364/OE.24.025129 

**Coplanar** models the other way to get laminographic sampling: keep the
detector flat and physically traverse the source/detector pair across the tilted
orbit. `SOD`/`SDD` are read as axial distances along Z, and detector offsets move the centre along the fixed `U`/`V` axes.
Ref: Porsch, F. (2010). Computed Laminography for X-ray Inspection of Lightweight Constructions. https://www.academia.edu/download/82596285/mo3a3.pdf

## Requirements

- Python 3.13 (required — the TIGRE build is a `cp313` extension)
- An NVIDIA GPU + driver. No system CUDA Toolkit is needed at runtime: `astra-toolbox`
  ships its own CUDA runtime via pip, and the TIGRE wheel is self-contained. The
  Toolkit (`nvcc`) is only required to *build* TIGRE from source.
- `numpy`, `matplotlib`, `opencv-python`, `scipy`, and `pytest` for the test suite
- One or both reconstruction backends:
  - [TIGRE](https://github.com/CERN/TIGRE) — built from source, see Installation
  - [ASTRA Toolbox](https://astra-toolbox.com/) — `astra-toolbox` on PyPI

## Pre-Requisites

The environment lives at `.venv` in the repository root and is created with conda
(Miniconda), because ASTRA and TIGRE both need a CUDA-aware Python.

**Python must be 3.13** — the TIGRE build is a compiled `cp313` extension and will
not load on another minor version.

### 1. Create the environment

```powershell
conda create -p ".venv" python=3.13 pip -y
```

### 2. Install the packages

```powershell
.venv\python.exe -m pip install numpy matplotlib opencv-python scipy pytest astra-toolbox
```

This covers everything except TIGRE:

| Package | Used by |
| --- | --- |
| `numpy` | everywhere |
| `matplotlib` | `reconTest.py` slice viewer, `plot_geometry` |
| `opencv-python` (`cv2`) | projection loading and normalisation |
| `scipy` | TIGRE / ASTRA dependency |
| `pytest` | `tests/` |
| `astra-toolbox` | Performance backend (bundles its own CUDA runtime + cuFFT) |

### 3. Install TIGRE

TIGRE is **not** on PyPI at the version used here (3.1.3) — the `tigre` package on
PyPI is a different project. It has to be built from the TIGRE source tree, which
requires the CUDA Toolkit (`nvcc`) and MSVC build tools:

```powershell
.venv\python.exe -m pip install "<path-to>\TIGRE-master"
```

If a matching wheel has already been built on this machine, pip caches it and the
rebuild is skipped. To install it directly:

```powershell
.venv\python.exe -m pip install "$env:LOCALAPPDATA\pip\Cache\wheels\...\tigre-3.1.3-cp313-cp313-win_amd64.whl"
```

TIGRE pulls in `h5py` and `tqdm` as dependencies.

### 4. Verify

```powershell
.venv\python.exe -c "import numpy, matplotlib, cv2, scipy, astra, tigre.algorithms; print('ok', astra.use_cuda())"
```

`astra.use_cuda()` must print `True` — if it prints `False`, the GPU is not visible
to ASTRA and reconstruction will fail.

## Running

```powershell
.venv\python.exe reconTest.py
```

Before running, check the parameters at the top of `reconTest.py`:

- `SRC` — folder of normalised raw projections
- `DetU` / `DetV` / `DetZ` — detector width, height, and number of projections
- `DetPitch`, `Binning`, `Interval`
- `SOD` / `SDD` — source-to-object and source-to-detector distance (mm)
- `DetTiltX` / `DetTiltY`, `DetOffsetU` / `DetOffsetV`
- `VolX` / `VolY` / `VolZ` and `VolMid*` — output volume size and centre
- `LEFT_PAD` / `RIGHT_PAD` — projection padding, used to suppress truncation artefacts
- `Iterations` — iteration count for the iterative algorithms (ignored by `FDK`)

## Tests

```powershell
.venv\python.exe -m pytest tests
```

`test_flow_equivalence.py` reconstructs three synthetic datasets — CT (inclined,
`DetTiltX = 90`), CL (inclined, `51`) and Coplanar (`51`) — through both backends,
then asserts the two volumes correlate and that each recovers the phantom the
projections came from. It needs the datasets on disk first:

```powershell
.venv\python.exe tests\make_synthetic_data.py
```

That writes roughly 4.5 GB of projections per dataset under `DATASET_ROOT`, an
absolute path set at the top of both files.

## Notes

- Raw projections, reconstructed volumes, and other imaging data are excluded by
  `.gitignore` — keep them outside the repository.
- `SRC` in `reconTest.py` and `DATASET_ROOT` in `tests/` are absolute,
  machine-specific paths and need adjusting per setup. There are no hardcoded CUDA
  or TIGRE paths — both are resolved from the installed packages in `.venv`.