# Latest CT / CL Module

Working code for the latest CT (computed tomography) and CL (computed laminography)
reconstruction module, plus supporting detector pre-processing.

## Layout

| Path | Purpose |
| --- | --- |
| `CT/New/reconTest.py` | Entry point — sets geometry/volume parameters, loads projections, reconstructs, and displays slices with a slider. |
| `CT/New/Util/param.py` | `VxParam` — the shared geometry/acquisition parameter container. |
| `CT/New/Util/load_files.py` | `FileLoader` — reads raw projection stacks with interval and binning. |
| `CT/New/Util/compute_geometry.py` | `VxComputeGeometry` — the shared coplanar/inclined ASTRA `cone_vec` geometry builders used by both backends. |
| `CT/New/Util/flow_method.py` | `VxFlowMethod` — backend selector enum (`QUALITY` / `PERFORMANCE`). |
| `CT/New/Util/laminography_method.py` | `LaminographyMethodConstants` — geometry selector enum (`INCLINED` / `COPLANAR`). |
| `CT/New/Util/recon_method.py` | `ReconMethodConstants` — reconstruction algorithm identifiers (`FDK`, `CGLS`, `SIRT`, …). |
| `CT/New/QualityFlow/get_structure.py` | TIGRE backend (`VxTool`) — builds the geometry and runs reconstruction. |
| `CT/New/PerformanceFlow/get_structure.py` | ASTRA backend (`VxTool`) — same interface, ASTRA toolbox. |
| `CT/normalise.py` | Projection normalisation (flat/dark handling) prior to reconstruction. |
| `Bad Pixel Correction/` | Detector bad-pixel mask generation (`MaskGenerator`) and correction (`BadPixelCorrector`). |

## CT vs CL

Both modes share the same code path; the geometry decides which one you get:

- **CT** — detector untilted (`DetTiltX = 0`), full circular scan.
- **CL** — detector tilted (`DetTiltX` non-zero, e.g. `30`), laminographic geometry.
  Set `LaminographyMethod = LaminographyMethodConstants.INCLINED` for a genuinely
  tilted detector, or `.COPLANAR` to keep a flat detector while the source/detector
  pair traverses the tilted orbit.

Switch backend with `Mode = VxFlowMethod.QUALITY` (TIGRE) or
`Mode = VxFlowMethod.PERFORMANCE` (ASTRA) in `reconTest.py`.

## Requirements

- Python 3.13 (required — the TIGRE build is a `cp313` extension)
- An NVIDIA GPU + driver. No system CUDA Toolkit is needed at runtime: `astra-toolbox`
  ships its own CUDA runtime via pip, and the TIGRE wheel is self-contained. The
  Toolkit (`nvcc`) is only required to *build* TIGRE from source.
- `numpy`, `matplotlib`, `opencv-python`, `scipy`
- One or both reconstruction backends:
  - [TIGRE](https://github.com/CERN/TIGRE) — built from source, see Installation
  - [ASTRA Toolbox](https://astra-toolbox.com/) — `astra-toolbox` on PyPI

## Installation

The environment lives at `CT/New/.venv` and is created with conda (Miniconda),
because ASTRA and TIGRE both need a CUDA-aware Python.

**Python must be 3.13** — the TIGRE build is a compiled `cp313` extension and will
not load on another minor version.

### 1. Create the environment

```powershell
conda create -p "CT\New\.venv" python=3.13 pip -y
```

### 2. Install the packages

```powershell
CT\New\.venv\python.exe -m pip install numpy matplotlib opencv-python scipy astra-toolbox
```

This covers everything except TIGRE:

| Package | Used by |
| --- | --- |
| `numpy` | everywhere |
| `matplotlib` | `reconTest.py` slice viewer |
| `opencv-python` (`cv2`) | `Util/load_files.py` image loading |
| `scipy` | TIGRE / ASTRA dependency |
| `astra-toolbox` | PerformanceFlow backend (bundles its own CUDA runtime + cuFFT) |

### 3. Install TIGRE

TIGRE is **not** on PyPI at the version used here (3.1.3) — the `tigre` package on
PyPI is a different project. It has to be built from the TIGRE source tree, which
requires the CUDA Toolkit (`nvcc`) and MSVC build tools:

```powershell
CT\New\.venv\python.exe -m pip install "<path-to>\TIGRE-master"
```

If a matching wheel has already been built on this machine, pip caches it and the
rebuild is skipped. To install it directly:

```powershell
CT\New\.venv\python.exe -m pip install "$env:LOCALAPPDATA\pip\Cache\wheels\...\tigre-3.1.3-cp313-cp313-win_amd64.whl"
```

TIGRE pulls in `h5py` and `tqdm` as dependencies.

### 4. Verify

```powershell
CT\New\.venv\python.exe -c "import numpy, matplotlib, cv2, scipy, astra, tigre.algorithms; print('ok', astra.use_cuda())"
```

`astra.use_cuda()` must print `True` — if it prints `False`, the GPU is not visible
to ASTRA and reconstruction will fail.

## Running

```powershell
CT\New\.venv\python.exe "CT\New\reconTest.py"
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

## Notes

- Raw projections, reconstructed volumes, and other imaging data are excluded by
  `.gitignore` — keep them outside the repository.
- `SRC` in `reconTest.py` is an absolute, machine-specific path and needs adjusting per
  setup. There are no longer any hardcoded CUDA or TIGRE paths — both are resolved from
  the installed packages in `CT/New/.venv`.
