# ScanToSMPL pipeline: model and tool availability in March 2026

**The multi-view uncalibrated SMPL registration pipeline you've designed is buildable today, but requires careful tool selection.** The most significant development since the pipeline's conception is that **PromptHMR (CVPR 2025) has replaced HMR2.0 as the state-of-the-art single-image HMR**, while **CameraHMR now provides the full perspective camera model your Tier 2 self-calibration needs**. For the multi-view fusion stages, **U-HMR is the only calibration-free method with released weights that actually works**, and for Tier 3 point cloud alignment, **NVIDIA Kaolin has overtaken PyTorch3D as the practical choice** for differentiable chamfer distance. Below is a component-by-component breakdown of what's available, what works, and what to avoid.

---

## HMR2.0 is effectively dead — use PromptHMR or CameraHMR instead

The original 4D-Humans repo (`shubham-goel/4D-Humans`) has **43 total commits and zero maintainer responses** to issues since mid-2023. Pre-trained weights still auto-download to `~/.cache/4DHumans` on first run, and an HMR2.0a checkpoint remains at `cs.utexas.edu/~pavlakos/4dhumans/hmr2a_model.tar.gz`, so the model is technically usable. But the installation experience is poor: the hard dependency on **detectron2** — which has no pre-built wheels for PyTorch 2.0+ — forces a source build requiring exact CUDA toolkit matching. A known GPU memory leak (issue #14, unfixed since June 2023) causes VRAM to grow monotonically across batches. The `neural-renderer-pytorch` dependency also fails to build on current toolchains (issue #174, Oct 2025).

**PromptHMR** (`yufu-wang/PromptHMR`, 336 stars) is the direct successor from the same MPI/Meshcapade lineage. It achieves **36.6mm PA-MPJPE on 3DPW** versus HMR2.0's 44.4mm — a 17.6% reduction. Code and weights (image + video models trained on BEDLAM1/BEDLAM2) are on GitHub via Google Drive. Critically, it supports **PyTorch 2.4.0+cu121 and 2.6.0+cu126**, processes full images rather than crops, and outputs SMPL-X. For your pipeline's Tier 1, PromptHMR gives better per-view estimates before fusion.

**CameraHMR** (`pixelite1201/CameraHMR`, 238 stars) is the upgrade you specifically need for Tier 2. It predicts **field-of-view with 5–7° error** via a HumanFoV module trained on ~500K Flickr images with EXIF data, enabling a full perspective camera model. The **138 dense surface keypoints** (COMA-sampled from SMPL vertices, trained on BEDLAM/AGORA) provide far richer 2D–3D correspondences for PnP than the 17–44 sparse joints from HMR2.0. Code is complete (demo + training + evaluation + CamSMPLify). Weights require free registration at `camerahmr.is.tue.mpg.de` — not on HuggingFace. Installation targets **Python 3.10, PyTorch 2.0.0, CUDA 11.8**. Wide downstream adoption (FastHMR, PhySIC, BEDLAM2, PromptHMR all reference it) confirms reliability.

**TokenHMR** (`saidwivedi/TokenHMR`) sits between HMR2.0 and PromptHMR chronologically — CVPR 2024, weights released June 2024, requires PyTorch 2.1.0 + CUDA 11.8. It still needs detectron2. For a new pipeline, skip it and go directly to PromptHMR or CameraHMR.

---

## Multi-view methods: U-HMR is practical, HeatFormer requires calibration, MUC is fragile

**HeatFormer** (CVPR 2025, `kyotovision-public/HeatFormer`, MIT license) achieves state-of-the-art calibrated multi-view results — **29.5mm MPJPE on Human3.6M at iteration 4**. Weights are on Google Drive, training used Human3.6M + MPI-INF-3DHP, and a Singularity container ensures reproducibility. However, **it fundamentally requires pre-calibrated cameras**: heatmaps are projected using known intrinsics and extrinsics. It cannot replace your Tier 1+2 unless you add a separate calibration step first. Training demands an A100 (80GB) with batch size 8 for 4 views. Useful if you later move to a fixed-camera studio setup, but not for your calibration-free architecture.

**U-HMR** (`XiaobenLi00/U-HMR`, 20 stars) is the most directly relevant method. Its Camera-Body Decoupling (CBD) architecture separates camera pose estimation (CPE module, shared MLP) from body mesh recovery (Arbitrary View Fusion via transformer decoder with SMPL query token). It handles **arbitrary numbers of uncalibrated views** — exactly your Tier 1+2 scenario. Weights are available via OneDrive and BaiduDisk (password: `uhmr`). The codebase inherits from 4D-Humans/SPIN, so expect similar dependency challenges. Trained and evaluated on Human3.6M, MPI-INF-3DHP, and TotalCapture. Main limitation: **single-person only**, and dataset preparation is complex (requires processing through multiple toolboxes). No GPU requirements documented, but the transformer decoder architecture suggests **8–12GB minimum** for inference.

**MUC** (AAAI 2025, `AbsterZhu/MUC`, 24 stars) is calibration-free and outputs SMPL-X, but has severe practical barriers. It's pinned to **PyTorch 1.12.0 + CUDA 11.3** with a `mmcv-full==1.7.1` dependency — a deprecated package that fails on modern GPUs (RTX 4090/5090 require CUDA ≥ 11.8). Only 7 commits, 0 forks, and **it's unclear whether full MUC-specific weights (JRN/SRN networks) are included** in the OneDrive download versus just the SMPLer-X backbone. The OneDrive provides `smpler_x_b32.tar` but the README doesn't enumerate pretrained MUC checkpoints explicitly. Treat as research reference only, not production tooling.

---

## The 2025 landscape has several important new entrants

**HSfM** (`hongsukchoi/HSfM_RELEASE`, CVPR 2025 Highlight) combines HMR2.0 + DUSt3R for joint "people, places, cameras" reconstruction from multi-view images. This is directly relevant — it solves multi-view human + scene + camera estimation simultaneously, which could potentially replace your entire three-tier architecture with a single forward pass.

**Human3R** (`fanegg/Human3R`) is a unified feed-forward 4D human-scene reconstruction model running at **15 FPS with 8GB GPU memory**, trained on BEDLAM in ~1 day on a single GPU. Checkpoints available on HuggingFace (`faneggg/human3r`). Designed for monocular video but the architecture may generalize.

**BLADE** (CVPR 2025, `NVlabs/blade`) from NVIDIA solves for camera pose + focal length + SMPL(-X) parameters from a single view using accurate depth estimation. Strong for close-range/perspective-distorted images typical of body scanners. Code and weights released.

**SMPLest-X** (`SMPLCap/SMPLest-X`, TPAMI 2025) is the scaled-up successor to SMPLer-X with a **Huge model (8.2GB)** released February 2025. Docker support available. This is the current ceiling for single-image expressive (hands+face) body estimation.

**DiffProxy** (arXiv 2025) uses diffusion-based generative priors with epipolar attention for multi-view consistent human mesh recovery, trained entirely on synthetic data. Code promised after acceptance — **not available yet**.

---

## ViTPose is stable on HuggingFace, RTMPose is faster

ViTPose was officially integrated into HuggingFace Transformers on **January 8, 2025** and is stable in v5.1.0+. All four sizes of ViTPose++ (the MoE multi-dataset variant) have weights under the `usyd-community` organization:

- **ViTPose++-Small** (33M params, ~66MB FP16) — any GPU
- **ViTPose++-Base** (100M params, ~200MB FP16) — 4GB+ GPU
- **ViTPose++-Large** (400M params, ~800MB FP16) — 6–8GB GPU
- **ViTPose++-Huge** (900M params, ~1.8GB FP16) — 8–12GB GPU

The recommended pipeline pairs **RT-DETR** (`PekingU/rtdetr_r50vd_coco_o365`) for person detection with ViTPose for keypoint estimation. ViTPose++ supports multiple dataset heads via a `dataset_index` parameter (COCO, AIC, MPII, AP-10K, APT-36K, COCO-WholeBody). For your pipeline, ViTPose++-Base at ~1–2GB total VRAM is the practical sweet spot.

**RTMPose** is dramatically faster — **430+ FPS** on a GTX 1660 Ti versus ~100–200 FPS for ViTPose-B — at matching accuracy (75.8% COCO AP for both RTMPose-m and ViTPose-B). It's **not in HuggingFace Transformers** but available via `rtmlib` (`Tau-J/rtmlib`) as ONNX models with zero mmcv dependency. For multi-camera processing where throughput matters, RTMPose is worth considering. **DWPose** (ICCV 2023, distilled RTMPose) provides whole-body 133-keypoint estimation needed for SMPL-X fitting.

---

## SMPL model files still require manual registration

The download process at `smpl.is.tue.mpg.de` and `smpl-x.is.tue.mpg.de` remains active. Registration is self-service with near-instant approval — create account, agree to license, download. Separate registrations are needed for SMPL, SMPL+H, and SMPL-X. The `pip install smplx` package installs only the PyTorch code; model `.pkl`/`.npz` files must be downloaded separately and placed in a `models/` directory. An extra step is needed for SMPL/SMPL+H: removing Chumpy objects from `.pkl` files using tools in `smplx/tools/`.

**The license is strictly non-commercial.** Commercial use requires licensing through **Meshcapade** (`sales@meshcapade.com`), which holds an exclusive sublicense from Max-Planck-Innovation. SMPL is patented (WO2016207311A1). An unofficial but complete mirror exists at `lithiumice/models_hub` on HuggingFace containing SMPL, SMPL-H, SMPL-X, and MANO files — technically a license violation, but widely referenced in community projects.

No truly open-source (MIT/Apache) alternative to SMPL exists. **STAR** (Sparse Trained Articulated Regressor) from the same group offers improved shape space but carries the same non-commercial restriction.

---

## Skip PyTorch3D for chamfer distance — use Kaolin or a standalone implementation

PyTorch3D's latest release is **0.7.8 (September 2024)** with official support only through **PyTorch 2.4.1**. PyTorch 2.5+ is unsupported, with open issues (#1949, #1962) and no ETA. PyPI only has macOS CPU-only wheels. The conda channel works for supported versions but channel mixing causes solver failures. A community hero, **MiroPsota**, provides pre-built wheels through PyTorch 2.8.0 at `miropsota.github.io/torch_packages_builder` — the de facto solution if you must use PyTorch3D.

For chamfer distance specifically, better options exist:

- **NVIDIA Kaolin** (v0.18.0, Apache 2.0): `kaolin.metrics.pointcloud.chamfer_distance()` is GPU-optimized, differentiable, and pip-installable with pre-built wheels supporting **PyTorch 2.1–2.8**. Also provides `sided_distance`, `f_score`, and `point_to_mesh_distance`. This is the recommended choice for your Tier 3.

- **ThibaultGROUEIX/ChamferDistancePytorch**: JIT-compiled CUDA implementation with a pure-Python fallback. ~529MB VRAM for a 32×2000×3 batch (CUDA) versus ~2.5GB (Python fallback). Used as a submodule in DavidBoja/SMPL-Fitting.

- **Pure PyTorch** via `torch.cdist`: Zero dependencies, fully differentiable, works on any version. For your use case (scan vs **SMPL's 6,890 vertices**), even a 500K-point scan produces a distance matrix of only ~13GB — feasible on a 24GB GPU. For larger scans, subsample to 50K points first.

---

## DavidBoja/SMPL-Fitting is the closest existing scan-to-SMPL pipeline

For Tier 3 specifically, **DavidBoja/SMPL-Fitting** (99 stars, actively maintained in 2024) provides a complete optimization pipeline: chamfer distance + landmark losses + regularization, with optional per-vertex displacement fitting (SMPL+D). It includes a Docker container, Plotly visualization dashboard, and uses `pyTorchChamferDistance` as a submodule. Input is a PLY scan + landmark JSON. A companion repo, **DavidBoja/SMPL-Anthropometry**, extracts body measurements from the fitted result.

**RVH_Mesh_Registration** (`bharat-b7/RVH_Mesh_Registration`) from MPI offers more fitting methods (IP-Net, LoopReg, standard optimization, point-cloud-specific fitting for noisy Kinect data) but depends on the legacy `psbody.mesh` library, which is itself painful to install. Research reference quality only.

No scanner-specific (Artec Eva, Structure Sensor) open-source SMPL tools exist. The workflow is: export mesh as PLY/OBJ → clean with Open3D → provide landmarks → run SMPL fitting. **Meshcapade** offers commercial scan-to-SMPL services and is the only company with SMPL commercial licensing (100+ enterprise customers, $6M seed funding).

For calibrated multi-view capture environments, **EasyMocap** (`zju3dv/EasyMocap`) and **XRMoCap** provide mature detection→triangulation→SMPL fitting pipelines, and **MvSMPLfitting** (`boycehbz/MvSMPLfitting`) offers multi-view SMPLify-X fitting.

---

## Recommended component stack for your three-tier architecture

Based on all findings, here is the most practical component selection for each tier:

**Tier 1 — Per-image HMR + parameter-space fusion:**
Use **CameraHMR** for per-view SMPL estimation with perspective-aware camera prediction. Its 138 dense surface keypoints feed directly into Tier 2. If CameraHMR's registration requirement is a blocker, use **PromptHMR** (Google Drive weights, no registration) with ViTPose++-Base for separate 2D keypoints. Budget **8–12GB VRAM** per view for inference. Person detection via RT-DETR (HuggingFace).

**Tier 2 — PnP self-calibration + triangulation:**
CameraHMR's HumanFoV module gives you field-of-view estimates (5–7° error), and its 138 dense 2D keypoints paired with the predicted SMPL mesh provide rich PnP correspondences. Alternatively, **consider U-HMR** as a potential single-model replacement for= Tier 1+2 — it jointly estimates camera poses and SMPL parameters from arbitrary uncalibrated views via its CBD + CPE + AVF architecture. Weights available on OneDrive.

**Tier 3 — Point cloud alignment via chamfer distance:**
Use **NVIDIA Kaolin** (`pip install kaolin`) for differentiable chamfer distance, with the SMPL mesh (~6,890 vertices) as target and subsampled scan (~10K–50K points) as source. For the full optimization pipeline, adapt **DavidBoja/SMPL-Fitting**'s approach (chamfer + landmark + regularization losses). Pure PyTorch `torch.cdist` is viable for prototyping if Kaolin installation is problematic.

**Also investigate HSfM** (CVPR 2025 Highlight) — it may collapse your entire three-tier pipeline into a single multi-view "humans, scenes, cameras" reconstruction step using HMR2.0 + DUSt3R, though at the cost of architectural control.

## References

[references doc](REFERENCES.md)