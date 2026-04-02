# REFERENCES.md — ScanToSMPL Research Bibliography

## Core Pipeline Components

### Per-View Human Mesh Recovery (Tier 1)

**CameraHMR: Aligning People with Perspective** (Primary)
Patel, P. & Black, M.J.
International Conference on 3D Vision (3DV), 2025
- Paper: https://arxiv.org/abs/2411.08128
- Code: https://github.com/pixelite1201/CameraHMR
- Weights: https://camerahmr.is.tue.mpg.de (free registration)
- Key: Full perspective camera model, HumanFoV (5-7° FoV error), 138 dense surface keypoints, CamSMPLify

**PromptHMR: Promptable Human Mesh Recovery** (Fallback)
Wang, Y., Sun, Y., Patel, P., Black, M.J.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025
- Paper: https://arxiv.org/abs/2504.06397
- Code: https://github.com/yufu-wang/PromptHMR
- Project: https://yufu-wang.github.io/phmr-page/
- Weights: Google Drive (linked in README, no registration)
- Key: 36.6mm PA-MPJPE on 3DPW, SMPL-X output, PyTorch 2.4-2.6

**HMR2.0 / 4DHumans: Reconstructing and Tracking Humans with Transformers**
Goel, S., Pavlakos, G., Rajasegaran, J., Kanazawa, A., Malik, J.
IEEE/CVF International Conference on Computer Vision (ICCV), 2023
- Paper: https://arxiv.org/abs/2305.20091
- Code: https://github.com/shubham-goel/4D-Humans
- HuggingFace: https://huggingface.co/spaces/brjathu/HMR2.0
- Note: Effectively unmaintained. detectron2 dependency issues on PyTorch 2.0+. Superseded by CameraHMR/PromptHMR.

**TokenHMR: Advancing Human Mesh Recovery with a Tokenized Pose Representation**
Dwivedi, S.K., Sun, Y., Patel, P., Feng, Y., Black, M.J.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024
- Paper: https://arxiv.org/abs/2404.16158
- Code: https://github.com/saidwivedi/TokenHMR
- Key: Tokenized pose, still requires detectron2

---

### Multi-View Uncalibrated Methods

**U-HMR: Human Mesh Recovery from Arbitrary Multi-view Images**
Li, X. et al., 2024
- Paper: https://arxiv.org/abs/2403.12434
- Code: https://github.com/XiaobenLi00/U-HMR
- Weights: OneDrive + BaiduDisk (password: uhmr)
- Key: Camera-Body Decoupling (CBD), arbitrary uncalibrated views, transformer decoder fusion

**MUC: Mixture of Uncalibrated Cameras for Robust 3D Human Body Reconstruction**
Zhu, A. et al.
AAAI Conference on Artificial Intelligence, 2025
- Paper: https://arxiv.org/abs/2403.05055
- Code: https://github.com/AbsterZhu/MUC
- Note: Pinned to PyTorch 1.12 + CUDA 11.3. Impractical on modern hardware.

**HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery**
Matsubara, Y. & Nishino, K.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025
- Paper: https://arxiv.org/abs/2412.04456
- Code: https://github.com/kyotovision-public/HeatFormer
- Project: https://vision.ist.i.kyoto-u.ac.jp/research/heatformer/
- Weights: Google Drive (linked in README)
- Key: Neural optimizer, heatmap alignment, 29.5mm MPJPE on H36M. Requires calibrated cameras.

**EasyRet3D: Uncalibrated Multi-View Multi-Human 3D Reconstruction and Tracking**
Yin, J.O., Li, T., Wang, J., Yuille, A.
IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025
- Paper: https://openaccess.thecvf.com/content/WACV2025/papers/Yin_EasyRet3D_Uncalibrated_Multi-View_Multi-Human_3D_Reconstruction_and_Tracking_WACV_2025_paper.pdf
- Key: HMR2.0 + ground plane constraints for automatic camera calibration from human body

**Multiview Human Body Reconstruction from Uncalibrated Cameras**
Yu, Z. et al.
- Paper: https://openreview.net/pdf?id=7vlIVOBKarp
- Key: DensePose-based semantic alignment across views without geometric calibration

**Progressive Multi-View Human Mesh Recovery with Self-Supervision**
AAAI Conference on Artificial Intelligence, 2023
- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/25144/24916
- Key: Consensus/diversity sampling, multi-view balance via consistency weighting

---

### Future Investigation

**HSfM: Reconstructing People, Places, and Cameras** (CVPR 2025 Highlight)
Choi, H. et al.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025
- Code: https://github.com/hongsukchoi/hsfm_release
- Key: Joint people + scene + camera reconstruction from multi-view. Could replace entire 3-tier pipeline.

**Human3R: Everyone Everywhere All at Once**
Fan, E. et al., 2025
- Paper: https://arxiv.org/abs/2510.06219
- Code: https://github.com/fanegg/Human3R
- HuggingFace: https://huggingface.co/faneggg/human3r
- Key: Unified 4D human-scene reconstruction, 15 FPS, 8GB GPU

**BLADE: Single-view Body Mesh Estimation through Accurate Depth Estimation** (CVPR 2025)
NVIDIA
- Code: https://github.com/NVlabs/blade
- Key: Solves camera pose + focal length + SMPL(-X) from single view. Good for close-range/scanner images.

**DiffProxy: Multi-View Human Mesh Recovery via Diffusion-Generated Dense Proxies**
- Code: https://github.com/wrk226/DiffProxy
- Key: Diffusion priors + epipolar attention. Code promised after acceptance.

---

## Body Models

**SMPL: A Skinned Multi-Person Linear Model**
Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., Black, M.J.
ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 34(6), 2015
- Website: https://smpl.is.tue.mpg.de/
- Code: https://github.com/vchoutas/smplx
- PyPI: https://pypi.org/project/smplx/
- License: Non-commercial (commercial via Meshcapade)

**SMPL-X: Expressive Body Capture: 3D Hands, Face, and Body from a Single Image**
Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman, A.A.A., Tzionas, D., Black, M.J.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019
- Website: https://smpl-x.is.tue.mpg.de/
- Paper: https://download.is.tue.mpg.de/smplx/SMPL-X.pdf

**SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation**
Cai, Z. et al.
Advances in Neural Information Processing Systems (NeurIPS), 2023
- Code: https://github.com/SMPLCap/SMPLer-X (now https://github.com/MotrixLab/SMPLer-X)

**SMPLest-X: Ultimate Scaling for Expressive Human Pose and Shape Estimation**
Yin, Z. et al.
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025
- Code: https://github.com/SMPLCap/SMPLest-X (now https://github.com/MotrixLab/SMPLest-X)
- Key: Huge model (8.2GB), Docker support

**STAR: A Sparse Trained Articulated Human Body Regressor**
Osman, A.A.A., Bolkart, T., Black, M.J.
European Conference on Computer Vision (ECCV), 2020

**Meshcapade** — Commercial SMPL licensing
- Website: https://meshcapade.com/smpl/
- Blender addon: https://github.com/Meshcapade/SMPL_blender_addon
- Wiki: https://github.com/Meshcapade/wiki/blob/main/wiki/SMPL.md

---

## 2D Pose Estimation

**ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation**
Xu, Y., Zhang, J., Zhang, Q., Tao, D.
Advances in Neural Information Processing Systems (NeurIPS), 2022
- Code: https://github.com/ViTAE-Transformer/ViTPose
- HuggingFace: https://huggingface.co/docs/transformers/en/model_doc/vitpose
- Weights (HuggingFace): `usyd-community/vitpose-plus-*` (S/B/L/H variants)

**ViTPose++: Vision Transformer for Generic Body Pose Estimation**
Xu, Y., Zhang, J., Zhang, Q., Tao, D.
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023
- Paper: https://arxiv.org/abs/2212.04246
- Key: MoE multi-dataset variant, 6 expert heads

**easy_ViTPose**
- Code: https://github.com/JunkyByte/easy_ViTPose
- Key: Simplified inference wrapper, ONNX support, multiple skeleton formats

**RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose**
Jiang, T. et al., 2023
- Paper: https://arxiv.org/abs/2303.07399
- Lightweight wrapper: https://github.com/Tau-J/rtmlib (ONNX, zero mmcv dependency)
- Key: 430+ FPS on GTX 1660 Ti, matching ViTPose-B accuracy. Alternative if speed matters.

---

## SMPL Fitting & Registration

**SMPL-Fitting: Fit an SMPL body model to a scan**
Bojanić, D., 2024
- Code: https://github.com/DavidBoja/SMPL-Fitting
- Key: Chamfer + landmark + regularisation losses, SMPL+D, Docker, Plotly viz. Reference for Tier 3.

**SMPL-Anthropometry: Measure the SMPL body model**
Bojanić, D.
- Code: https://github.com/DavidBoja/SMPL-Anthropometry
- Key: Extract body measurements from fitted SMPL

**RVH Mesh Registration: Code to fit SMPL model to scans**
Bhatnagar, B.L. et al.
- Code: https://github.com/bharat-b7/RVH_Mesh_Registration
- Papers: Combining Implicit Function Learning and Parametric Models (ECCV 2020), LoopReg (NeurIPS 2020)
- Note: Legacy psbody.mesh dependency

**MultiviewSMPLifyX**
Zheng, Z.
- Code: https://github.com/ZhengZerong/MultiviewSMPLifyX
- Key: Multi-view extension of SMPLify-X. Requires pre-calibrated cameras.

**DiffICP: Fully-Differentiable ICP in PyTorch**
- Code: https://github.com/fa9r/DiffICP
- Key: Includes ICPSMPL for non-rigid ICP with SMPL-X

---

## Multi-View Calibration & Triangulation

**Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics**
Pagnon, D., Domalain, M., Reveret, L.
Journal of Open Source Software, 2022; Sensors, 2021/2022
- Code: https://github.com/perfanalytics/pose2sim
- Key: Camera calibration, synchronisation, triangulation, filtering. Architecture reference.

**EasyMocap: Make human motion capture easier**
Zju3dv
- Code: https://github.com/zju3dv/EasyMocap
- Key: Mature calibrated multi-view detection→triangulation→SMPL pipeline

**XRMoCap: Multi-view Single-person SMPL Estimator**
- Docs: https://xrmocap.readthedocs.io/en/latest/estimation/mview_sperson_smpl_estimator.html
- Key: Camera selection, triangulation, SMPLify fitting pipeline

**DMMR: Dynamic Multi-Person Mesh Recovery From Uncalibrated Multi-View Cameras**
3DV, 2021
- Code: https://github.com/boycehbz/DMMR
- Key: Spatiotemporal multi-camera calibration using freely moving people

---

## 3D Libraries & Distance Metrics

**NVIDIA Kaolin: A PyTorch Library for Accelerating 3D Deep Learning Research** (Recommended)
- Code: https://github.com/NVIDIAGameWorks/kaolin
- Docs: https://developer.nvidia.com/kaolin
- API: `kaolin.metrics.pointcloud.chamfer_distance()`, `sided_distance()`, `f_score()`
- Key: pip-installable, PyTorch 2.1-2.8, Apache 2.0

**PyTorch3D** (Avoided — install issues)
- Code: https://github.com/facebookresearch/pytorch3d
- Install: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
- PyPI: https://pypi.org/project/pytorch3d/ (macOS CPU-only wheels)
- Conda: https://anaconda.org/pytorch3d/pytorch3d
- Community wheels (PyTorch 2.5+): https://miropsota.github.io/torch_packages_builder
- Note: Last release 0.7.8 (Sep 2024). No PyTorch 2.5+ support. Known build issues (#1962).

**ChamferDistancePytorch** (Lightweight standalone)
Groueix, T.
- Code: https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
- Key: JIT-compiled CUDA + pure Python fallback. Used as submodule in SMPL-Fitting.

**Open3D**
- Website: https://www.open3d.org/
- Key: Point cloud I/O, preprocessing, ICP, visualisation

---

## Datasets & Training Data

**BEDLAM: A Synthetic Dataset of Bodies Exhibiting Detailed Lifelike Animated Motion**
Black, M.J., Patel, P., Tesch, J., Yang, J.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023

**BEDLAM 2.0** (NeurIPS 2025)
- Render tools: https://github.com/PerceivingSystems/bedlam2_render

**Human3.6M**
Ionescu, C., Papava, D., Olaru, V., Sminchisescu, C.
IEEE TPAMI, 2014
- Key: Standard multi-view evaluation benchmark

**3DPW: Recovering Accurate 3D Human Pose in The Wild**
Von Marcard, T., Henschel, R., Black, M.J., Rosenhahn, B., Pons-Moll, G.
ECCV, 2018

**AGORA: Avatars in Geography Optimized for Regression Analysis**
Patel, P. et al., CVPR 2021

---

## Foundational Methods

**SMPLify: Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image**
Bogo, F., Kanazawa, A., Lassner, C., Gehler, P., Romero, J., Black, M.J.
European Conference on Computer Vision (ECCV), 2016

**HMR: End-to-end Recovery of Human Shape and Pose**
Kanazawa, A., Black, M.J., Jacobs, D.W., Malik, J.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018
- Project: https://akanazawa.github.io/hmr/

**SPIN: Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop**
Kolotouros, N., Pavlakos, G., Black, M.J., Daniilidis, K.
IEEE/CVF International Conference on Computer Vision (ICCV), 2019

**GenHMR: Generative Human Mesh Recovery**
Saleem et al., 2024
- Paper: https://arxiv.org/abs/2412.14444

**FastHMR: Accelerating Human Mesh Recovery** (WACV 2026)
- Code: https://github.com/TaatiTeam/FastHMR

---

## Other Relevant Work

**VolumetricSMPL: A Neural Volumetric Body Model**
- Paper: https://arxiv.org/abs/2506.23236
- Key: Lightweight SDF add-on for SMPL, collision detection

**FPCR-Net: Front Point Cloud Regression Network**
Sensors, 2025
- Paper: https://www.mdpi.com/1424-8220/25/15/4808
- Key: End-to-end SMPL regression from single front point cloud

**MPL: Lifting 3D Human Pose from Multi-view 2D Poses**
Arnaud et al., 2024
- Paper: https://arxiv.org/abs/2408.10805
- Key: Decoupled 2D detection + 3D pose lifting, AMASS-based training

**SegFit: Robust SMPL-X Fitting with Body Part Segmentation**
ICLR 2025 (Withdrawn)
- OpenReview: https://openreview.net/forum?id=HW8xnOUcBx
- Key: Segmentation-guided fitting concept. Unavailable implementation.

**RoGSplat: Learning Robust Generalizable Human Gaussian Splatting** (CVPR 2025)
- Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_RoGSplat_Learning_Robust_Generalizable_Human_Gaussian_Splatting_from_Sparse_Multi-View_CVPR_2025_paper.pdf

**SynCHMR: Synergistic Global-Space Camera and Human Reconstruction from Videos** (CVPR 2024)
- Key: Human-aware Metric SLAM + scene-conditioned SMPL denoising

**PhySIC: Physically Plausible 3D Human-Scene Interaction** (SIGGRAPH Asia 2025)
- Code: https://github.com/YuxuanSnow/Phy-SIC

---

## Tools & Infrastructure

| Tool | URL | Used For |
|------|-----|----------|
| detectron2 | https://github.com/facebookresearch/detectron2 | **Avoided** — no PyTorch 2.0+ wheels |
| RT-DETR | `PekingU/rtdetr_r50vd_coco_o365` on HuggingFace | Person detection |
| OpenCV | https://pypi.org/project/opencv-python/ | PnP, image I/O, calibration |
| trimesh | https://pypi.org/project/trimesh/ | Mesh I/O |
| Click | https://pypi.org/project/click/ | CLI framework |
| PyVista | https://pyvista.org/ | 3D visualisation |
| rtmlib | https://github.com/Tau-J/rtmlib | RTMPose ONNX (speed alternative to ViTPose) |
