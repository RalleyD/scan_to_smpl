"""CameraHMR inference wrapper: SMPL params + FoV + 138 dense surface keypoints."""

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

# Path to the CameraHMR submodule (parents[2] = project root from scantosmpl/hmr/camera_hmr.py)
_CAMERAHMR_ROOT = Path(__file__).parents[2] / "external" / "CameraHMR"


def _ensure_importable() -> None:
    root = str(_CAMERAHMR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _patch_smpl_mean_params(smpl_mean_params_path: str) -> None:
    """
    Monkey-patch SMPL_MEAN_PARAMS_FILE in smpl_head_cliff before model instantiation.

    smpl_head_cliff does `from ..constants import SMPL_MEAN_PARAMS_FILE` at import time,
    creating a module-level name. We patch that name so that SMPLTransformerDecoderHead()
    (called inside load_from_checkpoint) picks up the correct path.
    """
    _ensure_importable()
    import core.constants as _constants
    import core.heads.smpl_head_cliff as _smpl_head

    _constants.SMPL_MEAN_PARAMS_FILE = smpl_mean_params_path
    _smpl_head.SMPL_MEAN_PARAMS_FILE = smpl_mean_params_path


@dataclass
class HMROutput:
    """Output from CameraHMR inference on a single image."""

    betas: np.ndarray            # (10,) shape parameters
    body_pose: np.ndarray        # (69,) body pose axis-angle (23 joints × 3)
    global_orient: np.ndarray    # (3,) global orientation axis-angle
    cam_translation: np.ndarray  # (3,) camera-space translation (CLIFF conversion)
    dense_keypoints_2d: np.ndarray   # (138, 2) image pixel coordinates
    dense_keypoint_confs: np.ndarray  # (138,) confidence scores in [0, 1]
    fov_flnet: float | None      # vertical FoV from FLNet in degrees (None if failed)
    fov_exif: float              # vertical FoV from EXIF focal length in degrees
    vertices: np.ndarray | None = None  # (6890, 3) SMPL vertices in camera space


class CameraHMRInference:
    """Loads CameraHMR + DenseKP + FLNet and runs inference on person crops."""

    def __init__(self, config, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        _ensure_importable()
        self._load_models()

    def _load_models(self) -> None:
        cfg = self.config

        # Patch SMPL mean params path before any model instantiation
        _patch_smpl_mean_params(str(cfg.smpl_mean_params_path))

        # CameraHMR — Lightning checkpoint, model_type='smpl'
        from core.camerahmr_model import CameraHMR

        self.camerahmr = CameraHMR.load_from_checkpoint(
            str(cfg.checkpoint_path),
            map_location=self.device,
            strict=True,
            weights_only=False,
        ).to(self.device).eval()

        # DenseKP head only — load checkpoint on CPU, extract head weights, skip backbone.
        # Loading the full DenseKP model (another 7.5 GB ViT-H) to GPU would exhaust VRAM
        # on a 12 GB card. The backbone is already on GPU via CameraHMR; we only need
        # the KeypointsHead (~50 MB) from this checkpoint.
        from core.heads.smpl_head_keypoints import build_keypoints_head  # type: ignore[import]

        densekp_ckpt = torch.load(str(cfg.densekp_path), map_location="cpu", weights_only=False)
        densekp_state = densekp_ckpt.get("state_dict", densekp_ckpt)
        head_state = {
            k[len("head."):]: v
            for k, v in densekp_state.items()
            if k.startswith("head.")
        }
        self.densekp_head = build_keypoints_head().to(self.device).eval()
        self.densekp_head.load_state_dict(head_state, strict=True)

        # FLNet — plain nn.Module, loaded via state_dict
        from core.cam_model.fl_net import FLNet

        self.flnet = FLNet().to(self.device).eval()
        ckpt = torch.load(str(cfg.cam_model_path), map_location=self.device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip 'model.' prefix if present (matches load_valid in CameraHMR utils)
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        self.flnet.load_state_dict(state_dict, strict=True)

        # SMPL model for vertex computation (wireframe overlays)
        import smplx

        self.smpl = smplx.create(
            str(cfg.smpl_model_path),
            model_type="smpl",
            gender="neutral",
            use_face_contour=False,
        ).to(self.device).eval()
        self.smpl_faces = self.smpl.faces  # (13776, 3)

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _prepare_crop(
        self, image: Image.Image, bbox: np.ndarray
    ) -> tuple[torch.Tensor, float, float, float, np.ndarray]:
        """
        Affine-crop the image to 256×256 centred on the bounding box.

        Returns: (crop_tensor, cx, cy, box_size, affine_M)
          - crop_tensor: (3, 256, 256) ImageNet-normalised float32
          - cx, cy: bounding box centre in original image pixels
          - box_size: max(w, h) of the bounding box
          - affine_M: 2×3 affine matrix used for the crop (for inverse mapping)
        """
        x1, y1, x2, y2 = bbox
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        box_size = float(max(x2 - x1, y2 - y1))

        scale = 256.0 / box_size
        M = np.array(
            [[scale, 0.0, 128.0 - cx * scale], [0.0, scale, 128.0 - cy * scale]],
            dtype=np.float64,
        )

        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
        crop = cv2.warpAffine(img_np, M, (256, 256), flags=cv2.INTER_LINEAR)

        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crop_f = crop.astype(np.float32) / 255.0
        crop_f = (crop_f - mean) / std
        tensor = torch.from_numpy(crop_f.transpose(2, 0, 1)).float()

        return tensor, cx, cy, box_size, M

    def _prepare_full_image(self, image: Image.Image) -> torch.Tensor:
        """
        Resize to 256×256 preserving aspect ratio with white padding. For FLNet.
        Returns (3, 256, 256) ImageNet-normalised float32.
        """
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
        h, w = img_np.shape[:2]
        scale = 256.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.ones((256, 256, 3), dtype=np.uint8) * 255
        sx, sy = (256 - new_w) // 2, (256 - new_h) // 2
        canvas[sy : sy + new_h, sx : sx + new_w] = resized

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        canvas_f = canvas.astype(np.float32) / 255.0
        canvas_f = (canvas_f - mean) / std
        return torch.from_numpy(canvas_f.transpose(2, 0, 1)).float()

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rotmat_to_aa(rotmat: torch.Tensor) -> np.ndarray:
        """
        Convert rotation matrices to axis-angle vectors.
        rotmat: (N, 3, 3) → output: (N, 3) numpy float32.
        """
        mat_np = rotmat.detach().cpu().numpy().reshape(-1, 3, 3)
        aa = Rotation.from_matrix(mat_np).as_rotvec().astype(np.float32)
        return aa  # (N, 3)

    @staticmethod
    def _cliff_camera(
        pred_cam: np.ndarray,
        bbox_center: tuple[float, float],
        bbox_size: float,
        focal_length: float,
        img_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Convert weak-perspective prediction [s, tx, ty] to 3D camera translation.
        Matches convert_to_full_img_cam() from CameraHMR/core/utils/train_utils.py.

        img_size: (H, W) in pixels.
        Returns (3,) float32 [tx, ty, tz] in camera space.
        """
        s, tx, ty = float(pred_cam[0]), float(pred_cam[1]), float(pred_cam[2])
        H, W = img_size
        cx_bbox, cy_bbox = bbox_center

        tz = 2.0 * focal_length / (bbox_size * s + 1e-9)
        cx_off = 2.0 * (cx_bbox - W / 2.0) / (s * bbox_size + 1e-9)
        cy_off = 2.0 * (cy_bbox - H / 2.0) / (s * bbox_size + 1e-9)

        return np.array([tx + cx_off, ty + cy_off, tz], dtype=np.float32)

    @staticmethod
    def _denormalize_dense_kps(
        kps_norm: np.ndarray, affine_M: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map (138, 3) normalised crop coords → original image pixel coords.

        kps_norm[:, :2] in [-0.5, 0.5] (x, y); kps_norm[:, 2] = log-sigma confidence.
        Returns: kps_px (138, 2) float32, confs (138,) float32 in [0, 1].
        """
        # Crop pixels [0, 256]
        kps_crop = (kps_norm[:, :2] + 0.5) * 256.0  # (138, 2)

        # Inverse affine back to original image coordinates
        inv_M = cv2.invertAffineTransform(affine_M)
        ones = np.ones((kps_crop.shape[0], 1), dtype=np.float64)
        kps_hom = np.concatenate([kps_crop, ones], axis=1)  # (138, 3)
        kps_orig = (inv_M @ kps_hom.T).T.astype(np.float32)  # (138, 2)

        confs = np.exp(-np.abs(kps_norm[:, 2])).astype(np.float32)  # (138,)
        return kps_orig, confs

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(
        self,
        image: Image.Image,
        bbox: np.ndarray,
        focal_length_px: float,
    ) -> HMROutput:
        """
        Run CameraHMR + DenseKP + FLNet on a single image.

        Args:
            image: PIL image (already EXIF-corrected).
            bbox: (4,) [x1, y1, x2, y2] person bounding box in pixels.
            focal_length_px: EXIF-derived focal length in pixels.

        Returns:
            HMROutput with SMPL params, dense keypoints, FoV estimates, and vertices.
        """
        W, H = image.size  # PIL: (width, height)

        # 1. Prepare crop + build batch dict
        crop_tensor, cx, cy, box_size, affine_M = self._prepare_crop(image, bbox)
        K = torch.tensor(
            [
                [focal_length_px, 0.0, W / 2.0],
                [0.0, focal_length_px, H / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        batch = {
            "img": crop_tensor.unsqueeze(0).to(self.device),
            "box_center": torch.tensor([[cx, cy]], dtype=torch.float32).to(self.device),
            "box_size": torch.tensor([box_size], dtype=torch.float32).to(self.device),
            "img_size": torch.tensor([[H, W]], dtype=torch.float32).to(self.device),
            "cam_int": K.unsqueeze(0).to(self.device),
        }

        # 2. Backbone features — run once, shared by both heads
        x = batch["img"]
        conditioning_feats = self.camerahmr.backbone(x[:, :, :, 32:-32])

        # 3. Build bbox_info for SMPL head (mirrors CameraHMR.forward)
        fl_h = batch["cam_int"][:, 0, 0]
        bbox_info = torch.stack(
            [
                batch["box_center"][:, 0] - batch["img_size"][:, 1] / 2.0,
                batch["box_center"][:, 1] - batch["img_size"][:, 0] / 2.0,
                batch["box_size"],
            ],
            dim=-1,
        )
        bbox_info[:, :2] /= fl_h.unsqueeze(-1)
        bbox_info[:, 2] /= fl_h

        # 4. CameraHMR SMPL head → body params + weak-perspective camera
        pred_smpl_params, pred_cam, _, _ = self.camerahmr.smpl_head(
            conditioning_feats, bbox_info=bbox_info
        )
        pred_smpl_params["global_orient"] = pred_smpl_params["global_orient"].view(1, -1, 3, 3)
        pred_smpl_params["body_pose"] = pred_smpl_params["body_pose"].view(1, -1, 3, 3)[:, :23]

        # 5. DenseKP head → 138 dense surface keypoints
        densekp_out = self.densekp_head(conditioning_feats)
        kps_norm = densekp_out["pred_keypoints"][0].cpu().numpy()  # (138, 3)

        # 6. FLNet on full image → independent FoV cross-check
        fov_flnet = None
        try:
            full_tensor = self._prepare_full_image(image).unsqueeze(0).to(self.device)
            cam_pred, _ = self.flnet(full_tensor)
            vfov_rad = float(cam_pred[0, 1].item())
            fov_flnet = float(np.degrees(vfov_rad))
        except Exception:
            pass  # FLNet failure is non-fatal; EXIF focal length is primary

        # 7. Convert rotation matrices → axis-angle
        go_aa = self._rotmat_to_aa(pred_smpl_params["global_orient"][0])   # (1, 3)
        bp_aa = self._rotmat_to_aa(pred_smpl_params["body_pose"][0])        # (23, 3)

        # 8. CLIFF weak-perspective → 3D camera translation
        cam_trans = self._cliff_camera(
            pred_cam[0].cpu().numpy(), (cx, cy), box_size, focal_length_px, (H, W)
        )

        # 9. Denormalise dense keypoints → image pixel coordinates
        kps_px, kps_confs = self._denormalize_dense_kps(kps_norm, affine_M)

        # 10. SMPL vertices for wireframe overlay (non-fatal if it fails)
        # Use axis-angle tensors — smplx.SMPL expects (B, 3) and (B, 69), not rotation matrices
        vertices = None
        try:
            smpl_out = self.smpl(
                global_orient=torch.from_numpy(go_aa.flatten()).float().unsqueeze(0).to(self.device),
                body_pose=torch.from_numpy(bp_aa.flatten()).float().unsqueeze(0).to(self.device),
                betas=pred_smpl_params["betas"].view(1, -1),
            )
            v = smpl_out.vertices[0].detach().cpu().numpy()  # (6890, 3)
            vertices = v + cam_trans[None, :]
        except Exception:
            pass

        # Vertical FoV from EXIF focal length
        fov_exif = float(np.degrees(2.0 * np.arctan(H / (2.0 * focal_length_px))))

        return HMROutput(
            betas=pred_smpl_params["betas"][0].cpu().numpy().astype(np.float32),
            body_pose=bp_aa.flatten(),   # (69,)
            global_orient=go_aa.flatten(),  # (3,)
            cam_translation=cam_trans,
            dense_keypoints_2d=kps_px,
            dense_keypoint_confs=kps_confs,
            fov_flnet=fov_flnet,
            fov_exif=fov_exif,
            vertices=vertices,
        )

    @torch.no_grad()
    def infer_batch(
        self,
        images: list[Image.Image],
        bboxes: list[np.ndarray],
        focal_lengths: list[float],
    ) -> list[HMROutput]:
        """Process a list of images sequentially (shared model state)."""
        return [
            self.infer(img, bbox, fl)
            for img, bbox, fl in zip(images, bboxes, focal_lengths)
        ]
