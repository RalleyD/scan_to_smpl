# Phase 1 Specification: Image Loading, Detection & View Classification

## Context

- **Input**: ~19 JPEG images from a fixed scanner rig, single female subject in T-pose
- **Most images are partial**: subject is mostly visible but not always head-to-toe
- **No empty frames**: every image contains the subject
- **EXIF**: focal length (mm), make/model, width, height, aperture, ISO — but NOT `FocalLengthIn35mmFilm`
- **Quality over speed**: we want high-confidence detections, not throughput

## Deliverables

### 1. `scantosmpl/detection/image_loader.py`

Responsibilities:
- Load JPEG, apply EXIF orientation transpose (PIL `ImageOps.exif_transpose`)
- Extract EXIF metadata into a structured dict
- Compute intrinsic matrix K from focal length:
  - Primary: `f_px = f_mm * image_width / sensor_width_mm` (sensor size from make/model lookup table)
  - Fallback: `f_px = f_mm * image_width / 36.0` (assume full-frame 36mm sensor)
  - Store uncertainty flag when using fallback
- Return: PIL Image (transposed), EXIF dict, CameraParams (with focal_length in pixels)

We have FocalPlaneXResolution in EXIF: compute exact sensor width without a lookup table.

### 2. `scantosmpl/detection/person_detector.py`

Responsibilities:
- Load RT-DETR model (`PekingU/rtdetr_r50vd_coco_o365`) via HuggingFace `transformers`
- Detect persons in each image
- Return highest-confidence person bbox per image
- Since every image has exactly one person, we just take the top detection

Interface:
```python
class PersonDetector:
    def __init__(self, model_id: str, device: str, confidence_threshold: float = 0.5)
    def detect(self, image: PIL.Image) -> Detection | None
    def detect_batch(self, images: list[PIL.Image]) -> list[Detection | None]

@dataclass
class Detection:
    bbox: np.ndarray          # (4,) x1, y1, x2, y2 in pixels
    confidence: float
    image_size: tuple[int, int]  # (width, height)
```

Notes:
- Confidence threshold lowered to 0.5 (from 0.7 in config) since we know every image has a person. We can always filter later.
- Batch processing for efficiency but not critical with only 19 images.

### 3. `scantosmpl/detection/keypoint_detector.py`

Responsibilities:
- Load ViTPose++-Base (`usyd-community/vitpose-plus-base`) via HuggingFace `transformers`
- Takes person crop (from bbox) and estimates 17 COCO keypoints with confidence scores
- Returns keypoints in original image coordinates (not crop coordinates)
- dataset uses mixture-of-expers with 6 dataset heads, index 0 is COCO.
  - pass dataset_index=torch.tensor([0]) to the forward call

Interface:
```python
class KeypointDetector:
    def __init__(self, model_id: str, device: str)
    def detect(self, image: PIL.Image, bbox: np.ndarray) -> KeypointResult
    def detect_batch(self, images: list[PIL.Image], bboxes: list[np.ndarray]) -> list[KeypointResult]

@dataclass
class KeypointResult:
    keypoints: np.ndarray      # (17, 2) x, y in image coords
    confidences: np.ndarray    # (17,) per-keypoint confidence
    bbox: np.ndarray           # (4,) the bbox used for cropping
```

COCO-17 keypoint order:
```
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
```

ViTPose is inferring the full body pose even from partial views, and doing it with high confidence (0.7-0.9+). For SMPL registration, this is actually great:

In Phase 2, CameraHMR will similarly estimate the full SMPL body from partial views
In Tier 2, reprojection loss uses all views — predicted keypoints for cropped body parts still provide useful signal
The confidence scores are preserved in ViewResult, so downstream stages can weight them
The FULL_BODY/PARTIAL distinction was designed for filtering which views are "good enough for HMR". If ViTPose confidently predicts all keypoints — even for partially visible bodies — then the view IS usable for HMR.

Update the ground truth to match what ViTPose reports (most images → FULL_BODY)

"PARTIAL" really means "detector couldn't recover full keypoints" rather than "body is partially visible in image"

### 4. `scantosmpl/detection/view_classifier.py`

Responsibilities:
- Classify each view as FULL_BODY, PARTIAL, or SKIP based on keypoint visibility
- Decision based on which keypoints are detected with confidence > threshold

Classification rules:
```
FULL_BODY: All of the following visible (conf > 0.3):
  - At least one shoulder (5 or 6)
  - At least one hip (11 or 12)
  - At least one ankle (15 or 16)
  - At least one wrist (9 or 10)
  → i.e., the view spans from head/shoulders down to feet

PARTIAL: Person detected AND at least 5 keypoints visible (conf > 0.3)
  but missing some extremities (e.g., feet cropped, arms cut off)
  → Still useful for Phase 2 (contributes visible joints to reprojection)

SKIP: Fewer than 5 keypoints visible, or no person detected
  → Not useful for any tier
```

**Note on confidence threshold**: Using 0.3 (not 0.5) for classification decisions because:
- We want to be generous in what counts as "visible" for classification
- The confidence scores themselves are preserved in ViewResult for downstream weighting
- Phase 2 will use confidence-weighted losses, so low-confidence keypoints get naturally downweighted

Interface:
```python
class ViewClassifier:
    def __init__(self, full_body_conf: float = 0.3, min_partial_keypoints: int = 5)
    def classify(self, keypoints: KeypointResult) -> ViewType
```

### 5. `scantosmpl/detection/pipeline.py`

Orchestrates the full detection pipeline:
```python
class DetectionPipeline:
    def __init__(self, config: DetectionConfig, device: str)
    def process_directory(self, image_dir: Path) -> list[ViewResult]
    def process_image(self, image_path: Path) -> ViewResult
```

Flow per image:
1. Load image + EXIF transpose + extract intrinsics
2. Detect person → bbox
3. Detect keypoints in person crop → 17 COCO keypoints
4. Classify view → FULL_BODY / PARTIAL / SKIP
5. Return populated ViewResult

### 6. Debug output

```
python -c "
from scantosmpl.detection.pipeline import DetectionPipeline
from pathlib import Path

pipeline = DetectionPipeline(device='cuda')
results = pipeline.process_directory(
    Path('data/t-pose/jpg'),
    debug_dir=Path('output/debug/detection'),
)
print(f'\nProcessed {len(results)} images')
for r in results:
    n_vis = int((r.keypoint_confs > 0.3).sum()) if r.keypoint_confs is not None else 0
    print(f'  {r.image_path.name}: {r.view_type.value} ({n_vis}/17 kps)')
"
```

Save to `{output_dir}/debug/detection/`:
- `detections.json` — per-image: bbox, keypoints, confidences, classification, intrinsics
- `{image_name}_keypoints.jpg` — image with overlaid skeleton + bbox + classification label
- `summary.txt` — counts per classification, aggregate keypoint stats

```
Processed 17 images
  cam01_2.JPG: full_body (17/17 kps)
  cam01_6.JPG: full_body (17/17 kps)
  cam02_4.JPG: full_body (16/17 kps)
  cam02_5.JPG: full_body (15/17 kps)
  cam03_5.JPG: full_body (17/17 kps)
  cam03_6.JPG: full_body (16/17 kps)
  cam04_4.JPG: full_body (17/17 kps)
  cam04_5.JPG: full_body (17/17 kps)
  cam05_4.JPG: full_body (17/17 kps)
  cam05_5.JPG: partial (15/17 kps)
  cam05_6.JPG: full_body (17/17 kps)
  cam06_4.JPG: full_body (17/17 kps)
  cam07_4.JPG: full_body (17/17 kps)
  cam07_6.JPG: full_body (17/17 kps)
  cam10_2.JPG: full_body (17/17 kps)
  cam10_4.JPG: full_body (17/17 kps)
  cam10_5.JPG: full_body (17/17 kps)
```

### 7. `tests/test_detection.py`

Tests that work WITHOUT scanner images (unit tests with synthetic data):
- `test_exif_transpose`: All 8 EXIF orientations produce correct output
- `test_intrinsics_extraction`: Known focal length + sensor size → correct K matrix
- `test_intrinsics_fallback`: Missing sensor info → falls back to 36mm with warning
- `test_view_classifier_full_body`: All extremities visible → FULL_BODY
- `test_view_classifier_partial`: Missing ankles → PARTIAL
- `test_view_classifier_skip`: Too few keypoints → SKIP

Tests that REQUIRE scanner images (integration, skipped if no data):
- `test_person_detection`: Every image gets exactly 1 detection
- `test_keypoint_detection`: Reasonable keypoint counts on real images
- `test_classification_vs_ground_truth`: Compare against labelled CSV
- `test_pipeline_end_to_end`: Full pipeline runs on image directory

## Design Decisions

### Why RT-DETR + ViTPose (two-stage) instead of a single model?

- CameraHMR in Phase 2 expects a person crop as input — we need the bbox anyway
- ViTPose is the standard keypoint backbone, well-tested with SMPL pipelines
- Two-stage lets us debug detection vs keypoint issues independently
- Both are native HuggingFace `transformers` — no mmcv/detectron2

### Why not run CameraHMR in Phase 1?

- CameraHMR IS the Phase 2 deliverable
- Phase 1 establishes the detection foundation that Phase 2 builds on
- Keeping them separate lets us validate detection quality before adding HMR complexity

### Focal length: mm to pixels

The scanner cameras report focal length in mm. To build the intrinsic matrix K:

```
f_px = f_mm * image_width_px / sensor_width_mm
```

We need sensor width. Options:
1. **Lookup table** by camera make/model (preferred — exact)
2. **Derive from FocalLengthIn35mmFilm** if available: `sensor_width = 36 * f_mm / f_35mm`
3. **Assume 36mm** (full-frame) as fallback — may be wrong for crop sensors

This is a Phase 1 "best effort". Phase 2's CameraHMR will predict FoV directly, giving us a second (often better) intrinsics estimate.

## Expected Outputs (17 images)

Given the dataset characteristics:
- 16 FULL_BODY views (subject visible head-to-toe)
- 1 PARTIAL views (subject mostly visible, some extremities cropped)
- 0 SKIP views (all images have been manually filtered)

These numbers will be validated against the ground truth CSV.

## Config Changes

Update `DetectionConfig` defaults:
- `person_confidence: 0.5` (from 0.7 — we know every image has a person)
- `keypoint_confidence: 0.3` (from 0.5 — for classification; raw scores preserved)
- Add `full_body_keypoint_groups` field for classification rule customisation
- Add `save_debug: bool = True`
- Add `debug_dir: Path = Path("output/debug/detection")`
