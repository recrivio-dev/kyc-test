# LIVENESS.md — Port the liveness + face-match layers into `ocr-all-classifier`

**Audience:** Claude Code running inside `ocr-all-classifier/`.
**Goal:** add face liveness + ID-photo face-match as a set of HTTP endpoints on the
*existing* `api:app` FastAPI service, so it deploys as ONE container alongside OCR.
No second service, no second venv, no second image.

**Source of truth for the port:** `../live-mini/` (sibling directory).
Read the source files there and port them. Do NOT re-derive the algorithms — they
are tuned and working. This is a port, not a rewrite.

---

## 0. Context: why this is being merged, not deployed separately

The compute budget allows one container. `ocr-all-classifier` already runs a
FastAPI app (`api.py`) behind gunicorn (`gunicorn_conf.py`) with a Docker image
capped at 3 GB / 2 vCPU (`docker-compose.yml`). The liveness code must fit inside
that, sharing the process.

The liveness feature is invoked **during a video call** (LiveKit lives in the
separate `recriauth` service). So the endpoints are called server-to-server by
`recriauth` with frames pulled off the video track — not by a browser form.
Design the API for that caller: it will most easily send **base64 JPEG frames in
a JSON body**, so that variant is mandatory (see §6).

---

## 1. Dependency constraints — VERIFIED, do not deviate

The target venv currently has (checked, not assumed):

```
Python  3.11.14
numpy   2.2.6
cv2     4.13.0
onnxruntime 1.22.1
torch   2.8.0 (CPU wheels, installed via the pytorch CPU index in the Dockerfile)
```

`live-mini/requirements.txt` pins numpy 1.26.4 / opencv 4.10 / onnxruntime 1.19.2
/ mediapipe 0.10.14 / insightface 0.7.3. **Those pins are incompatible and must
NOT be copied.** Use the newer releases instead:

```
mediapipe==1.0.0
insightface==1.0.1
```

A `pip install --dry-run` of exactly those two into the existing venv resolves
cleanly and **does not touch numpy, cv2, onnxruntime, or torch**. It additively
installs:

```
onnx-1.22.0 scikit-image-0.26.0 matplotlib-3.11.1 absl-py-2.5.0 sounddevice-0.5.5
ImageIO-2.37.4 tifffile-2026.3.3 lazy-loader-0.5 ml_dtypes-0.5.4 cffi-2.1.0
pycparser-3.0 contourpy-1.3.3 cycler-0.12.1 fonttools-4.63.0 kiwisolver-1.5.0
pyparsing-3.3.2
```

Append to `requirements.txt` (with a comment explaining the version delta from
live-mini, matching the file's existing commenting style):

```
# Liveness / face-match — MediaPipe FaceLandmarker + InsightFace ArcFace, both
# CPU/ONNX. Newer than live-mini's pins (mediapipe 0.10.14 / insightface 0.7.3)
# because those predate numpy 2 and would force a numpy/opencv downgrade that
# breaks the OCR path. These two resolve additively against numpy 2.2.6,
# opencv 4.13, onnxruntime 1.22.1.
mediapipe==1.0.0
insightface==1.0.1
```

### Dependencies from live-mini you must NOT add

| live-mini dep | Why not | Do instead |
|---|---|---|
| `opencv-contrib-python` | Every `cv2.*` call in the ported code is core opencv (verified: `createCLAHE`, `findHomography`, `Laplacian`, `perspectiveTransform`, `rotate`, `imdecode`, `VideoCapture`, …). No contrib API is used. | Nothing — existing `opencv-python` is enough |
| `loguru` | OCR side uses stdlib `logging.getLogger("uvicorn.error")` | Rewrite `utils/logger.py` — see §4 |
| `pydantic-settings`, `python-dotenv` | OCR side uses a plain `@dataclass` + `os.getenv` in `config.py` | Rewrite `config.py` — see §4 |
| `streamlit`, `plotly`, `matplotlib` (as frontend deps) | The Streamlit UI is not being ported | Skip `live-mini/frontend/` entirely |
| `scipy`, `pandas` | Not actually imported anywhere in `live-mini/backend/` (only listed in its requirements) | Skip |

### numpy 2 compatibility

Grepped the whole of `live-mini/backend/` for numpy-2-removed aliases
(`np.float_`, `np.int_`, `np.bool8`, `np.object`, `np.alltrue`, `np.NaN`,
`np.Inf`, `np.unicode_`, `np.string_`) — **zero hits**. The code is numpy-2
clean as written. Do not "modernise" it beyond what §4 requires.

---

## 2. Target file layout

`ocr-all-classifier` is a **flat module layout** (`api.py`, `config.py`,
`kyc_pipeline.py`, … all at repo root). `live-mini` is a **package layout**
(`backend/...`). Do not flatten liveness into the root — `config.py`,
`app.py`, `main.py` would collide with existing files.

Create ONE new package:

```
ocr-all-classifier/
├── api.py                        # MODIFIED — mount the liveness router
├── config.py                     # UNTOUCHED (OCR settings)
├── kyc_pipeline.py               # UNTOUCHED
├── requirements.txt              # MODIFIED — 2 lines appended
├── Dockerfile                    # MODIFIED — COPY + model cache dirs
├── docker-compose.yml            # MODIFIED — cache volume + memory
└── liveness/                     # NEW package
    ├── __init__.py
    ├── config.py                 # from backend/config.py (rewritten, §4)
    ├── router.py                 # NEW — single APIRouter, all endpoints
    ├── schemas/
    │   ├── __init__.py
    │   ├── requests.py           # from backend/schemas/requests.py
    │   └── responses.py          # from backend/schemas/responses.py
    ├── services/
    │   ├── __init__.py
    │   ├── layer1_capture.py     # from backend/services/  (186 lines)
    │   ├── layer2_identity.py    #                          (261 lines)
    │   ├── layer3_liveness.py    #                          (429 lines)
    │   ├── scoring.py            #                          (78 lines)
    │   ├── pipeline.py           #                          (80 lines)
    │   ├── reporting.py          #                          (146 lines)
    │   └── aadhaar_extractor.py  #                          (98 lines)
    └── utils/
        ├── __init__.py
        ├── face_engine.py        # from backend/utils/      (258 lines)
        ├── face_alignment.py     #                          (51 lines)
        ├── image.py              #                          (191 lines)
        ├── quality.py            #                          (139 lines)
        ├── metrics.py            #                          (123 lines)
        ├── model_manager.py      #                          (174 lines)
        └── logger.py             # rewritten, §4
```

Do NOT port: `live-mini/backend/app.py` (its FastAPI app is replaced by the
existing `api.py`), `live-mini/backend/routes/*` (replaced by `liveness/router.py`),
`live-mini/frontend/**`, `live-mini/backend/models/**` (see §7 — models are
fetched at runtime, not committed).

---

## 3. Port procedure

For each file in the table above:

1. Copy the file verbatim from `../live-mini/<path>`.
2. Rewrite imports: `from backend.` → `from liveness.` (and `import backend.` →
   `import liveness.`). Nothing else in the import lines changes.
3. Leave docstrings, comments, constants, thresholds and algorithm bodies
   **byte-identical**. The tuning comments (e.g. the periocular-corroboration
   rationale in `config.py`, the "largest face not highest-scoring face"
   reasoning in `face_engine.primary_face`) explain non-obvious decisions —
   keep them.
4. Exceptions to (3) are ONLY the three rewrites in §4.

`live-mini/backend/routes/deps.py` is not copied as a module but its three
helpers (`read_frames`, `read_image`, `parse_payload`) ARE needed — inline them
into `liveness/router.py`, with `from fastapi import HTTPException, UploadFile`
and `from liveness.utils.image import decode_image_bytes, decode_video_bytes`.

---

## 4. The three mandatory rewrites

### 4a. `liveness/utils/logger.py` — drop loguru

`live-mini/backend/utils/logger.py` builds a loguru logger with a rotating file
sink into `logs/`. In a container that writes to a non-persistent layer and
duplicates what Docker's json-file driver already captures
(`docker-compose.yml` sets `max-size: 10m`, `max-file: 5`).

Replace the whole module with a stdlib shim that keeps the same import surface,
so no call site changes:

```python
"""Logging for the liveness package.

Uses the stdlib logger the rest of this service already logs through
(`uvicorn.error`), rather than live-mini's loguru + rotating file sink: in a
container the file sink writes to a throwaway layer and duplicates what the
Docker json-file driver already captures.

Loguru's brace-style call signature — logger.info("{} frames", n) — is
preserved via a thin adapter so the ported call sites need no edits.
"""
from __future__ import annotations

import logging

_base = logging.getLogger("uvicorn.error")


class _BraceLogger:
    """Adapts loguru's `logger.info("{}", x)` onto stdlib `%`-style logging."""

    @staticmethod
    def _fmt(msg: str, args: tuple) -> str:
        if not args:
            return str(msg)
        try:
            return str(msg).format(*args)
        except (IndexError, KeyError, ValueError):
            return f"{msg} {args}"

    def debug(self, msg, *a, **kw):    _base.debug(self._fmt(msg, a))
    def info(self, msg, *a, **kw):     _base.info(self._fmt(msg, a))
    def success(self, msg, *a, **kw):  _base.info(self._fmt(msg, a))
    def warning(self, msg, *a, **kw):  _base.warning(self._fmt(msg, a))
    def error(self, msg, *a, **kw):    _base.error(self._fmt(msg, a))
    def exception(self, msg, *a, **kw): _base.exception(self._fmt(msg, a))


logger = _BraceLogger()


def configure_logging() -> None:
    """No-op — gunicorn/uvicorn own the handler configuration."""


__all__ = ["logger", "configure_logging"]
```

`logger.success(...)` must exist — `face_engine.py` and `model_manager.py` call
it. Do not silently drop those call sites.

### 4b. `liveness/config.py` — drop pydantic-settings

`live-mini/backend/config.py` uses `BaseSettings` + a `.env` file. Convert to
the house style of the existing root `config.py` (frozen-ish `@dataclass` +
`os.getenv` helpers). **Preserve every field name, default value and the
explanatory comments verbatim** — those thresholds are tuned:

- decision bands: `threshold_verified=95`, `threshold_high=90`,
  `threshold_medium=80`, `threshold_review=70`
- identity: `similarity_threshold_high=0.40`, `similarity_threshold_low=0.30`
- periocular: `periocular_crop_fraction=0.55`, `periocular_match_threshold=0.42`,
  `periocular_margin=0.08`
- fusion weights: `weight_capture=0.20`, `weight_identity=0.35`,
  `weight_liveness=0.45`
- models: `insightface_model="buffalo_l"`, `det_size=640`,
  `ort_providers="CPUExecutionProvider"`

Two required changes:

- `PROJECT_ROOT` must resolve to the repo root, not the package parent. In the
  new layout `liveness/config.py` is one level below root, so
  `PROJECT_ROOT = Path(__file__).resolve().parent.parent` still gives the repo
  root — verify this after moving, don't assume.
- `models_dir` default changes from `"backend/models"` to a **cache path outside
  the image**, env-overridable, so the ~200 MB of weights land on the mounted
  volume rather than the container layer:
  ```python
  models_dir: str = os.getenv("LIVENESS_MODELS_DIR", "/cache/liveness")
  ```
  For local dev outside Docker, `LIVENESS_MODELS_DIR=./liveness_models` works.

Keep the `models_path` and `providers` properties with identical semantics
(`models_path` must still `mkdir(parents=True, exist_ok=True)`).

Expose a module-level `settings` object so `from liveness.config import settings`
keeps working across every ported file.

### 4c. `liveness/utils/face_engine.py` — trim the InsightFace model pack

**This is the memory-critical change. Do not skip it.**

`buffalo_l` on disk is 601 MB:

```
w600k_r50.onnx   166M   <- recognition, REQUIRED
det_10g.onnx      16M   <- detection,   REQUIRED
2d106det.onnx    4.8M   <- 2D landmarks, used by face_align
1k3d68.onnx      137M   <- 3D landmarks, NOT used by any ported code
genderage.onnx   1.3M   <- NOT used by any ported code
```

`FaceAnalysis(...)` loads every module in the pack by default. `1k3d68` and
`genderage` are dead weight in this service. Restrict them at construction, in
BOTH `face_engine.get_arcface()` and `model_manager.ensure_insightface_model()`:

```python
app = FaceAnalysis(
    name=settings.insightface_model,
    root=root,
    providers=settings.providers,
    # Only detection + recognition + the 2D landmarks face_align needs.
    # The default pack also loads 1k3d68 (137 MB) and genderage, neither of
    # which any code path here calls — and this process shares a 3 GB
    # container with the OCR pipeline.
    allowed_modules=["detection", "recognition", "landmark_2d_106"],
)
```

Verify that `insightface.utils.face_align` (imported by
`utils/face_alignment.py`) still works with that module set — it needs `kps`
from detection, which detection provides. If `landmark_2d_106` turns out to be
unnecessary, drop it too.

Keep `_INFER_LOCK` exactly as-is. MediaPipe `FaceLandmarker` and the
InsightFace app are not safe for concurrent inference on one instance, and
gunicorn's `UvicornWorker` dispatches sync work onto a threadpool.

---

## 5. Known API-drift risks to verify (do not assume they're fine)

The version jumps are large. Check these three things explicitly and fix what
broke — reporting "ported successfully" without running them is a failure.

1. **MediaPipe 0.10.14 → 1.0.0.** `face_engine.get_landmarker()` uses:
   ```python
   from mediapipe.tasks import python as mp_python
   from mediapipe.tasks.python import vision
   vision.FaceLandmarkerOptions(base_options=mp_python.BaseOptions(...),
                                running_mode=vision.RunningMode.IMAGE,
                                num_faces=1,
                                output_face_blendshapes=True,
                                output_facial_transformation_matrixes=True, ...)
   ```
   and `mp.Image(image_format=mp.ImageFormat.SRGB, data=...)`. Confirm those
   symbols and kwarg names still exist in 1.0.0. Also confirm the result fields
   `face_landmarks`, `face_blendshapes`, `facial_transformation_matrixes` are
   unchanged — Layer 3 reads all three (blendshapes drive blink detection, the
   4x4 matrix drives the head-pose challenge).

2. **InsightFace 0.7.3 → 1.0.1.** Confirm `from insightface.app import
   FaceAnalysis`, `from insightface.utils import face_align`, the
   `app.models["recognition"]` dict key used by `get_recognizer()`, and
   `rec.get_feat(list_of_112x112_bgr)` returning `(N, 512)`. Confirm
   `allowed_modules` is still a valid `FaceAnalysis` kwarg.

3. **Model download URLs.** `model_manager.MODEL_SPECS` fetches
   `face_landmarker.task` and `blaze_face_short_range.tflite` from
   `https://storage.googleapis.com/mediapipe-models/...`. Confirm both still
   resolve; if the float16/1 path 404s, bump to the current version path and
   note the change in a comment.

If a mediapipe 1.0.0 API break is not trivially fixable, fall back to
`mediapipe==0.10.35` and re-run `pip install --dry-run` to confirm it still
leaves numpy 2.2.6 alone. **Do not fall back to 0.10.14** — that one forces
numpy < 2 and breaks OCR.

---

## 6. The API surface to build

All endpoints live in `liveness/router.py` as a single `APIRouter`, mounted in
`api.py`:

```python
from liveness.router import router as liveness_router
app.include_router(liveness_router)
```

Keep the existing `/api/v1/...` prefix convention of the OCR endpoints. Full
paths:

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/v1/liveness/check` | multipart: `video` **or** `frames[]`, optional `payload` JSON form field | `LivenessResult` |
| POST | `/api/v1/liveness/frames` | **JSON**: `{challenge, fps, frames: [b64 jpeg, …]}` | `LivenessResult` |
| POST | `/api/v1/liveness/identity` | multipart: `reference`, `probe` | `IdentityResult` |
| POST | `/api/v1/liveness/capture-check` | multipart: `video` or `frames[]`, optional `payload` | `CaptureResult` |
| POST | `/api/v1/liveness/verify` | multipart: `reference` + (`video` or `frames[]`), optional `payload` | `VerificationResponse` |
| POST | `/api/v1/liveness/verify-json` | **JSON**: `{reference: b64, frames: [b64,…], challenge, fps, metadata}` | `VerificationResponse` |
| POST | `/api/v1/liveness/score` | JSON `ScoreRequest` | `DecisionResponse` |
| GET | `/api/v1/liveness/health` | — | model-readiness dict |

The multipart endpoints are a straight port of `live-mini/backend/routes/*.py`
(`capture.py`, `identity.py`, `liveness.py`, `score.py`, `verify.py`,
`health.py`) — same handler bodies, same response models, just re-pathed onto
one router.

### The two JSON endpoints are new — build them

`recriauth` grabs frames off a LiveKit video track server-side. Posting a JSON
array of base64 JPEGs is far simpler for a Node caller than assembling a
multipart body with N file parts. Both JSON endpoints:

- accept `frames: list[str]` of base64-encoded JPEG/PNG (with or without a
  `data:image/jpeg;base64,` prefix — strip it if present),
- reject `len(frames) == 0` and `len(frames) > 90` with 400 (the existing
  `read_frames` cap is `max_frames=90` — keep the same ceiling),
- decode via the same `liveness.utils.image.decode_image_bytes` used by the
  multipart path, so a malformed frame produces the same 400 message,
- otherwise call the exact same service functions. **No duplicated logic
  between the multipart and JSON paths** — both are thin adapters over
  `verify_liveness()` / `run_full_pipeline()`.

Add a `max_bytes` guard on the JSON body (base64 inflates ~33%; 90 frames of
720p JPEG is on the order of 10 MB). Return 413 above the limit rather than
letting the worker OOM.

### Response contract

Reuse `liveness/schemas/responses.py` **unchanged** — the pydantic models
already define the full explainable breakdown (`liveness_score`,
`position_score`, `lighting_score`, `blink_score`, `challenge_score`,
`depth_score`, `motion_score`, `replay_resistance_score`, `blink_detected`,
`challenge_passed`, `depth_passed`, `rppg_bpm`, `reasons[]`).

**Do not wrap them in the OCR `success_envelope`/`failure_envelope` from
`output_schema.py`.** Those are the OCR contract; the liveness contract is
different and already documented by its own schemas. Mixing them would make
both harder to consume. Note this divergence in the router docstring so it
reads as deliberate.

`rppg_bpm` is advisory and must never gate a decision — that is already how the
ported code behaves; do not "improve" it into a gate.

---

## 7. Model provisioning, Docker, and memory

### 7a. Weights are runtime-fetched, never committed

Total: ~3.8 MB MediaPipe + ~190 MB of the trimmed InsightFace pack. Do not
commit them and do not bake them into the image layer. `model_manager.py`
already downloads lazily, atomically (`.part` + rename), with retries and a
sha256 manifest — keep all of that.

Point them at the cache volume via `LIVENESS_MODELS_DIR` (§4b) and add the
volume in `docker-compose.yml` next to the existing `hf_cache` / `torch_cache`:

```yaml
    volumes:
      - hf_cache:/cache/huggingface
      - torch_cache:/cache/torch
      - liveness_models:/cache/liveness    # NEW
      - ./sample-docs:/app/sample-docs

volumes:
  hf_cache:
  torch_cache:
  liveness_models:                          # NEW
```

Add to `docker-compose.yml` environment:

```yaml
      LIVENESS_MODELS_DIR: "/cache/liveness"
```

### 7b. Dockerfile

The `COPY` in the runtime stage is an **explicit file list** — a new package
directory will be silently missing otherwise. Add it:

```dockerfile
COPY --chown=app:app api.py config.py kyc_pipeline.py layout_detector.py \
                     ocr_engines.py output_schema.py preprocessing.py \
                     gunicorn_conf.py /app/
COPY --chown=app:app liveness/ /app/liveness/
```

And create the cache dir with the right owner alongside the existing ones:

```dockerfile
RUN mkdir -p /cache/huggingface /cache/torch /cache/liveness /app/sample-docs \
 && chown -R app:app /cache /app/sample-docs
```

The runtime stage already installs `libgomp1`, `libglib2.0-0`, `libgl1`,
`libsm6`, `libxext6`, `libxrender1` — that covers MediaPipe and onnxruntime.
Verify no additional system lib is needed after the first container build; if
`import mediapipe` fails on a missing `.so`, add it with a comment saying which
import needed it (the existing Dockerfile comments follow that convention).

### 7c. Memory — the real risk

`docker-compose.yml` caps the container at **3 GB** and runs
**`WEB_CONCURRENCY: "2"`**. Each gunicorn worker that serves a liveness request
loads its own MediaPipe graph + InsightFace ONNX sessions on top of the OCR
models it already holds. Rough resident cost per worker after the trim in §4c:
+400–600 MB.

Actions:

1. Measure it. After the port, hit `/api/v1/liveness/verify` on both workers and
   record `docker stats` RSS. Report the actual number — do not estimate in the
   final summary.
2. If total RSS approaches the 3 GB cap, drop to `WEB_CONCURRENCY: "1"` and say
   so explicitly. The existing config comment already documents this exact
   trade-off for Surya; follow that precedent and add a parallel comment for
   liveness.
3. `preload_app = False` in `gunicorn_conf.py` must stay `False` — onnxruntime
   sessions do not survive `fork()` cleanly, and that now applies to the
   InsightFace sessions too. Leave the existing comment; extend it to mention
   InsightFace.

### 7d. Startup self-test and `/healthz`

`api.py` has a startup self-test that drives a synthetic PAN card through the
whole OCR pipeline and makes `/healthz` return **503** unless it passed. That
mechanism exists because a boots-but-500s build previously went green.

Extend the same discipline, but **keep the two subsystems independently
reported**. Do not make an OCR-only deploy unhealthy because liveness models
have not downloaded yet:

```python
{
  "ok": <ocr_ok and liveness_ok>,
  "ocr_available": ...,
  "selftest_ok": ...,
  "selftest_error": ...,
  "liveness_ready": ...,     # models present on disk
  "liveness_error": ...      # populated when provisioning/self-test failed
}
```

Liveness readiness = `model_manager` reports the MediaPipe files present AND at
least one `.onnx` in the InsightFace pack (this is what the ported
`health.py` already checks) AND one synthetic frame ran through
`detect_landmarks()` + `detect_faces()` without raising.

Do NOT block startup on the ~190 MB first download. On a cold volume that would
exceed the 60 s `start_period` in the healthcheck and flap the container. Either
run provisioning in a background task and report `liveness_ready: false` until
it completes, or raise `start_period` to cover the download — pick one, and
state which in your summary.

---

## 8. Verification checklist — all of it, with real output

Do not report done until each of these has actually run:

1. `pip install -r requirements.txt` in the existing venv completes and
   `python -c "import numpy,cv2,onnxruntime;print(numpy.__version__,cv2.__version__,onnxruntime.__version__)"` still prints `2.2.6 4.13.0 1.22.1`.
   **A numpy or cv2 change here means the merge broke OCR — stop and report.**
2. `python -c "import liveness.router"` imports clean (catches the §5 API drift).
3. OCR regression: `POST /api/v1/ocr/aadhaar` with a file from `sample/` returns
   the same JSON as before the change. `/healthz` `selftest_ok` is `true`.
4. `POST /api/v1/liveness/identity` with `../live-mini/aadhar.jpeg` as
   `reference` and `../live-mini/selfi.png` as `probe` returns a match. That
   pair is the known-good fixture the source project tuned against — a
   non-match means the port is broken, most likely in `face_alignment` or the
   trimmed `allowed_modules`.
5. `POST /api/v1/liveness/frames` with a short base64 frame burst returns a
   populated `LivenessResult` (non-zero sub-scores, non-empty `reasons`).
6. `docker compose build && docker compose up` — container reaches healthy, and
   `docker stats` RSS is recorded (§7c).

---

## 9. Out of scope — do not do these

- Do not port `live-mini/frontend/` (Streamlit). The consumer is `recriauth`
  over HTTP.
- Do not modify `kyc_pipeline.py`, `ocr_engines.py`, `preprocessing.py`,
  `layout_detector.py`, `output_schema.py`, or the root `config.py`. The only
  edits outside `liveness/` are: `api.py` (mount router + extend `/healthz`),
  `requirements.txt`, `Dockerfile`, `docker-compose.yml`, and possibly a comment
  in `gunicorn_conf.py`.
- Do not merge liveness face-crop logic with OCR document logic. They overlap in
  subject matter but not in code: `liveness/services/aadhaar_extractor.py`
  states in its own docstring that it performs **no** OCR or field extraction —
  it only locates and crops the ID portrait for embedding. The OCR pipeline does
  field extraction and never touches faces. Keep them separate.
- Do not "clean up" the ported thresholds, the `_INFER_LOCK`, or the
  largest-face-wins selection in `primary_face()`. Each has a comment
  explaining why it is what it is.
- Do not upgrade or downgrade any package already in `requirements.txt`.

---

## 10. Report back

State plainly:

- which of the §5 drift risks actually broke, and how each was fixed;
- the measured RSS per worker and whether `WEB_CONCURRENCY` had to drop to 1;
- the cold-start behaviour chosen in §7d;
- anything in §8 that did **not** pass.
