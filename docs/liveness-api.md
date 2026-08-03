# Liveness & Face-Match — API Reference

Base URL: `http://127.0.0.1:8000` (production: same host and port as the OCR API — liveness ships inside the **same** container and process).

These endpoints prove (a) a live human is physically present in front of the camera, and (b) that human is the person in a reference ID photo. They are designed to be called **server-to-server** during a video call — `recriauth` pulls frames off the LiveKit track and posts them here.

> **Response contract differs from OCR.** The OCR endpoints wrap everything in the `{data, status_code, message_code, message, success}` envelope. Liveness endpoints do **not** — they return their score objects at the top level, and signal failure with an HTTP status plus a FastAPI `{"detail": ...}` body. This is deliberate: the liveness contract is an explainable score breakdown, and nesting it inside the OCR envelope would make both harder to consume. Branch on the HTTP status code, not on a `success` field.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Concepts](#2-concepts)
3. [Choosing an endpoint](#3-choosing-an-endpoint)
4. [Endpoints](#4-endpoints)
5. [Response objects](#5-response-objects)
6. [Errors](#6-errors)
7. [Integration recipes](#7-integration-recipes)
8. [Operational notes](#8-operational-notes)

---

## 1. Quick start

### Running the service

There is **no separate liveness server**. The liveness routes are mounted on
the OCR app (`app.include_router(liveness_router)` in `api.py`), so the one
command that starts the OCR backend starts these endpoints too:

```bash
# local dev, from the repo root
LIVENESS_MODELS_DIR=./liveness_models uvicorn api:app --reload --port 8000

# in Docker — LIVENESS_MODELS_DIR is already baked into the image
uvicorn api:app --port 8000
# or: docker compose up
```

`LIVENESS_MODELS_DIR` defaults to `/cache/liveness` (the container path). Set
it to the repo-local `./liveness_models` when running outside Docker, or the
weights get cached to a path that doesn't exist. See
[Environment variables](#environment-variables) for the rest.

Then wait for the face weights before calling anything — on a cold cache they
download in the background (see [Model provisioning](#model-provisioning)):

```bash
curl -s http://127.0.0.1:8000/healthz | jq '.liveness_ready, .liveness_models'
# true
# {"face_landmarker": true, "blaze_face": true, "insightface": true}
```

`/api/v1/liveness/health` reports the same readiness scoped to this subsystem.

### The call

The single call most integrations need — full pipeline, JSON transport:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/liveness/verify-json \
  -H "Content-Type: application/json" \
  -d '{
    "reference":  "<base64 JPEG of the ID photo>",
    "frames":     ["<base64 JPEG>", "<base64 JPEG>", "..."],
    "challenges": ["turn_left", "turn_right", "blink"],
    "mirrored":   false,
    "fps": 15.0,
    "frame_timestamps_ms": [0, 66.7, 133.4, "..."],
    "metadata": {"device_name": "FaceTime HD Camera", "width": 1280, "height": 720, "fps": 15}
  }'
```

```json
{
  "status": "verified_high_confidence",
  "capture_score": 83.5,
  "identity_score": 71.5,
  "liveness_score": 91.2,
  "final_score": 90.6,
  "confidence": "high",
  "reasons": [
    {"code": "REAL_CAMERA", "message": "Real hardware timing observed with natural frame jitter.", "severity": "low"},
    {"code": "CHALLENGE_SEQUENCE_PASS", "message": "All challenges completed in the requested order.", "severity": "low"}
  ],
  "warnings": [],
  "fraud_indicators": {"virtual_camera": false, "replay_attack": false, "multiple_faces": false},
  "capture":  { "...": "CaptureResult" },
  "identity": { "...": "IdentityResult" },
  "liveness": { "...": "LivenessResult" },
  "quality":  { "capture_quality_index": 79.4, "confidence_penalty_pct": 3.0, "issues": [] },
  "notes": [],
  "processing": {"total_ms": 2104, "capture_ms": 41, "identity_ms": 986, "liveness_ms": 1041, "decision_ms": 36},
  "models": {
    "face_detector":  "insightface_buffalo_l_scrfd_det_10g",
    "face_embedding": "insightface_buffalo_l_arcface_w600k_r50",
    "face_landmarks": "mediapipe_face_landmarker_v2"
  },
  "pipeline_version": "v1.1.0"
}
```

Gate on `status` (or on `final_score` against your own bands). Everything else is there to explain the decision to a reviewer.

### Reasons carry a code and a message

`reasons` (and `quality.issues`) are objects, not strings:

```json
{"code": "NO_BLINK", "message": "No clear blink detected.", "severity": "medium"}
```

Branch on `code` — it is stable and safe to store, group and count. Show `message`
to the candidate — it is written as user-facing guidance and gets reworded whenever
the copy improves. Never string-match the message.

`severity` reflects how much a human reviewer should care, not how much the score
moved: `low` is informational or positive, `medium` is a capture problem a retry
could fix, `high` is an active fraud indicator or a hard failure. The full code
list lives in `liveness/services/reason_codes.py` — treat the values as a
published contract that only ever gets appended to.

### Fraud indicators only cover what is actually measured

```json
"fraud_indicators": {"virtual_camera": false, "replay_attack": false, "multiple_faces": false}
```

There are deliberately **no** `deepfake_detected`, `face_swap_detected` or
`document_tampering` keys. No model in this pipeline evaluates them, and a
hardcoded `false` would read to a reviewer as "checked and clean". Absent is
honest; false is not. If those checks are added later they will appear as new
keys — so treat a missing key as "not evaluated", never as "passed".

---

## 2. Concepts

### The three layers

| Layer | Question | Weight in `final_score` |
|---|---|---|
| **1 · Capture integrity** | Is this stream from a real camera, or injected / replayed? | 0.20 |
| **2 · Identity** | Is this the same person as the ID photo? | 0.35 |
| **3 · Liveness** | Is a live human physically present right now? | 0.45 |

`final_score = 0.20·capture + 0.35·identity + 0.45·liveness`, clamped to 0–100.

### Decision bands

| `status` | `final_score` |
|---|---|
| `verified` | ≥ 95 |
| `verified_high_confidence` | ≥ 90 |
| `verified_medium_confidence` | ≥ 80 |
| `manual_review` | ≥ 70 |
| `rejected` | < 70 |

`confidence` (`high` / `medium` / `low`) is separate: it downgrades when capture quality was poor or identity did not match, so a high score taken under bad conditions still reads as lower-confidence.

### Challenges

`challenges` is the ordered list of actions the user was prompted to perform:

`turn_left` · `turn_right` · `blink` · `look_up` · `look_down`

The production flow uses `["turn_left", "turn_right", "blink"]`. Head turns are scored from landmark yaw/pitch, `blink` from the eye-blink signal; it is expressed as a challenge so it can hold a position in the order — "blink *after* you turn" is stronger proof than "blink at some point".

Each challenge is located independently by its **peak moment**, then the peaks are checked for increasing order. `liveness.challenge_sequence` reports every step:

```json
"challenge_sequence": [
  {"challenge": "turn_left",  "passed": true, "score": 92.0, "peak_frame_index": 7},
  {"challenge": "turn_right", "passed": true, "score": 88.4, "peak_frame_index": 21},
  {"challenge": "blink",      "passed": true, "score": 95.1, "peak_frame_index": 34}
],
"challenge_sequence_passed": true
```

`challenge_sequence_passed` is `true` only when **every** challenge passed **and** they happened in the requested order. Order is evidence in itself: a spoof that splices separately-captured actions, or replays a generic "person moving" clip, routinely produces every requested action but not on this session's cue order. Out-of-order halves `challenge_score` and emits `CHALLENGE_SEQUENCE_OUT_OF_ORDER`.

> **Send ONE continuous recording.** The ordering check spans the whole take. Three separate clips, one per challenge, defeat the entire point — and Layer 1's timing and entropy analysis also degrades when it sees stitched segments.

The singular `challenge` field still works for a one-challenge request and behaves exactly as before.

#### `mirrored` — send it

Front-facing cameras are usually presented mirrored, which swaps left and right. From landmark yaw alone a mirrored `turn_left → turn_right` is **bit-for-bit identical** to an unmirrored `turn_right → turn_left`. These are not merely hard to tell apart — they are the same signal, and no amount of analysis separates them.

So when your sequence contains both `turn_left` and `turn_right`:

| `mirrored` | Behaviour |
|---|---|
| `true` / `false` | Ambiguity resolved. Full ordering enforced, including left-vs-right. |
| omitted / `null` | Both hypotheses scored, better reading kept, but left-vs-right ordering is **excluded** from the check and the response carries `CHALLENGE_SEQUENCE_ORDER_PARTIAL` (severity `medium`). |

Ordering against `blink` and the vertical moves always holds — a horizontal flip cannot move a blink in time.

Your client knows the answer: `mirrored` is `true` when the video element carries a `scaleX(-1)` transform **and** the frames were captured from that transformed element. Capturing from the raw `MediaStream` while only the *preview* is CSS-mirrored means `mirrored: false` — the pixels you send were never flipped.

### Frame timestamps

`frame_timestamps_ms` are the real client-side capture times, in milliseconds. Layer 1 scores their jitter: genuine hardware has small but non-zero jitter, whereas a perfectly regular cadence reads as **deterministic, i.e. injected**.

- **Send real timestamps** taken from a clock at capture time.
- **Do not synthesise** an evenly-spaced sequence — that actively scores *worse* than sending none.
- **Send an empty list** when you genuinely have no timing data; the timing check then reports neutral (60/100) and says so in `reasons`.

### Capture quality compensation

Poor lighting, blur or a small face reduce **confidence** with an explanation rather than hard-failing a genuine user. `quality.confidence_penalty_pct` quantifies it and `quality.issues` lists what to tell the user — same `{code, message, severity}` shape as `reasons`, e.g. `{"code": "LOW_LIGHT", "message": "Lighting is too dark; increase ambient light.", "severity": "medium"}`.

### rPPG

`liveness.rppg_bpm` is an advisory pulse estimate from facial colour variation. It is **never** used in scoring and is frequently `null`. Do not gate on it.

---

## 3. Choosing an endpoint

```
Need a single yes/no on a live user vs. their ID?
  └─ frames as base64 in JSON  ──────────────►  POST /api/v1/liveness/verify-json   ★ recommended
  └─ frames/video as file parts  ────────────►  POST /api/v1/liveness/verify

Need one layer only (a step-by-step wizard, or you already have the ID match)?
  ├─ "is this stream real?"  ────────────────►  POST /api/v1/liveness/capture-check
  ├─ "same person as the ID?"  ──────────────►  POST /api/v1/liveness/identity
  └─ "is a live human present?"  ────────────►  POST /api/v1/liveness/check   (multipart)
                                                POST /api/v1/liveness/frames  (JSON)

Already have three layer scores and just need the decision?
  └─────────────────────────────────────────►  POST /api/v1/liveness/score

Checking whether the service can serve liveness at all?
  └─────────────────────────────────────────►  GET  /api/v1/liveness/health
                                                GET  /healthz
```

**Prefer the JSON endpoints from a server-side caller.** Assembling a multipart body with N file parts from Node is awkward; posting an array of base64 strings is not. They are thin adapters over the same service code — the multipart and JSON paths return byte-identical results for the same input.

---

## 4. Endpoints

### 4.1 `POST /api/v1/liveness/verify-json`

Full pipeline (Layers 1 → 2 → 3 → fused decision) over a JSON body.

**Request** — `application/json`

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `reference` | string | ✅ | — | Base64 JPEG/PNG of the ID photo or portrait. A `data:image/jpeg;base64,` prefix is accepted and stripped. |
| `frames` | string[] | ✅ | — | Base64 JPEG/PNG frames **in capture order**. 1–90 items. |
| `challenge` | enum | | `turn_left` | See [Challenges](#challenges). |
| `fps` | number | | `15.0` | Must be > 0. Used by the rPPG band only. |
| `frame_timestamps_ms` | number[] | | `[]` | Real capture times. See [Frame timestamps](#frame-timestamps). |
| `metadata` | object | | `{}` | Camera metadata — see below. |

`metadata` fields, all optional: `device_name`, `device_id`, `driver` (strings), `width`, `height`, `fps` (numbers ≥ 0). Device names and drivers are matched against a virtual-camera blacklist (OBS Virtual Camera, ManyCam, DroidCam, v4l2loopback, …).

**Response** `200` → [`VerificationResponse`](#54-verificationresponse)

---

### 4.2 `POST /api/v1/liveness/verify`

Same pipeline, `multipart/form-data`.

| Part | Type | Required | Notes |
|---|---|---|---|
| `reference` | file | ✅ | ID photo. |
| `video` | file | ⬥ | A short clip. Sampled **evenly across the whole clip** (max 90 frames), so a blink late in the clip is not cut off. |
| `frames` | file[] | ⬥ | Repeat the part once per frame, in capture order. |
| `payload` | string | | JSON string: `{challenge, fps, frame_timestamps_ms, metadata}`. |

⬥ Supply **either** `video` **or** `frames`; `video` wins if both are present.

**Response** `200` → [`VerificationResponse`](#54-verificationresponse)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/liveness/verify \
  -F "reference=@aadhaar.jpg" \
  -F "frames=@f00.jpg" -F "frames=@f01.jpg" -F "frames=@f02.jpg" \
  -F 'payload={"challenge":"turn_left","fps":15,"frame_timestamps_ms":[0,66.7,133.4],"metadata":{"device_name":"FaceTime HD Camera"}}'
```

---

### 4.3 `POST /api/v1/liveness/frames`

Layer 3 only, JSON transport.

**Request** — same as `verify-json` minus `reference` and `metadata`.

**Response** `200` → [`LivenessResult`](#53-livenessresult)

---

### 4.4 `POST /api/v1/liveness/check`

Layer 3 only, `multipart/form-data`. Parts: `video` **or** `frames[]`, plus optional `payload` (`{challenge, fps}`).

**Response** `200` → [`LivenessResult`](#53-livenessresult)

---

### 4.5 `POST /api/v1/liveness/identity`

Layer 2 only — ArcFace match between a reference and a probe.

| Part | Type | Required | Notes |
|---|---|---|---|
| `reference` | file | ✅ | ID photo. A **whole card** works: the portrait is located and cropped automatically, across 0°/90°/180°/270° rotations. |
| `probe` | file | ✅ | A live still of the user. |

**Response** `200` → [`IdentityResult`](#52-identityresult)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/liveness/identity \
  -F "reference=@aadhaar.jpg" -F "probe=@selfie.png"
```

```json
{
  "identity_score": 71.7,
  "similarity": 0.421,
  "threshold": 0.4,
  "match": true,
  "landmark_score": 78.0,
  "quality_score": 81.4,
  "reasons": [
    "Reference image auto-rotated 270° to locate the face.",
    "Portrait region extracted from the document and restored before matching.",
    "Strong embedding match (similarity 0.42 >= 0.40).",
    "Facial geometry remained highly consistent (age-stable)."
  ]
}
```

Note `threshold` is **quality-adaptive** — a degraded or aged reference gets a more forgiving floor. Compare `similarity` against the returned `threshold`, never against a constant of your own; or simply read `match`.

---

### 4.6 `POST /api/v1/liveness/capture-check`

Layer 1 only. Parts: `video` **or** `frames[]`, plus optional `payload` (`{frame_timestamps_ms, metadata}`).

**Response** `200` → [`CaptureResult`](#51-captureresult)

---

### 4.7 `POST /api/v1/liveness/score`

Fuse three already-computed layer scores into a decision without re-running anything. Useful for a multi-step wizard that collected each layer separately.

**Request** — `application/json`

| Field | Type | Required | Default | Range |
|---|---|---|---|---|
| `capture_score` | number | ✅ | — | 0–100 |
| `identity_score` | number | ✅ | — | 0–100 |
| `liveness_score` | number | ✅ | — | 0–100 |
| `identity_match` | boolean | | `true` | `false` forces `confidence: "low"` |
| `quality_index` | number | | `100.0` | 0–100 |

**Response** `200` → [`DecisionResponse`](#55-decisionresponse)

---

### 4.8 `GET /api/v1/liveness/health`

Reports whether the model weights are present on disk. Pure filesystem check — loads nothing, safe to poll.

```json
{
  "status": "ok",
  "app": "Enterprise Liveness Verification",
  "version": "1.0.0",
  "models_ready": true,
  "details": {"face_landmarker": true, "blaze_face": true, "insightface": true}
}
```

### `GET /healthz` (service-wide)

The container's health probe, covering **both** subsystems:

```json
{
  "ok": true,
  "ocr_available": true,
  "selftest_ok": true,
  "selftest_error": null,
  "liveness_ready": true,
  "liveness_error": null,
  "liveness_required": false,
  "liveness_models": {"face_landmarker": true, "blaze_face": true, "insightface": true}
}
```

- `liveness_ready` — weights are on disk **and** a synthetic frame ran through both the MediaPipe and InsightFace graphs without raising. This is a real inference check, not a `stat()`.
- `ok` gates on **OCR only** by default, because the liveness weights download in the background after boot and an OCR-only deploy must not go unhealthy while that happens. Set `LIVENESS_REQUIRED=true` to fold liveness into `ok` (returns `503` until liveness can serve).

**Poll `liveness_ready` before routing liveness traffic to a freshly started container.**

---

## 5. Response objects

### 5.1 `CaptureResult`

| Field | Type | Notes |
|---|---|---|
| `capture_score` | 0–100 | `0.60·timing + 0.20·metadata + 0.20·entropy` |
| `timing_score` | 0–100 | Frame-cadence jitter analysis |
| `metadata_score` | 0–100 | Virtual-camera / blacklisted-driver detection |
| `entropy_score` | 0–100 | Temporal entropy; catches repeated/replayed frames |
| `injection_score` | 0–100 | **Higher = more injection risk** (inverted vs. the others) |
| `median_fps` | number | Derived from `frame_timestamps_ms` |
| `timing_variance` | number | Normalised inter-frame jitter (MAD / median delta). Near `0` is deterministic, i.e. injected. |
| `camera_type` | enum | `physical` · `virtual` · `unknown`. **`unknown` means no metadata was supplied**, not that the camera is clean. |
| `frame_count` | int | Frames received in the request |
| `selected_frames` | int | Frames actually analysed after sampling. Lower than `frame_count` means downsampling, not a short recording. |
| `reasons` | Reason[] | `{code, message, severity}` — see §1 |

### 5.2 `IdentityResult`

| Field | Type | Notes |
|---|---|---|
| `identity_score` | 0–100 | `0.70·similarity + 0.20·landmark + 0.10·quality` |
| `similarity` | −1…1 | ArcFace cosine similarity |
| `threshold` | number | Quality-adaptive decision threshold |
| `match` | boolean | The decision. May be `true` slightly below `threshold` when an age-stable channel (facial geometry or periocular) corroborates — `reasons` always says when this happened. |
| `landmark_score` | 0–100 | Facial-geometry stability |
| `quality_score` | 0–100 | Lower of the two images' capture quality |
| `embedding_distance` | number | `1 - similarity`. Provided because reviewers commonly think in distance. |
| `multiple_faces` | boolean | More than one face in the **live capture**. The reference is not checked — ID cards legitimately carry a ghost portrait. |
| `reasons` | Reason[] | `{code, message, severity}` |

### 5.3 `LivenessResult`

| Field | Type | Notes |
|---|---|---|
| `liveness_score` | 0–100 | Fused; replay resistance gates it multiplicatively |
| `position_score` | 0–100 | Face centring / size / roll |
| `lighting_score` | 0–100 | Brightness + contrast |
| `blink_score` | 0–100 | Blink transient from blendshapes (EAR fallback) |
| `challenge_score` | 0–100 | Mean of the per-challenge scores; halved when the sequence is out of order |
| `depth_score` | 0–100 | Non-planarity — a flat photo/screen maps by a homography, a real face does not |
| `motion_score` | 0–100 | Involuntary micro-movement |
| `replay_resistance_score` | 0–100 | **Low = periodic/looping motion detected** |
| `blink_detected` | boolean | |
| `challenge_passed` | boolean | |
| `depth_passed` | boolean | |
| `challenge_sequence` | ChallengeStep[] | `{challenge, passed, score, peak_frame_index}` per requested challenge. A single-challenge request still populates this with one entry, so consumers need only one code path. |
| `challenge_sequence_passed` | boolean | Every challenge passed **and** in the requested order. See §2 for what `mirrored` does to this. |
| `rppg_bpm` | number \| null | **Advisory only; never gates.** Often `null`. |
| `reasons` | Reason[] | `{code, message, severity}` |

Stage weights: `0.10·position + 0.10·lighting + 0.25·blink + 0.25·challenge + 0.20·depth + 0.10·motion`, then scaled by a replay factor in `[0.55, 1.0]`.

### 5.4 `VerificationResponse`

| Field | Type | Notes |
|---|---|---|
| `status` | enum | `verified` · `verified_high_confidence` · `verified_medium_confidence` · `manual_review` · `rejected` |
| `final_score` | 0–100 | |
| `confidence` | enum | `high` · `medium` · `low` |
| `capture_score`, `identity_score`, `liveness_score` | 0–100 | Layer headlines |
| `reasons` | Reason[] | Concatenated from all three layers; `{code, message, severity}` |
| `warnings` | string[] | Actionable risk flags. Plain strings — these are editorial summaries, not coded conditions. |
| `fraud_indicators` | object | `{virtual_camera, replay_attack, multiple_faces}`. Only measured checks appear — see §1. |
| `capture`, `identity`, `liveness` | object \| null | The full sub-results above |
| `quality` | object \| null | `{capture_quality_index, confidence_penalty_pct, issues[]}`; `issues` are `Reason` objects |
| `notes` | string[] | Quality-compensation explanations |
| `processing` | object \| null | `{total_ms, capture_ms, identity_ms, liveness_ms, decision_ms}`. Diagnostics; never affects the decision. |
| `models` | object \| null | The models that actually ran. No depth or spoof model exists — depth is geometric, from landmark homography residual. |
| `pipeline_version` | string | Bumped when scoring changes. **Persist it with the decision** so an audit can tell which rules produced it. |

### 5.5 `DecisionResponse`

`{status, final_score, confidence, capture_score, identity_score, liveness_score}`.

---

## 6. Errors

Failures return a FastAPI error body, **not** the OCR envelope:

```json
{"detail": "Too many frames: 91 (max 90)."}
```

`422` returns `detail` as an **array** of pydantic validation errors instead of a string.

| Status | When | Example `detail` |
|---|---|---|
| `400` | No frames supplied | `Provide at least one base64 frame in 'frames'.` |
| `400` | Multipart with neither part | `Provide either a 'video' file or one or more 'frames' image files.` |
| `400` | Over the frame cap | `Too many frames: 91 (max 90).` |
| `400` | Not valid base64 | `Invalid frame[0]: not valid base64 (...)` |
| `400` | Decodes, but isn't an image | `Invalid frame[0]: Could not decode image bytes (unsupported format?)` |
| `400` | Undecodable video | `Invalid video: No frames decoded from video` |
| `413` | Body over the byte cap | `Decoded frame payload exceeds 25165824 bytes.` |
| `422` | Bad enum / out-of-range field | `Input should be 'turn_left', 'turn_right', 'look_up' or 'look_down'` |
| `500` | Unhandled server error | |
| `503` | `/healthz` when the service is not ready | |

**A low score is not an error.** `rejected` comes back as `200` with a populated body. Only malformed input and server faults are non-2xx.

### Limits

| Limit | Default | Env override |
|---|---|---|
| Frames per request | 90 | `LIVENESS_MAX_JSON_FRAMES` |
| Decoded payload bytes | 25165824 (24 MiB) | `LIVENESS_MAX_JSON_BYTES` |
| Request timeout | 120 s | `GUNICORN_TIMEOUT` |

The byte cap applies to the **decoded** size, checked incrementally as frames are decoded — the request is rejected as soon as the running total crosses the limit, before the worker can OOM. Base64 inflates by ~33%, so budget the wire size accordingly.

---

## 7. Integration recipes

### 7.1 Node — full verification from a LiveKit track

```js
const FPS = 15;
const DURATION_MS = 5000;

const frames = [];
const timestamps = [];
const t0 = Date.now();

// Sample the decoded video track. Push the REAL clock time per frame —
// a synthetic evenly-spaced sequence scores as injected.
for await (const frame of sampleTrack(track, { fps: FPS, durationMs: DURATION_MS })) {
  frames.push(frame.toJPEG().toString('base64'));
  timestamps.push(Date.now() - t0);
  if (frames.length >= 90) break;             // server cap
}

const res = await fetch(`${OCR_API}/api/v1/liveness/verify-json`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    reference: idPhotoBase64,
    frames,
    // In the order you prompted them, captured in ONE continuous take.
    challenges: ['turn_left', 'turn_right', 'blink'],
    // Whether THESE FRAMES are flipped — not whether the preview looked flipped.
    // Without it, left-vs-right ordering cannot be verified. See §2.
    mirrored: framesWereCapturedFromAMirroredElement,
    fps: FPS,
    frame_timestamps_ms: timestamps,
    metadata: { device_name: clientDeviceName, width: 1280, height: 720, fps: FPS },
  }),
});

if (!res.ok) {
  const { detail } = await res.json();
  throw new Error(`Liveness call failed (${res.status}): ${JSON.stringify(detail)}`);
}

const r = await res.json();
switch (r.status) {
  case 'verified':
  case 'verified_high_confidence':
    return approve(r);
  case 'verified_medium_confidence':
  case 'manual_review':
    return queueForReview(r);                  // r.reasons / r.warnings explain why
  default:
    return reject(r);
}
```

### 7.2 Python

```python
import base64, requests

b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()

r = requests.post(
    "http://127.0.0.1:8000/api/v1/liveness/verify-json",
    json={
        "reference": b64("aadhaar.jpg"),
        "frames": [b64(f"frames/f{i:02d}.jpg") for i in range(20)],
        "challenges": ["turn_left", "turn_right", "blink"],
        "mirrored": False,                     # were THESE frames flipped?
        "fps": 15.0,
        "frame_timestamps_ms": timestamps,     # real capture clock
        "metadata": {"device_name": "FaceTime HD Camera"},
    },
    timeout=120,
)
r.raise_for_status()
result = r.json()
print(result["status"], result["final_score"])

for step in result["liveness"]["challenge_sequence"]:
    print(step["challenge"], "ok" if step["passed"] else "FAILED", step["score"])

for reason in result["reasons"]:
    print(reason["severity"], reason["code"], reason["message"])
```

### 7.3 Prompting the user

Score quality depends heavily on capture. Drive the UI like this:

1. Show an **oval guide** and wait for the face to be centred (`position_score` rewards it).
2. Prompt each challenge **one at a time, in the order you will send them** — "turn your head left", then "turn your head right", then "blink". Hold each ~1 s and leave a beat between them so the peaks separate cleanly.
3. Keep recording **across all of them**. One continuous take, never one clip per challenge — the ordering check and Layer 1's timing analysis both operate over the whole stream.
4. Record **8–12 s at ≥10 fps** for a three-challenge sequence. Fewer than 4 frames is rejected outright; the depth check needs actual rotation between frames.
5. Feed `reasons[].message` and `quality.issues[].message` straight back to the user on a retry — they are already written as user-facing guidance ("Lighting is too dark; increase ambient light."). Branch your own logic on `.code`.

### 7.4 Handling a cold container

```js
for (let i = 0; i < 60; i++) {
  const h = await (await fetch(`${OCR_API}/healthz`)).json();
  if (h.liveness_ready) break;
  if (h.liveness_error) throw new Error(`Liveness provisioning failed: ${h.liveness_error}`);
  await sleep(5000);
}
```

---

## 8. Operational notes

### Model provisioning

Weights are fetched at runtime — never committed, never baked into the image:

| Artefact | Size | Purpose |
|---|---|---|
| `face_landmarker.task` | 3.6 MB | MediaPipe 478-pt landmarks, blendshapes, head-pose matrix |
| `blaze_face_short_range.tflite` | 228 KB | MediaPipe face detector |
| `det_10g.onnx` | 16 MB | InsightFace detection |
| `w600k_r50.onnx` | 166 MB | InsightFace ArcFace recognition |
| `2d106det.onnx` | 4.8 MB | 2D landmarks used by face alignment |

They land in `LIVENESS_MODELS_DIR` (`/cache/liveness` in Docker, backed by the `liveness_models` volume). Downloads are lazy, atomic (`.part` + rename), retried with back-off, and sha256-verified against a manifest.

**Size the volume for ~700 MB.** Steady state is 329 MB, but provisioning peaks at ~605 MB: the InsightFace pack ships as one 275 MB archive that coexists with its extracted contents until unpacking finishes. The archive is then deleted (on the cold path and again on every subsequent boot, so pre-existing caches get swept too). `1k3d68.onnx` (137 MB) and `genderage.onnx` remain on disk even though this service never loads them — `allowed_modules` gates *loading*, not *downloading*.

First cold start downloads ~290 MB and takes a few minutes on a typical link. It runs **in the background** so the container becomes healthy immediately; `liveness_ready` stays `false` until it completes.

### Memory

The InsightFace pack is loaded with `allowed_modules=["detection", "recognition", "landmark_2d_106"]`, skipping the 137 MB 3D-landmark and gender/age models that no code path here calls.

Measured A/B on peak RSS per gunicorn worker: **967 MB** OCR-only vs **1096 MB** with both subsystems hot — liveness costs **~130 MB** per worker. With `WEB_CONCURRENCY=2` that is ~2.2 GB against the 3 GB container cap. Drop to `WEB_CONCURRENCY=1` if you also enable Surya (`KYC_ENABLE_SURYA=true`), which adds ~1.5–2 GB per worker on its own.

### Concurrency

MediaPipe's `FaceLandmarker` and the InsightFace app are **not** safe for concurrent inference on one instance, so the actual model calls are serialised behind a per-process lock. Requests still overlap on I/O and decoding, but two simultaneous liveness requests to one worker will queue at the inference step. Scale with workers, not threads.

Typical latencies on 2 vCPU, models warm: `/identity` ~2.5 s, `/frames` (20 frames) ~0.2 s, `/verify` ~2 s. The first call after a worker starts additionally pays the ONNX session build.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LIVENESS_MODELS_DIR` | `/cache/liveness` | Where weights are cached. Local dev: `./liveness_models` |
| `LIVENESS_REQUIRED` | `false` | Fold `liveness_ready` into `/healthz`'s `ok` |
| `LIVENESS_MAX_JSON_FRAMES` | `90` | Frame-count cap on the JSON endpoints |
| `LIVENESS_MAX_JSON_BYTES` | `25165824` | Decoded-payload cap (413 above it) |
| `LIVENESS_INSIGHTFACE_MODEL` | `buffalo_l` | InsightFace pack name |
| `LIVENESS_DET_SIZE` | `640` | Detector input size |
| `LIVENESS_ORT_PROVIDERS` | `CPUExecutionProvider` | Comma-separated onnxruntime providers |

### Interactive testing

- **Swagger** — `http://127.0.0.1:8000/docs`, liveness endpoints under the `liveness` tag.
- **Streamlit harness** — `streamlit run liveness_app.py`. Records off the local webcam, drives every endpoint over HTTP, and shows the raw request/response bodies. Local dev tool; excluded from the Docker image.
