"""Streamlit test harness for the liveness / face-match endpoints.

Run the API first, then this UI against it::

    LIVENESS_MODELS_DIR=./liveness_models uvicorn api:app --port 8000
    streamlit run liveness_app.py

This deliberately drives the service **over HTTP** rather than importing the
services directly: the point is to exercise the exact contract ``recriauth``
will consume, including the multipart-vs-JSON transports and the error codes.
Every endpoint in ``liveness/router.py`` has a tab here, and the raw request and
response bodies are always shown so a frontend integrator can copy them.

Like ``app.py`` and ``mask_identity_app.py`` this is a local dev tool — it is
excluded from the Docker image via ``.dockerignore``.
"""
from __future__ import annotations

import base64
import json
import time

import cv2
import numpy as np
import requests
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8000"
# The product flow prompts turn-left, turn-right, blink — in that order — and
# nothing else. `ChallengeType` still defines look_up / look_down as pre-existing
# API surface, but no screen asks for them, so they are not offered here.
CHALLENGES = ["turn_left", "turn_right", "blink"]
DEFAULT_SEQUENCE = ["turn_left", "turn_right", "blink"]

st.set_page_config(page_title="Liveness & Face Match", page_icon="🎥", layout="wide")
st.markdown(
    "<style>.block-container{padding-top:2rem;padding-bottom:2rem;}</style>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Transport helpers
# --------------------------------------------------------------------------- #
def api_base() -> str:
    return st.session_state.get("api_url", DEFAULT_API).rstrip("/")


def call(method: str, path: str, **kw) -> tuple[int, object, float]:
    """Issue one API call; return ``(status, parsed_body, elapsed_seconds)``.

    Never raises — a connection error is surfaced as status 0 with the reason in
    the body, so the UI can report it the same way it reports a 4xx.
    """
    url = f"{api_base()}{path}"
    t0 = time.perf_counter()
    try:
        resp = requests.request(method, url, timeout=kw.pop("timeout", 180), **kw)
    except requests.RequestException as exc:
        return 0, {"detail": f"Request failed: {exc}"}, time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    try:
        return resp.status_code, resp.json(), elapsed
    except ValueError:
        return resp.status_code, {"detail": resp.text[:2000]}, elapsed


def show(status: int, body: object, elapsed: float) -> bool:
    """Render a status line for a response. Returns True when it was a 2xx."""
    ok = 200 <= status < 300
    line = f"HTTP {status} · {elapsed:.2f}s"
    (st.success if ok else st.error)(line if ok else f"{line} — {body}")
    return ok


def frames_multipart(frames: list[bytes]) -> list[tuple]:
    """Build the ``frames[]`` multipart file list."""
    return [("frames", (f"frame_{i}.jpg", fb, "image/jpeg")) for i, fb in enumerate(frames)]


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# --------------------------------------------------------------------------- #
# Capture widgets
# --------------------------------------------------------------------------- #
def _oval(img: np.ndarray) -> np.ndarray:
    """Draw a face-placement oval on a *preview copy* — never on stored frames."""
    disp = img.copy()
    h, w = disp.shape[:2]
    cv2.ellipse(disp, (w // 2, h // 2), (int(w * 0.30), int(h * 0.42)),
                0, 0, 360, (0, 200, 255), 2)
    return disp


def _even_sample_indices(total: int, keep: int) -> list[int]:
    """Evenly spaced, strictly DISTINCT indices into a ``total``-length sequence.

    The distinctness matters. A naive ``linspace(...).round()`` repeats indices
    when ``total`` is only slightly above ``keep``, which would send the same
    frame twice — and Layer 1 counts near-identical consecutive frames as a
    replay signature. The harness would then report `FRAME_DUPLICATES` and
    `replay_attack: true` for a perfectly genuine capture.
    """
    if total <= keep:
        return list(range(total))
    idx = np.linspace(0, total - 1, keep).round().astype(int)
    return sorted(set(int(i) for i in idx))


# The service caps a JSON burst at FRAME_COUNT_CAP frames AND 24 MiB of decoded
# payload (LIVENESS_MAX_JSON_FRAMES / LIVENESS_MAX_JSON_BYTES). Frame count
# alone is not enough: 160 raw 1280x720 webcam JPEGs are ~48 MB and get a 413.
#
# Held as a literal rather than imported from liveness.config: this harness is
# deliberately a pure HTTP client so it can be pointed at a deployed service
# without the package on the path. Keep it in step by hand.
FRAME_COUNT_CAP = 160
#
# Downscaling costs nothing that matters here. InsightFace detects at 640x640 and
# MediaPipe landmarks are computed on a normalised crop, so pixels beyond ~720 on
# the long side are discarded by the models anyway — they only inflate the wire.
FRAME_LONG_SIDE = 720
FRAME_JPEG_QUALITY = 82
# 80% of the server cap. The headroom absorbs base64 framing and any per-frame
# variance, so a capture near the limit fails here (where we can adapt) rather
# than at the server (where it is a hard 413).
FRAME_BYTE_BUDGET = int(25_165_824 * 0.80)


def _encode_frames(
    raw: list[np.ndarray], stamps: list[float]
) -> tuple[list[bytes], list[float], str]:
    """Downscale + JPEG-encode a capture to fit the service's payload limits.

    Returns ``(jpeg_frames, timestamps_ms, note)``. Degrades in the order that
    preserves the liveness signal longest:

      1. Downscale — the models discard the extra pixels regardless.
      2. Lower JPEG quality — costs sharpness, which only nudges `blur_score`.
      3. Drop frames, evenly — LAST, because dropping frames is what actually
         destroys the signal: a blink spans 100-300 ms and thinning the sequence
         is how you lose it.
    """
    def encode(images: list[np.ndarray], quality: int) -> list[bytes]:
        out: list[bytes] = []
        for f in images:
            h, w = f.shape[:2]
            scale = FRAME_LONG_SIDE / float(max(h, w))
            if scale < 1.0:
                f = cv2.resize(f, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                out.append(buf.tobytes())
        return out

    # Frame count first — it is a hard server limit, not a budget.
    if len(raw) > FRAME_COUNT_CAP:
        idx = _even_sample_indices(len(raw), FRAME_COUNT_CAP)
        raw = [raw[i] for i in idx]
        stamps = [stamps[i] for i in idx]

    notes: list[str] = []
    quality = FRAME_JPEG_QUALITY
    frames = encode(raw, quality)

    while sum(map(len, frames)) > FRAME_BYTE_BUDGET and quality > 55:
        quality -= 10
        frames = encode(raw, quality)
        notes.append(f"JPEG quality reduced to {quality}")

    while sum(map(len, frames)) > FRAME_BYTE_BUDGET and len(frames) > 20:
        keep = int(len(frames) * 0.85)
        idx = _even_sample_indices(len(frames), keep)
        frames = [frames[i] for i in idx]
        stamps = [stamps[i] for i in idx]
        raw = [raw[i] for i in idx]
        notes.append(f"dropped to {len(frames)} frames")

    total_mb = sum(map(len, frames)) / 1e6
    note = f"{len(frames)} frames, {total_mb:.1f} MB"
    if notes:
        note += " (" + "; ".join(dict.fromkeys(notes)) + ")"
    return frames, stamps, note


def record_webcam(
    key: str, seconds: float, device_index: int, countdown: int = 3
) -> tuple[list[bytes], list[float]]:
    """Record a short clip off the local webcam via OpenCV.

    Returns ``(jpeg_frames, timestamps_ms)``. The real capture timestamps matter:
    Layer 1 scores frame-timing jitter, and synthetic evenly-spaced timestamps
    read as *deterministic* — i.e. injected — so they must come from the clock,
    not from a range().

    Streamlit's ``st.camera_input`` can only take stills, which cannot show a
    blink or a head turn; liveness needs genuine temporal frames.
    """
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        cap.release()
        st.error(
            f"Could not open camera at index {device_index}. On macOS grant camera "
            "access to your terminal/IDE under System Settings → Privacy & Security "
            "→ Camera, or try another index (0 = built-in, 1/2 = external/virtual)."
        )
        return [], []

    preview, status = st.empty(), st.empty()
    for n in range(countdown, 0, -1):
        ok, frame = cap.read()
        if ok:
            preview.image(_oval(cv2.resize(frame, (420, 315))), channels="BGR",
                          caption=f"Get ready… {n} — centre your face in the oval")
        time.sleep(1.0)

    raw: list[np.ndarray] = []
    stamps: list[float] = []
    start = time.time()
    i = 0
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        raw.append(frame)
        stamps.append((now - start) * 1000.0)
        if i % 2 == 0:
            elapsed = now - start
            preview.image(_oval(cv2.resize(frame, (420, 315))), channels="BGR",
                          caption=f"🔴 {elapsed:0.1f}/{seconds:0.0f}s — blink, then turn your head on cue")
            status.progress(min(1.0, elapsed / seconds))
        i += 1
    cap.release()
    preview.empty()
    status.empty()

    if not raw:
        st.error("No frames captured. Check camera permissions / index.")
        return [], []

    out, out_ts, note = _encode_frames(raw, stamps)
    fps = len(out) / max(seconds, 1e-6)
    st.session_state[f"frames_{key}"] = out
    st.session_state[f"stamps_{key}"] = out_ts
    st.success(f"Captured {note} at ~{fps:.0f} fps.")
    return out, out_ts


# --------------------------------------------------------------------------- #
# Guided candidate flow — mirrors what recriauth's "Get Started" step will do
# --------------------------------------------------------------------------- #
# (challenge, seconds, on-screen instruction). `None` for the challenge means
# the phase is not scored as a challenge — the opening "look straight" segment
# exists so the service has clean frontal frames to pick an identity probe from
# (it selects the sharpest frame containing a face itself).
#
# Durations MUST mirror recriauth's capture hook (use-liveness-capture.ts).
# This harness is what the motion thresholds get calibrated against, so a
# sequence that is paced differently from the one candidates actually perform
# calibrates against a take nobody makes.
#
# 2.0 s per cue was too short in practice — that budget has to cover reading
# the prompt AND completing the movement, so a candidate who reacts at all
# slowly peaks after the cue closes or never moves far enough to clear the
# excursion threshold. 3.5 s, inside a FRAME_COUNT_CAP raised to 160.
#
# The 0.6 s "look straight ahead" beats between challenges are load-bearing,
# not padding: _challenge_stage takes its excursion baseline as the MEDIAN pose
# over the whole clip, so a candidate who holds each turn drags that median
# toward the turn and shrinks their own measured excursion. Returning to centre
# keeps the median at rest.
FLOW_PHASES: list[tuple[str | None, float, str]] = [
    (None,         3.0, "Look straight at the camera"),
    ("turn_left",  3.5, "Turn your head LEFT"),
    (None,         0.6, "Look straight ahead"),
    ("turn_right", 3.5, "Turn your head RIGHT"),
    (None,         0.6, "Look straight ahead"),
    ("blink",      3.5, "Blink slowly, twice"),
]
FLOW_SEQUENCE = [c for c, _, _ in FLOW_PHASES if c]
MAX_ATTEMPTS = 3


def record_guided_sequence(
    key: str, device_index: int, countdown: int = 3
) -> tuple[list[bytes], list[float], list[dict]]:
    """Record ONE continuous take while cueing each challenge in turn.

    This is the whole point of the harness: the service checks that the actions
    happened in the requested ORDER across a single recording. Capturing one clip
    per challenge and concatenating them would pass locally and fail in
    production, because the timing/entropy analysis sees the seams.

    Returns ``(jpeg_frames, timestamps_ms, phase_marks)`` where each phase mark
    records when a cue was shown, so the result view can line the detected peak
    up against the moment the candidate was actually asked.
    """
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        cap.release()
        st.error(
            f"Could not open camera at index {device_index}. On macOS grant camera "
            "access to your terminal/IDE under System Settings → Privacy & Security "
            "→ Camera, or try another index (0 = built-in, 1/2 = external/virtual)."
        )
        return [], [], []

    preview, cue, bar = st.empty(), st.empty(), st.empty()

    for n in range(countdown, 0, -1):
        ok, frame = cap.read()
        if ok:
            preview.image(_oval(cv2.resize(frame, (480, 360))), channels="BGR")
        cue.info(f"### Starting in {n}…\nCentre your face in the oval.")
        time.sleep(1.0)

    raw: list[np.ndarray] = []
    stamps: list[float] = []
    marks: list[dict] = []
    total = sum(sec for _, sec, _ in FLOW_PHASES)
    start = time.time()
    i = 0

    for challenge, seconds, instruction in FLOW_PHASES:
        phase_start = time.time()
        marks.append({
            "challenge": challenge,
            "instruction": instruction,
            "cue_at_ms": (phase_start - start) * 1000.0,
            "duration_ms": seconds * 1000.0,
            "first_frame_index": len(raw),
        })
        cue.warning(f"### {instruction}")
        while time.time() - phase_start < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            raw.append(frame)
            stamps.append((now - start) * 1000.0)
            if i % 2 == 0:
                preview.image(_oval(cv2.resize(frame, (480, 360))), channels="BGR")
                bar.progress(min(1.0, (now - start) / total))
            i += 1
        marks[-1]["last_frame_index"] = max(marks[-1]["first_frame_index"], len(raw) - 1)

    cap.release()
    preview.empty()
    cue.empty()
    bar.empty()

    if not raw:
        st.error("No frames captured. Check camera permissions / index.")
        return [], [], []

    captured = len(raw)
    out, out_ts, note = _encode_frames(raw, stamps)

    # Re-point the phase marks at the surviving frames, so the result view can
    # still say "this action was cued over frames 12-21". Timestamps survive
    # every downsampling step, so map through those rather than trying to track
    # indices through each stage.
    for m in marks:
        start_ms = m["cue_at_ms"]
        end_ms = start_ms + m["duration_ms"]
        window = [i for i, t in enumerate(out_ts) if start_ms <= t <= end_ms]
        m["first_frame_index"] = window[0] if window else 0
        m["last_frame_index"] = window[-1] if window else max(0, len(out_ts) - 1)

    st.session_state[f"frames_{key}"] = out
    st.session_state[f"stamps_{key}"] = out_ts
    st.session_state[f"marks_{key}"] = marks
    fps = len(out) / max(total, 1e-6)
    st.success(
        f"Captured {captured} frames over {total:.0f}s → sending {note} (~{fps:.0f} fps)."
    )
    return out, out_ts, marks


def frame_source(key: str) -> tuple[list[bytes], list[float]]:
    """Frame input: record from the webcam, or upload images / a video clip."""
    mode = st.radio("Frame source", ["Webcam", "Upload images", "Upload video"],
                    horizontal=True, key=f"src_{key}")

    if mode == "Webcam":
        c1, c2 = st.columns(2)
        secs = c1.slider("Seconds", 2.0, 10.0, 5.0, 0.5, key=f"sec_{key}")
        dev = c2.number_input("Camera index", 0, 8, 0, key=f"dev_{key}")
        if st.button("● Record", key=f"rec_{key}", use_container_width=True):
            record_webcam(key, secs, int(dev))
    elif mode == "Upload images":
        ups = st.file_uploader("Frame images (in capture order)", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True, key=f"up_{key}")
        if ups:
            st.session_state[f"frames_{key}"] = [u.getvalue() for u in ups]
            # No real clock available for uploads: leave timestamps empty so
            # Layer 1 reports its timing check as neutral rather than scoring
            # fabricated cadence.
            st.session_state[f"stamps_{key}"] = []
    else:
        vid = st.file_uploader("Video clip", type=["mp4", "mov", "webm", "avi"], key=f"vid_{key}")
        if vid:
            st.session_state[f"video_{key}"] = vid.getvalue()
            st.session_state.pop(f"frames_{key}", None)
            st.info(f"Video held ({len(vid.getvalue())/1e6:.1f} MB) — sent as the `video` part.")

    frames = st.session_state.get(f"frames_{key}", [])
    stamps = st.session_state.get(f"stamps_{key}", [])
    if frames:
        st.caption(f"{len(frames)} frames held" + (f", {len(stamps)} timestamps" if stamps else ", no timestamps"))
        cols = st.columns(8)
        step = max(1, len(frames) // 8)
        for c, fb in zip(cols, frames[::step][:8]):
            c.image(fb, use_container_width=True)
    return frames, stamps


def image_input(label: str, key: str) -> bytes | None:
    """A single still: upload a file or grab one from the browser camera."""
    mode = st.radio(label, ["Upload", "Camera"], horizontal=True, key=f"m_{key}")
    if mode == "Upload":
        up = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=f"f_{key}")
        if up:
            st.session_state[f"img_{key}"] = up.getvalue()
    else:
        shot = st.camera_input(label, key=f"c_{key}")
        if shot:
            st.session_state[f"img_{key}"] = shot.getvalue()
    data = st.session_state.get(f"img_{key}")
    if data:
        st.image(data, width=220)
    return data


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def bars(body: dict, fields: list[tuple[str, str]]) -> None:
    """Render 0-100 sub-scores as labelled progress bars."""
    for key, label in fields:
        v = body.get(key)
        if isinstance(v, (int, float)):
            st.progress(min(1.0, max(0.0, v / 100.0)), text=f"{label} — {v:.1f}")


def flags(body: dict, fields: list[tuple[str, str]]) -> None:
    cols = st.columns(len(fields))
    for c, (key, label) in zip(cols, fields):
        v = body.get(key)
        c.metric(label, "✅ yes" if v else "—" if v is None else "❌ no")


SEVERITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def reasons(body: dict) -> None:
    """Render reasons, warnings and notes.

    ``reasons`` (and ``quality.issues``) are ``{code, message, severity}``
    objects; ``warnings`` and ``notes`` are still plain strings. Both shapes are
    handled so this panel keeps working either way — an integrator reading this
    harness should see the code, since that is what they will branch on.
    """
    for key, icon in (("reasons", "•"), ("warnings", "⚠️"), ("notes", "ℹ️")):
        items = body.get(key) or []
        if not items:
            continue
        st.markdown(f"**{key.title()}**")
        for item in items:
            if isinstance(item, dict):
                sev = SEVERITY_ICON.get(item.get("severity", ""), "•")
                st.markdown(f"{sev} `{item.get('code','')}` — {item.get('message','')}")
            else:
                st.markdown(f"{icon} {item}")

    quality = body.get("quality") or {}
    issues = quality.get("issues") or []
    if issues:
        st.markdown("**Capture quality issues**")
        for item in issues:
            if isinstance(item, dict):
                sev = SEVERITY_ICON.get(item.get("severity", ""), "•")
                st.markdown(f"{sev} `{item.get('code','')}` — {item.get('message','')}")
            else:
                st.markdown(f"• {item}")


def challenge_steps(body: dict) -> None:
    """Per-step table for an ordered challenge sequence."""
    steps = (body.get("liveness") or body).get("challenge_sequence") or []
    if not steps:
        return
    passed = (body.get("liveness") or body).get("challenge_sequence_passed")
    st.markdown(
        f"**Challenge sequence** — {'✅ in order' if passed else '❌ not verified in order'}"
    )
    st.dataframe(
        [
            {
                "challenge": s["challenge"],
                "passed": "✅" if s["passed"] else "❌",
                "score": s["score"],
                "peak frame": s["peak_frame_index"],
            }
            for s in steps
        ],
        hide_index=True,
        use_container_width=True,
    )


def raw(body: object) -> None:
    with st.expander("Raw JSON response"):
        st.json(body)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎥 Liveness tester")
    st.text_input("API base URL", DEFAULT_API, key="api_url")
    if st.button("Check /healthz", use_container_width=True):
        s, b, e = call("GET", "/healthz")
        show(s, b, e)
        st.json(b)
    if st.button("Check liveness models", use_container_width=True):
        s, b, e = call("GET", "/api/v1/liveness/health")
        show(s, b, e)
        st.json(b)
    st.divider()
    st.caption(
        "On a cold cache the first liveness call downloads ~330 MB of weights "
        "and will be slow. `/healthz` reports `liveness_ready` once the "
        "background provisioning finishes."
    )

st.title("Liveness & Face-Match — endpoint tester")

tabs = st.tabs([
    "🎬 Candidate flow", "Full verify", "Identity (L2)", "Liveness (L3)",
    "Capture (L1)", "Score fusion",
])

# --------------------------------------------------------------------------- #
# Candidate flow — a faithful rehearsal of recriauth's "Get Started" step
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Get Started — identity gate rehearsal")
    st.caption(
        "Runs the exact flow recriauth will run before digital address "
        "verification: look straight, turn left, turn right, blink — in ONE "
        "continuous recording — matched against the candidate's Aadhaar. "
        "Retries up to 3 times, then proceeds flagged. Build the frontend "
        "against what this tab sends."
    )

    ATT = "flow_attempts"
    RESULTS = "flow_results"
    st.session_state.setdefault(ATT, 0)
    st.session_state.setdefault(RESULTS, [])

    setup, stage = st.columns([1, 1.4], gap="large")

    with setup:
        st.markdown("**1 · Reference — Aadhaar front**")
        st.caption(
            "recriauth pulls this from S3 (the already-uploaded Aadhaar front) — "
            "the candidate never uploads it again here."
        )
        flow_ref = image_input("Aadhaar front / ID portrait", "flowref")

        st.markdown("**2 · Capture**")
        flow_dev = st.number_input("Camera index", 0, 8, 0, key="dev_flow")
        mirror_flow = st.radio(
            "Frames mirrored?", ["not mirrored", "mirrored", "not declared"],
            horizontal=True, key="mir_flow",
            help="OpenCV hands back RAW camera frames, so this harness is "
                 "'not mirrored'. A browser grabbing frames from a scaleX(-1) "
                 "element would be 'mirrored'. Pick 'not declared' to see the "
                 "CHALLENGE_SEQUENCE_ORDER_PARTIAL degradation.",
        )
        flow_mirrored = {"mirrored": True, "not mirrored": False}.get(mirror_flow)

        st.markdown("**Sequence**")
        for c, sec, instruction in FLOW_PHASES:
            tag = f"`{c}`" if c else "_probe only, not scored_"
            st.markdown(f"- {instruction} — {sec:.0f}s · {tag}")

        attempts = st.session_state[ATT]
        st.divider()
        st.metric("Attempt", f"{min(attempts + 1, MAX_ATTEMPTS)} of {MAX_ATTEMPTS}")

        disabled = not flow_ref or attempts >= MAX_ATTEMPTS
        if st.button("● Start capture", type="primary",
                     use_container_width=True, disabled=disabled):
            st.session_state["flow_go"] = True
        if not flow_ref:
            st.warning("Provide the Aadhaar front first.")
        if attempts >= MAX_ATTEMPTS:
            st.error(f"All {MAX_ATTEMPTS} attempts used.")
        if attempts and st.button("Reset candidate", use_container_width=True):
            st.session_state[ATT] = 0
            st.session_state[RESULTS] = []
            st.session_state.pop("flow_go", None)
            st.rerun()

    with stage:
        if st.session_state.pop("flow_go", False):
            frames, stamps, marks = record_guided_sequence("flow", int(flow_dev))
            if frames:
                payload = {
                    "reference": b64(flow_ref),
                    "frames": [b64(f) for f in frames],
                    "challenges": FLOW_SEQUENCE,
                    "mirrored": flow_mirrored,
                    "fps": round(len(frames) / sum(s for _, s, _ in FLOW_PHASES), 1),
                    "frame_timestamps_ms": [round(t, 1) for t in stamps],
                    "metadata": {
                        "device_name": f"OpenCV camera {int(flow_dev)}",
                        "fps": round(len(frames) / sum(s for _, s, _ in FLOW_PHASES), 1),
                    },
                }
                s, b, e = call("POST", "/api/v1/liveness/verify-json", json=payload)
                st.session_state[ATT] += 1
                if isinstance(b, dict) and s == 200:
                    st.session_state[RESULTS].append(
                        {"status_code": s, "body": b, "elapsed": e, "marks": marks,
                         "payload_keys": {k: (len(v) if isinstance(v, list) else v)
                                          for k, v in payload.items() if k != "reference"}})
                else:
                    st.session_state[RESULTS].append(
                        {"status_code": s, "body": b, "elapsed": e, "marks": marks})
                st.rerun()

        results = st.session_state[RESULTS]
        if not results:
            st.info(
                "Load the Aadhaar front, then press **Start capture**. Follow each "
                "on-screen cue as it appears — the service checks the actions "
                "happened in that order."
            )
        else:
            latest = results[-1]
            body, code = latest["body"], latest["status_code"]

            if code != 200 or not isinstance(body, dict):
                st.error(f"Call failed (HTTP {code}).")
                st.json(body)
            else:
                passed = body["status"] in (
                    "verified", "verified_high_confidence", "verified_medium_confidence")
                attempts = st.session_state[ATT]

                # ---- What the CANDIDATE sees -------------------------------
                st.markdown("#### Candidate sees")
                if passed:
                    st.success("### ✅ You're all set\nContinue to address verification.")
                elif attempts < MAX_ATTEMPTS:
                    st.warning(
                        f"### Let's try that again\nAttempt {attempts} of {MAX_ATTEMPTS}.")
                else:
                    st.error(
                        "### We couldn't verify you automatically\n"
                        "You can continue — a reviewer will check this manually.")

                # Only the prose, never a score. Exactly what the candidate UI
                # should render on a retry.
                guidance = [r["message"] for r in body.get("reasons", [])
                            if r.get("severity") in ("medium", "high")]
                guidance += [i["message"] for i in
                             (body.get("quality") or {}).get("issues", [])]
                if guidance and not passed:
                    st.markdown("**What to fix**")
                    for g in dict.fromkeys(guidance):
                        st.markdown(f"- {g}")

                if not passed and attempts >= MAX_ATTEMPTS:
                    st.info("→ Proceeds to address verification, **flagged for the reviewer**.")

                st.divider()

                # ---- What OPS / the reviewer sees ---------------------------
                st.markdown("#### Reviewer sees")
                c1, c2, c3 = st.columns(3)
                c1.metric("Decision", body["status"])
                c2.metric("Final score", f"{body['final_score']:.1f}")
                c3.metric("Confidence", body["confidence"])
                bars(body, [("capture_score", "Capture · 0.20"),
                            ("identity_score", "Identity · 0.35"),
                            ("liveness_score", "Liveness · 0.45")])

                liv = body.get("liveness") or {}
                steps = liv.get("challenge_sequence") or []
                if steps:
                    marks = {m["challenge"]: m for m in latest.get("marks", []) if m["challenge"]}
                    st.markdown(
                        "**Challenge sequence** — "
                        + ("✅ completed in order" if liv.get("challenge_sequence_passed")
                           else "❌ not verified in order")
                    )
                    st.dataframe(
                        [{
                            "challenge": s["challenge"],
                            "passed": "✅" if s["passed"] else "❌",
                            "score": s["score"],
                            "peak frame": s["peak_frame_index"],
                            "cued frames": (
                                f"{marks[s['challenge']]['first_frame_index']}–"
                                f"{marks[s['challenge']]['last_frame_index']}"
                                if s["challenge"] in marks else "—"),
                        } for s in steps],
                        hide_index=True, use_container_width=True,
                    )
                    st.caption(
                        "“peak frame” inside “cued frames” means the action landed "
                        "while it was being asked for. Well outside it suggests the "
                        "candidate anticipated or lagged the cue."
                    )

                if body.get("fraud_indicators"):
                    st.markdown("**Fraud indicators**")
                    flags(body["fraud_indicators"], [
                        ("virtual_camera", "Virtual camera"),
                        ("replay_attack", "Replay"),
                        ("multiple_faces", "Multiple faces")])

                idn = body.get("identity") or {}
                if idn:
                    st.caption(
                        f"Face match: similarity {idn.get('similarity')} vs threshold "
                        f"{idn.get('threshold')} → **{'match' if idn.get('match') else 'no match'}**"
                    )

                reasons(body)

                with st.expander("Request sent (build the frontend against this)"):
                    st.json(latest.get("payload_keys", {}))
                    st.caption(
                        "`reference` and `frames` omitted — base64 blobs. Note "
                        "`challenges` is ordered, `mirrored` is declared, and "
                        "`frame_timestamps_ms` came from a real clock."
                    )
                with st.expander("Full response"):
                    st.json(body)

            if len(results) > 1:
                st.divider()
                st.markdown("**Attempt history**")
                st.dataframe(
                    [{
                        "attempt": i + 1,
                        "http": r["status_code"],
                        "status": (r["body"] or {}).get("status") if isinstance(r["body"], dict) else "—",
                        "final": (r["body"] or {}).get("final_score") if isinstance(r["body"], dict) else "—",
                        "seconds": round(r["elapsed"], 1),
                    } for i, r in enumerate(results)],
                    hide_index=True, use_container_width=True,
                )

# --------------------------------------------------------------------------- #
# Full verify — both transports
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("POST /api/v1/liveness/verify · /verify-json")
    st.caption(
        "Runs all three layers and returns the fused decision. The JSON transport "
        "sends the same data base64-encoded — it is what a server-side caller "
        "(recriauth, pulling frames off a LiveKit track) will use."
    )
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown("**Reference ID photo**")
        ref = image_input("Reference (ID card or portrait)", "vref")
        st.markdown("**Live capture**")
        frames, stamps = frame_source("verify")
        sequence = st.multiselect(
            "Challenge sequence (order matters)", CHALLENGES, default=DEFAULT_SEQUENCE,
            key="ch_v",
            help="Performed in ONE continuous take. The service checks they happened "
                 "in this order — three separate clips would defeat the point.",
        )
        mirror_choice = st.radio(
            "Camera mirroring", ["not declared", "mirrored", "not mirrored"],
            horizontal=True, key="mir_v",
            help="A mirrored left→right is an identical signal to an unmirrored "
                 "right→left. Leave undeclared and left/right ordering cannot be "
                 "verified — the response says so via CHALLENGE_SEQUENCE_ORDER_PARTIAL.",
        )
        fps = st.number_input("Declared fps", 1.0, 60.0, 15.0, key="fps_v")
        device = st.text_input("Camera device_name (Layer 1 metadata)", "FaceTime HD Camera")
        transport = st.radio("Transport", ["multipart", "JSON (base64)"], horizontal=True)
        go = st.button("Run full verification", type="primary", use_container_width=True)

    mirrored = {"mirrored": True, "not mirrored": False}.get(mirror_choice)

    with right:
        if go:
            video = st.session_state.get("video_verify")
            if not ref:
                st.warning("A reference photo is required.")
            elif not frames and not video:
                st.warning("Record or upload a live capture first.")
            else:
                meta = {"device_name": device, "fps": fps}
                if transport.startswith("JSON"):
                    if not frames:
                        st.warning("The JSON transport needs image frames, not a video clip.")
                        st.stop()
                    payload = {
                        "reference": b64(ref),
                        "frames": [b64(f) for f in frames],
                        "challenges": sequence,
                        "mirrored": mirrored,
                        "fps": fps,
                        "frame_timestamps_ms": stamps,
                        "metadata": meta,
                    }
                    size = len(json.dumps(payload)) / 1e6
                    st.caption(f"JSON body ≈ {size:.1f} MB")
                    s, b, e = call("POST", "/api/v1/liveness/verify-json", json=payload)
                else:
                    files = [("reference", ("reference.jpg", ref, "image/jpeg"))]
                    files += ([("video", ("clip.mp4", video, "video/mp4"))] if video
                              else frames_multipart(frames))
                    data = {"payload": json.dumps({
                        "challenges": sequence, "mirrored": mirrored, "fps": fps,
                        "frame_timestamps_ms": stamps, "metadata": meta})}
                    s, b, e = call("POST", "/api/v1/liveness/verify", files=files, data=data)

                if show(s, b, e) and isinstance(b, dict):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Decision", b["status"])
                    c2.metric("Final score", f"{b['final_score']:.1f}")
                    c3.metric("Confidence", b["confidence"])
                    bars(b, [("capture_score", "Layer 1 · capture"),
                             ("identity_score", "Layer 2 · identity"),
                             ("liveness_score", "Layer 3 · liveness")])
                    if b.get("liveness"):
                        st.markdown("**Liveness breakdown**")
                        bars(b["liveness"], [
                            ("position_score", "Position"), ("lighting_score", "Lighting"),
                            ("blink_score", "Blink"), ("challenge_score", "Challenge"),
                            ("depth_score", "Depth"), ("motion_score", "Micro-movement"),
                            ("replay_resistance_score", "Replay resistance")])
                        flags(b["liveness"], [("blink_detected", "Blink"),
                                              ("challenge_passed", "Challenge"),
                                              ("depth_passed", "Depth")])
                        challenge_steps(b)
                    if b.get("fraud_indicators"):
                        st.markdown("**Fraud indicators**")
                        flags(b["fraud_indicators"], [
                            ("virtual_camera", "Virtual camera"),
                            ("replay_attack", "Replay"),
                            ("multiple_faces", "Multiple faces")])
                        st.caption(
                            "Only checks with a real signal behind them are listed. "
                            "Deepfake / face-swap / document-tampering are absent because "
                            "no model here evaluates them — absent, not `false`.")
                    if b.get("quality"):
                        q = b["quality"]
                        st.caption(
                            f"Capture quality index {q['capture_quality_index']:.1f} · "
                            f"confidence penalty {q['confidence_penalty_pct']:.1f}%")
                    reasons(b)
                    if b.get("processing") or b.get("models"):
                        with st.expander("Diagnostics (models + timings)"):
                            st.write({"pipeline_version": b.get("pipeline_version"),
                                      "models": b.get("models"),
                                      "processing": b.get("processing")})
                    raw(b)

# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("POST /api/v1/liveness/identity")
    st.caption(
        "ArcFace match between a reference ID photo and a probe. Full-card uploads "
        "are handled — the portrait is located and cropped first, across rotations."
    )
    c1, c2, c3 = st.columns([1, 1, 1.4], gap="large")
    with c1:
        ref = image_input("Reference (ID)", "iref")
    with c2:
        probe = image_input("Probe (live face)", "iprobe")
    with c3:
        if st.button("Compare faces", type="primary", use_container_width=True):
            if not (ref and probe):
                st.warning("Both a reference and a probe image are required.")
            else:
                s, b, e = call("POST", "/api/v1/liveness/identity", files=[
                    ("reference", ("reference.jpg", ref, "image/jpeg")),
                    ("probe", ("probe.jpg", probe, "image/jpeg"))])
                if show(s, b, e) and isinstance(b, dict):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Match", "✅ yes" if b["match"] else "❌ no")
                    m2.metric("Similarity", f"{b['similarity']:.3f}")
                    m3.metric("Threshold", f"{b['threshold']:.3f}")
                    bars(b, [("identity_score", "Identity score"),
                             ("landmark_score", "Landmark geometry"),
                             ("quality_score", "Image quality")])
                    reasons(b)
                    raw(b)

# --------------------------------------------------------------------------- #
# Liveness
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("POST /api/v1/liveness/check · /frames")
    st.caption(
        "Layer 3 only. Blink and head-turn need real motion, so record rather than "
        "upload a still — and actually perform the challenge on cue."
    )
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        frames, stamps = frame_source("live")
        sequence_l = st.multiselect(
            "Challenge sequence (order matters)", CHALLENGES,
            default=DEFAULT_SEQUENCE, key="ch_l",
            help="One continuous take — the ordering check spans the whole recording.",
        )
        mirror_l = st.radio("Camera mirroring", ["not declared", "mirrored", "not mirrored"],
                            horizontal=True, key="mir_l")
        mirrored_l = {"mirrored": True, "not mirrored": False}.get(mirror_l)
        fps = st.number_input("Declared fps", 1.0, 60.0, 15.0, key="fps_l")
        transport = st.radio("Transport", ["multipart (/check)", "JSON (/frames)"],
                             horizontal=True, key="tr_l")
        go = st.button("Run liveness check", type="primary", use_container_width=True)
    with right:
        if go:
            video = st.session_state.get("video_live")
            if not frames and not video:
                st.warning("Record or upload a capture first.")
            elif transport.startswith("JSON"):
                if not frames:
                    st.warning("The JSON transport needs image frames, not a video clip.")
                else:
                    s, b, e = call("POST", "/api/v1/liveness/frames", json={
                        "frames": [b64(f) for f in frames], "challenges": sequence_l,
                        "mirrored": mirrored_l, "fps": fps,
                        "frame_timestamps_ms": stamps})
                    if show(s, b, e) and isinstance(b, dict):
                        st.metric("Liveness score", f"{b['liveness_score']:.1f}")
                        bars(b, [("position_score", "Position"), ("lighting_score", "Lighting"),
                                 ("blink_score", "Blink"), ("challenge_score", "Challenge"),
                                 ("depth_score", "Depth"), ("motion_score", "Micro-movement"),
                                 ("replay_resistance_score", "Replay resistance")])
                        flags(b, [("blink_detected", "Blink"), ("challenge_passed", "Challenge"),
                                  ("depth_passed", "Depth"), ("rppg_bpm", "rPPG bpm")])
                        challenge_steps(b)
                        reasons(b)
                        raw(b)
            else:
                files = ([("video", ("clip.mp4", video, "video/mp4"))] if video
                         else frames_multipart(frames))
                s, b, e = call("POST", "/api/v1/liveness/check", files=files,
                               data={"payload": json.dumps({"challenges": sequence_l,
                                                            "mirrored": mirrored_l,
                                                            "fps": fps})})
                if show(s, b, e) and isinstance(b, dict):
                    st.metric("Liveness score", f"{b['liveness_score']:.1f}")
                    bars(b, [("position_score", "Position"), ("lighting_score", "Lighting"),
                             ("blink_score", "Blink"), ("challenge_score", "Challenge"),
                             ("depth_score", "Depth"), ("motion_score", "Micro-movement"),
                             ("replay_resistance_score", "Replay resistance")])
                    flags(b, [("blink_detected", "Blink"), ("challenge_passed", "Challenge"),
                              ("depth_passed", "Depth"), ("rppg_bpm", "rPPG bpm")])
                    challenge_steps(b)
                    reasons(b)
                    raw(b)

# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("POST /api/v1/liveness/capture-check")
    st.caption(
        "Layer 1 only — is this stream from a real camera? Scores frame-timing "
        "jitter, camera metadata and temporal entropy. Real capture timestamps "
        "matter here: without them the timing check reports neutral."
    )
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        frames, stamps = frame_source("cap")
        dev_name = st.text_input("device_name", "FaceTime HD Camera", key="cdn")
        driver = st.text_input("driver", "", key="cdr")
        st.caption("Try `OBS Virtual Camera` to see the virtual-camera detector fire.")
        go = st.button("Run capture check", type="primary", use_container_width=True)
    with right:
        if go:
            video = st.session_state.get("video_cap")
            if not frames and not video:
                st.warning("Record or upload a capture first.")
            else:
                files = ([("video", ("clip.mp4", video, "video/mp4"))] if video
                         else frames_multipart(frames))
                payload = {"frame_timestamps_ms": stamps,
                           "metadata": {"device_name": dev_name, "driver": driver or None}}
                s, b, e = call("POST", "/api/v1/liveness/capture-check", files=files,
                               data={"payload": json.dumps(payload)})
                if show(s, b, e) and isinstance(b, dict):
                    m1, m2 = st.columns(2)
                    m1.metric("Capture score", f"{b['capture_score']:.1f}")
                    m2.metric("Median fps", f"{b['median_fps']:.2f}")
                    bars(b, [("timing_score", "Frame timing"),
                             ("metadata_score", "Camera metadata"),
                             ("entropy_score", "Temporal entropy"),
                             ("injection_score", "Injection RISK (higher = worse)")])
                    reasons(b)
                    raw(b)

# --------------------------------------------------------------------------- #
# Score fusion
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("POST /api/v1/liveness/score")
    st.caption(
        "Fuse three already-computed layer scores into the final decision without "
        "re-running the pipeline — weights 0.20 capture / 0.35 identity / 0.45 liveness."
    )
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        cap_s = st.slider("capture_score", 0.0, 100.0, 85.0)
        idn_s = st.slider("identity_score", 0.0, 100.0, 72.0)
        liv_s = st.slider("liveness_score", 0.0, 100.0, 88.0)
        qidx = st.slider("quality_index", 0.0, 100.0, 80.0)
        match = st.checkbox("identity_match", True)
        go = st.button("Fuse scores", type="primary", use_container_width=True)
    with c2:
        if go:
            s, b, e = call("POST", "/api/v1/liveness/score", json={
                "capture_score": cap_s, "identity_score": idn_s, "liveness_score": liv_s,
                "identity_match": match, "quality_index": qidx})
            if show(s, b, e) and isinstance(b, dict):
                m1, m2, m3 = st.columns(3)
                m1.metric("Decision", b["status"])
                m2.metric("Final score", f"{b['final_score']:.1f}")
                m3.metric("Confidence", b["confidence"])
                raw(b)
