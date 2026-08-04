"""Layer 3 - Active Liveness Verification.

Proves a live human is physically present. Operates on a short sequence of
frames and fuses seven stages:

1. Face position (oval alignment)           - 10%
2. Lighting quality                          - 10%
3. Blink detection (involuntary/prompted)    - 25%
4. Head-turn challenge (motion parallax)     - 25%
5. Depth verification (non-planarity)        - 20%
6. Micro-movement analysis                   - 10%
7. Replay detection (periodicity)            - penalty on the fused score
   rPPG                                       - advisory only, never gates

The Quality Compensation System ensures poor capture conditions reduce
*confidence* with an explanation rather than causing a hard failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from liveness.config import settings
from liveness.schemas.requests import ChallengeType
from liveness.schemas.responses import ChallengeStepResult, LivenessResult, Reason
from liveness.services import reason_codes as rc
from liveness.utils.face_engine import LandmarkResult, detect_landmarks
from liveness.utils.image import sample_frames
from liveness.utils.logger import logger
from liveness.utils.metrics import (
    clamp,
    dominant_frequency,
    linear_map,
    periodicity_score,
)
from liveness.utils.quality import assess_quality

# --- MediaPipe FaceLandmarker (478-pt) indices ---
NOSE_TIP = 1
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
LEFT_CHEEK, RIGHT_CHEEK = 234, 454   # face silhouette sides
CHIN, FOREHEAD = 152, 10
# Rigid, age-stable points used for the planar (homography) depth test.
RIGID_POINTS = [33, 133, 362, 263, 1, 61, 291, 199, 4, 6, 168, 152]
# Tracked points for micro-movement (eye corners, nose, mouth corners).
MICRO_POINTS = [33, 263, 1, 61, 291]

# Landmark detection is the expensive part of this layer, so the frame sequence
# is sub-sampled before analysis. The ceiling depends on how much has to fit in
# the take:
#
#   * One challenge is a ~4 s clip; 60 frames is ~15 fps, ample.
#   * A three-challenge sequence runs ~9-12 s. Holding 60 frames across that is
#     ~5 fps, and a blink lasts 100-300 ms — it would routinely fall BETWEEN
#     sampled frames and read as "no blink" for a candidate who blinked properly.
#
# So a sequence keeps the full server-side input cap. The extra landmark passes
# cost a few hundred ms; a false "no blink" costs a real candidate a retry.
#
# MAX_FRAMES_SEQUENCE is DERIVED from ``settings.max_json_frames`` rather than
# written down, because LIVENESS_MAX_JSON_FRAMES is env-overridable and a
# hard-coded twin drifts the first time someone raises it in a deployment. It
# is not an independent tuning knob — two things break the moment it sits below
# the transport cap:
#
#   1. Blink loss, the failure this ceiling exists to prevent. Thinning a
#      ~12 s take down to 90 frames is ~7.5 fps, and at 100-300 ms a blink
#      occupies 0.8-2.3 frames — right on the edge of vanishing between
#      samples. Raising the transport cap WITHOUT raising this one is worse
#      than not raising it at all: more frames arrive, then get thinned harder.
#   2. ``peak_frame_index`` desync. Peaks are reported as indices into the
#      sampled list. Callers compare them against cue windows expressed in the
#      indices they SENT, so any sub-sampling here silently shifts every peak
#      and makes "landed when asked" read wrong.
#
# Both are avoided by tracking the transport cap exactly, so sub-sampling can
# never fire on a sequence: nothing larger than it can arrive.
MAX_FRAMES = 60
MAX_FRAMES_SEQUENCE = settings.max_json_frames


def _interocular(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[RIGHT_EYE_OUTER, :2] - lm[LEFT_EYE_OUTER, :2])) + 1e-6


def _head_signals(lm: np.ndarray) -> tuple[float, float]:
    """Robust, scale-invariant yaw & pitch proxies from 2-D landmarks.

    These reflect *visible* head rotation directly and are far more reliable for
    detecting a head-turn than MediaPipe's (often damped) transformation matrix.

    * **yaw**  - based on how close the nose projects to each cheek silhouette;
      ``(d_right - d_left)/(d_right + d_left)`` in roughly ``[-1, 1]``. Sign
      flips with turn direction; ~0 when facing forward.
    * **pitch** - vertical nose offset between forehead and chin, normalised to
      ``[-1, 1]`` (negative looking up, positive looking down).
    """
    nose = lm[NOSE_TIP, :2]
    lc = lm[LEFT_CHEEK, :2]
    rc = lm[RIGHT_CHEEK, :2]
    d_left = float(np.linalg.norm(nose - lc))
    d_right = float(np.linalg.norm(nose - rc))
    yaw = (d_right - d_left) / (d_right + d_left + 1e-6)

    fore = lm[FOREHEAD, :2]
    chin = lm[CHIN, :2]
    span = float(np.linalg.norm(chin - fore)) + 1e-6
    # 0 at forehead, 1 at chin -> centre ~0.5; remap to [-1,1].
    pitch = (float(nose[1] - fore[1]) / span) * 2.0 - 1.0
    return yaw, pitch


# --------------------------------------------------------------------------- #
# Stage 1 - Face position
# --------------------------------------------------------------------------- #
def _position_stage(results: list[LandmarkResult]) -> tuple[float, list[Reason]]:
    reasons: list[Reason] = []
    scores = []
    for r in results:
        if not r.found:
            continue
        w, h = r.image_size
        lm = r.landmarks
        cx, cy = lm[:, 0].mean(), lm[:, 1].mean()
        # Centering: distance of face centre from frame centre.
        off = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        center_score = clamp(linear_map(off, 0.30, 0.0, 0.0, 100.0))
        # Size: face span vs frame height.
        span = (lm[:, 1].max() - lm[:, 1].min()) / h
        size_score = clamp(linear_map(span, 0.25, 0.75, 0.0, 100.0))
        # Pose: small roll/pitch preferred for the neutral position frames.
        roll = abs(r.pose["roll"])
        roll_score = clamp(linear_map(roll, 25.0, 0.0, 0.0, 100.0))
        scores.append(0.45 * center_score + 0.35 * size_score + 0.20 * roll_score)

    if not scores:
        reasons.append(rc.NO_FACE_FOR_POSITION("No face detected for position validation."))
        return 0.0, reasons
    score = float(np.mean(scores))
    if score < 60:
        reasons.append(rc.POSITION_POOR("Face not well centred/sized within the oval guide."))
    else:
        reasons.append(rc.POSITION_OK("Face correctly positioned within the oval guide."))
    return clamp(score), reasons


# --------------------------------------------------------------------------- #
# Stage 2 - Lighting
# --------------------------------------------------------------------------- #
def _lighting_stage(frames: list[np.ndarray], results: list[LandmarkResult]) -> tuple[float, list[Reason]]:
    reports = []
    for f, r in zip(frames, results):
        bbox = None
        if r.found:
            lm = r.landmarks
            bbox = (int(lm[:, 0].min()), int(lm[:, 1].min()),
                    int(lm[:, 0].max()), int(lm[:, 1].max()))
        reports.append(assess_quality(f, bbox))
    bright = float(np.mean([q.brightness_score for q in reports]))
    contrast = float(np.mean([q.contrast_score for q in reports]))
    score = clamp(0.6 * bright + 0.4 * contrast)
    reasons: list[Reason] = []
    if bright < 55:
        reasons.append(rc.LIGHTING_UNDEREXPOSED(
            "Lighting reduced confidence; scene is under-exposed."))
    elif bright > 90 and contrast > 70:
        reasons.append(rc.LIGHTING_OK("Lighting conditions are good."))
    return score, reasons


# --------------------------------------------------------------------------- #
# Stage 3 - Blink
# --------------------------------------------------------------------------- #
def _blink_series(results: list[LandmarkResult]) -> np.ndarray:
    """Per-frame "eyes are closed" signal over the face-found frames.

    Higher means more closed, so the series peaks at the blink — which is what
    both the blink stage and the challenge-ordering logic need. Prefers the
    MediaPipe blendshapes and falls back to an inverted eye-aspect-ratio, so the
    two callers can never disagree about *when* the blink happened.
    """
    left = np.array([r.blendshapes.get("eyeBlinkLeft", 0.0) for r in results if r.found])
    right = np.array([r.blendshapes.get("eyeBlinkRight", 0.0) for r in results if r.found])
    if left.size >= 3:
        return np.maximum(left, right)

    ear = _ear_series(results)
    if ear.size < 3:
        return np.asarray([])
    # EAR falls as the eye closes; invert so the blink is a peak like above.
    med = float(np.median(ear)) + 1e-6
    return np.clip(1.0 - (ear / med), 0.0, 1.0)


def _blink_stage(results: list[LandmarkResult]) -> tuple[float, bool, list[Reason]]:
    """Detect a blink as a transient spike in eye-blink blendshapes."""
    reasons: list[Reason] = []
    left = np.array([r.blendshapes.get("eyeBlinkLeft", 0.0) for r in results if r.found])
    if left.size < 3:
        # Fall back to eye-aspect-ratio if blendshapes unavailable.
        ear = _ear_series(results)
        if ear.size < 3:
            reasons.append(rc.BLINK_FRAMES_INSUFFICIENT(
                "Insufficient frames for blink detection."))
            return 0.0, False, reasons
        closed = ear < (np.median(ear) * 0.75)
        blinked = bool(np.any(closed) and np.any(~closed))
        score = 85.0 if blinked else 25.0
        reasons.append(rc.BLINK_DETECTED("Blink detected (EAR).") if blinked
                       else rc.NO_BLINK("No blink observed."))
        return score, blinked, reasons

    signal = _blink_series(results)
    peak = float(signal.max())
    baseline = float(np.percentile(signal, 20))
    transient = peak - baseline
    blinked = peak > 0.5 and transient > 0.3
    # Reward a clear, transient closure.
    score = clamp(linear_map(transient, 0.1, 0.6, 20.0, 100.0)) if blinked else \
        clamp(linear_map(peak, 0.0, 0.5, 0.0, 45.0))
    reasons.append(rc.BLINK_DETECTED("Natural blink detected.") if blinked
                   else rc.NO_BLINK("No clear blink detected."))
    return score, blinked, reasons


def _ear_series(results: list[LandmarkResult]) -> np.ndarray:
    vals = []
    for r in results:
        if not r.found:
            continue
        lm = r.landmarks
        # Vertical eye opening over horizontal width (left eye).
        horiz = np.linalg.norm(lm[LEFT_EYE_OUTER, :2] - lm[LEFT_EYE_INNER, :2]) + 1e-6
        vert = np.linalg.norm(lm[159, :2] - lm[145, :2]) if lm.shape[0] > 159 else 0.0
        vals.append(vert / horiz)
    return np.asarray(vals)


# --------------------------------------------------------------------------- #
# Stage 4 - Head-turn challenge (motion parallax)
# --------------------------------------------------------------------------- #
HORIZONTAL_CHALLENGES = (ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT)
VERTICAL_CHALLENGES = (ChallengeType.LOOK_UP, ChallengeType.LOOK_DOWN)

# Calibrated for the normalised yaw/pitch signals: ~0.05 is a weak nod,
# >=0.22 is an unmistakable turn. Carried over from the single-challenge
# implementation so scores stay comparable across the change.
_MOTION_WEAK, _MOTION_STRONG, _MOTION_PASS = 0.05, 0.22, 0.09
# Blink blendshape excursion above the resting baseline that counts as closed.
_BLINK_WEAK, _BLINK_STRONG, _BLINK_PASS = 0.15, 0.55, 0.30


@dataclass
class _Step:
    """One challenge located within the recording."""

    challenge: ChallengeType
    passed: bool
    score: float
    peak_index: int
    reasons: list[Reason] = field(default_factory=list)


def _challenge_signal(
    challenge: ChallengeType, yaw: np.ndarray, pitch: np.ndarray,
    blink: np.ndarray, mirrored: bool,
) -> np.ndarray | None:
    """Series whose MAXIMUM marks this challenge being performed.

    Normalising every challenge to "find the peak" is what lets one ordering
    check span head turns and blinks alike. Horizontal turns flip under
    ``mirrored`` because front-facing cameras commonly present a mirror image,
    which swaps left and right; vertical motion and blinks are unaffected.
    """
    if challenge is ChallengeType.BLINK:
        return blink if blink.size else None
    if challenge is ChallengeType.TURN_LEFT:
        return yaw if mirrored else -yaw
    if challenge is ChallengeType.TURN_RIGHT:
        return -yaw if mirrored else yaw
    if challenge is ChallengeType.LOOK_UP:
        return -pitch
    return pitch  # LOOK_DOWN


def _score_step(
    challenge: ChallengeType, yaw: np.ndarray, pitch: np.ndarray,
    blink: np.ndarray, mirrored: bool, search_from: int = 0,
) -> _Step:
    """Locate and score one challenge within the recording.

    ``search_from`` restricts the search to frames at or after that index, which
    is how a sequence is scored: each challenge is looked for only *after* the
    previous one was found. Without that constraint the blink challenge is
    unusable — people blink involuntarily every few seconds, so a global search
    routinely locks onto a spontaneous blink from early in the take and reports
    the sequence as out of order for a candidate who did everything correctly.
    """
    reasons: list[Reason] = []
    signal = _challenge_signal(challenge, yaw, pitch, blink, mirrored)
    if signal is None or signal.size < 3:
        reasons.append(rc.CHALLENGE_FRAMES_INSUFFICIENT(
            f"Insufficient frames to score the '{challenge.value}' challenge."))
        return _Step(challenge, False, 0.0, -1, reasons)

    # Excursion from the resting pose, not from frame 0 — a candidate who is
    # already mid-turn when recording starts would otherwise score zero. The
    # baseline is taken over the WHOLE clip even when the search is windowed:
    # resting pose is a property of the person, not of the window.
    baseline = float(np.median(signal))
    window = signal[search_from:]
    if window.size == 0:
        reasons.append(rc.CHALLENGE_FAIL(
            f"No frames left in which to find the '{challenge.value}' challenge."))
        return _Step(challenge, False, 0.0, -1, reasons)
    peak_index = search_from + int(np.argmax(window))
    excursion = float(signal[peak_index]) - baseline

    if challenge is ChallengeType.BLINK:
        score = clamp(linear_map(excursion, _BLINK_WEAK, _BLINK_STRONG, 0.0, 100.0))
        passed = excursion >= _BLINK_PASS
        reasons.append(rc.BLINK_DETECTED("Blink observed on cue.") if passed
                       else rc.NO_BLINK("No clear blink observed when prompted."))
        return _Step(challenge, passed, clamp(score), peak_index, reasons)

    score = clamp(linear_map(excursion, _MOTION_WEAK, _MOTION_STRONG, 0.0, 100.0))

    # The motion must be predominantly along the axis we asked for — a big
    # nod does not satisfy "turn left".
    horizontal = challenge in HORIZONTAL_CHALLENGES
    axis_range = float(yaw.max() - yaw.min()) if horizontal else float(pitch.max() - pitch.min())
    other_range = float(pitch.max() - pitch.min()) if horizontal else float(yaw.max() - yaw.min())
    axis_dominant = axis_range >= (other_range * 0.7)
    if not axis_dominant:
        score *= 0.7
        reasons.append(rc.CHALLENGE_AXIS_MISMATCH(
            f"Head motion during '{challenge.value}' was not predominantly along the "
            "requested axis."))

    passed = excursion >= _MOTION_PASS and axis_dominant
    if passed:
        reasons.append(rc.CHALLENGE_PASS(f"Successful '{challenge.value}' head-turn challenge."))
    else:
        reasons.append(rc.CHALLENGE_FAIL(
            f"Head motion for '{challenge.value}' was insufficient; turn more clearly on cue."))
    return _Step(challenge, passed, clamp(score), peak_index, reasons)


def _score_in_sequence(
    challenges: list[ChallengeType], yaw: np.ndarray, pitch: np.ndarray,
    blink: np.ndarray, mirrored: bool,
) -> list[_Step]:
    """Locate each challenge strictly after the previous one.

    Ordering is therefore satisfied *by construction* for every step that gets
    found — a challenge performed out of position simply isn't there to be found
    in its window and fails on magnitude, which is both the fair reading for a
    genuine candidate and the correct one for a spoof.
    """
    steps: list[_Step] = []
    cursor = 0
    for challenge in challenges:
        step = _score_step(challenge, yaw, pitch, blink, mirrored, search_from=cursor)
        steps.append(step)
        if step.peak_index >= 0:
            cursor = step.peak_index + 1
    return steps


def _happened_but_out_of_position(
    steps: list[_Step], challenges: list[ChallengeType], yaw: np.ndarray,
    pitch: np.ndarray, blink: np.ndarray, mirrored: bool,
) -> bool:
    """Did a failed challenge actually occur, just not where it was asked for?

    Sequential scoring alone cannot distinguish "never did it" from "did it in
    the wrong order" — both surface as a failed step. So for each failure we
    re-score it unconstrained: if it passes over the full recording, the action
    IS present and merely landed out of position. That is the stitched-clip /
    replayed-generic-footage signature worth flagging to a reviewer.
    """
    for step, challenge in zip(steps, challenges):
        if step.passed:
            continue
        if _score_step(challenge, yaw, pitch, blink, mirrored, search_from=0).passed:
            return True
    return False


def _challenge_stage(
    results: list[LandmarkResult],
    challenges: list[ChallengeType],
    mirrored: bool | None = None,
) -> tuple[float, bool, bool, list[_Step], list[Reason]]:
    """Score an ordered challenge sequence over ONE continuous recording.

    Returns ``(score, all_passed, sequence_passed, steps, reasons)``.

    Each challenge is located independently by its peak, then the peaks are
    checked for increasing order. Order is evidence in its own right: a spoof
    that splices separately-captured actions, or replays a generic "person
    moving" clip, routinely produces every requested action but not in the
    order this session asked for. A single-challenge request degrades to the
    pre-sequence behaviour — one step, order trivially satisfied.

    **Mirroring.** Front-facing cameras are commonly presented mirrored, which
    swaps left and right. From landmark yaw alone, a mirrored "left then right"
    is *bit-for-bit identical* to an unmirrored "right then left" — the two
    hypotheses are not merely hard to tell apart, they are the same signal. So:

    * ``mirrored`` declared by the caller (the browser knows whether it applied
      a ``scaleX(-1)``) — the ambiguity disappears and full ordering is enforced.
    * ``mirrored=None`` — we score both hypotheses and keep the better reading,
      but must NOT then claim to have verified the order of one horizontal turn
      against another. Those pairs are excluded from the ordering check and the
      result is flagged, so nobody mistakes a partially-verified sequence for a
      fully-verified one. Ordering against blinks and vertical moves still holds,
      since neither is affected by a horizontal flip.
    """
    reasons: list[Reason] = []
    valid = [r for r in results if r.found]
    if len(valid) < 4:
        reasons.append(rc.CHALLENGE_FRAMES_INSUFFICIENT(
            "Insufficient frames for the head-turn challenge."))
        return 0.0, False, False, [], reasons

    sigs = np.array([_head_signals(r.landmarks) for r in valid])  # (N, 2): yaw, pitch
    yaw, pitch = sigs[:, 0], sigs[:, 1]
    blink = _blink_series(results)
    # _blink_series filters on r.found the same way, so the two are aligned; be
    # defensive in case a caller passes pre-filtered results.
    if blink.size and blink.size != yaw.size:
        blink = np.asarray([])

    horizontal_count = sum(1 for c in challenges if c in HORIZONTAL_CHALLENGES)
    ambiguous = mirrored is None and horizontal_count > 1

    if mirrored is not None:
        steps = _score_in_sequence(challenges, yaw, pitch, blink, mirrored)
        hypothesis_used = mirrored
    else:
        # Rank hypotheses on what we can legitimately observe: how many
        # challenges were completed, then raw magnitude. Ties resolve to the
        # unmirrored reading so the outcome is deterministic, not arbitrary.
        best: tuple[tuple[int, float], list[_Step], bool] | None = None
        for hypothesis in (False, True):
            candidate = _score_in_sequence(challenges, yaw, pitch, blink, hypothesis)
            rank = (sum(1 for s in candidate if s.passed),
                    sum(s.score for s in candidate))
            if best is None or rank > best[0]:
                best = (rank, candidate, hypothesis)
        steps, hypothesis_used = best[1], best[2]

    reasons.extend(r for s in steps for r in s.reasons)

    all_passed = bool(steps) and all(s.passed for s in steps)
    score = float(np.mean([s.score for s in steps])) if steps else 0.0

    if len(challenges) > 1:
        out_of_position = (
            not all_passed
            and _happened_but_out_of_position(
                steps, challenges, yaw, pitch, blink, hypothesis_used)
        )
        if out_of_position:
            # Halve rather than zero: the actions did happen, so this is strong
            # evidence of a stitched or replayed clip but not proof, and the
            # reviewer sees the flag either way.
            score *= 0.5
            reasons.append(rc.CHALLENGE_SEQUENCE_OUT_OF_ORDER(
                "Challenges were not performed in the order requested "
                f"({' → '.join(c.value for c in challenges)})."))
        elif all_passed:
            if ambiguous:
                reasons.append(rc.CHALLENGE_SEQUENCE_ORDER_PARTIAL(
                    "All challenges completed, but left/right ordering could not be "
                    "verified because the camera's mirror state was not declared."))
            else:
                reasons.append(rc.CHALLENGE_SEQUENCE_PASS(
                    "All challenges completed in the requested order."))

    # Ordering is guaranteed by construction for located steps, so "the sequence
    # was satisfied" reduces to "every challenge was found where it was expected".
    return clamp(score), all_passed, all_passed, steps, reasons


# --------------------------------------------------------------------------- #
# Stage 5 - Depth verification (non-planarity / parallax)
# --------------------------------------------------------------------------- #
def _depth_stage(results: list[LandmarkResult]) -> tuple[float, bool, list[Reason]]:
    """A flat photo/screen maps frame-to-frame by a homography; a real 3D face
    leaves large residuals (parallax) under rotation."""
    reasons: list[Reason] = []
    valid = [r for r in results if r.found]
    if len(valid) < 4:
        reasons.append(rc.DEPTH_FRAMES_INSUFFICIENT("Insufficient frames for depth verification."))
        return 0.0, False, reasons

    # Pick the two frames with the most yaw separation (landmark-based signal).
    yaw = np.array([_head_signals(r.landmarks)[0] for r in valid])
    i_min, i_max = int(np.argmin(yaw)), int(np.argmax(yaw))
    if abs(yaw[i_max] - yaw[i_min]) < 0.06:
        # Genuinely cannot measure depth without rotation -> neutral, NOT planar.
        reasons.append(rc.DEPTH_ROTATION_INSUFFICIENT(
            "Not enough head rotation to measure depth (turn your head for this check)."))
        return 55.0, False, reasons

    a = valid[i_min].landmarks[RIGID_POINTS, :2].astype(np.float32)
    b = valid[i_max].landmarks[RIGID_POINTS, :2].astype(np.float32)
    if a.shape[0] < 8:
        return 50.0, False, reasons

    H, _ = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    if H is None:
        reasons.append(rc.DEPTH_MODEL_UNFIT("Could not fit planar model; depth check neutral."))
        return 55.0, False, reasons

    proj = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2)
    residual = np.linalg.norm(proj - b, axis=1)
    iod = _interocular(valid[i_min].landmarks)
    norm_residual = float(np.median(residual) / iod)

    # Planar surface (photo/screen) -> tiny residual; real 3D face -> larger.
    score = clamp(linear_map(norm_residual, 0.008, 0.06, 0.0, 100.0))
    passed = norm_residual >= 0.02
    if passed:
        reasons.append(rc.DEPTH_PASS("Strong depth parallax observed (genuine 3D structure)."))
    else:
        reasons.append(rc.DEPTH_PLANAR("Motion is planar; possible photo or screen replay."))
    return score, passed, reasons


# --------------------------------------------------------------------------- #
# Stage 6 - Micro-movement
# --------------------------------------------------------------------------- #
def _micro_movement_stage(results: list[LandmarkResult]) -> tuple[float, np.ndarray, list[Reason]]:
    reasons: list[Reason] = []
    valid = [r for r in results if r.found]
    if len(valid) < 5:
        reasons.append(rc.MICRO_FRAMES_INSUFFICIENT(
            "Insufficient frames for micro-movement analysis."))
        return 0.0, np.array([]), reasons

    series = []  # normalized micro-point positions per frame, nose-anchored
    for r in valid:
        lm = r.landmarks
        iod = _interocular(lm)
        nose = lm[NOSE_TIP, :2]
        pts = (lm[MICRO_POINTS, :2] - nose) / iod  # remove global translation
        series.append(pts.flatten())
    arr = np.asarray(series)
    motion = np.linalg.norm(np.diff(arr, axis=0), axis=1)  # per-frame micro displacement
    mean_motion = float(np.mean(motion))

    # Too still -> photo; healthy involuntary motion -> alive; too much -> noise.
    score = clamp(linear_map(mean_motion, 0.002, 0.03, 0.0, 100.0))
    if mean_motion < 0.002:
        reasons.append(rc.MICRO_STATIC("Almost no micro-movement; possible static image."))
    else:
        reasons.append(rc.MICRO_OK(
            "Involuntary micro-movements consistent with a live subject."))
    return score, motion, reasons


# --------------------------------------------------------------------------- #
# Stage 7 - Replay detection
# --------------------------------------------------------------------------- #
def _replay_stage(motion: np.ndarray) -> tuple[float, list[Reason]]:
    reasons: list[Reason] = []
    if motion.size < 8:
        return 70.0, reasons
    period = periodicity_score(motion)
    # High periodicity => looping replay => low resistance.
    resistance = clamp(linear_map(period, 0.2, 0.85, 100.0, 0.0))
    if period > 0.7:
        reasons.append(rc.REPLAY_PERIODIC(
            "Periodic, repeating motion detected (replay indicator)."))
    else:
        reasons.append(rc.REPLAY_PASS("No replay/looping characteristics detected."))
    return resistance, reasons


# --------------------------------------------------------------------------- #
# Optional - rPPG (advisory only)
# --------------------------------------------------------------------------- #
def _rppg_bpm(frames: list[np.ndarray], results: list[LandmarkResult], fps: float) -> float | None:
    greens = []
    for f, r in zip(frames, results):
        if not r.found:
            continue
        lm = r.landmarks
        # Forehead ROI above the eyes.
        x1 = int(lm[:, 0].min()); x2 = int(lm[:, 0].max())
        ytop = int(lm[:, 1].min())
        eye_y = int(lm[LEFT_EYE_OUTER, 1])
        roi = f[max(0, ytop):max(1, eye_y), max(0, x1):max(1, x2)]
        if roi.size == 0:
            continue
        greens.append(float(roi[:, :, 1].mean()))  # green channel
    g = np.asarray(greens)
    if g.size < int(fps * 2):  # need ~2s of signal
        return None
    freq, power = dominant_frequency(g, fps)
    if power < 0.1 or not (0.7 <= freq <= 4.0):
        return None
    return round(freq * 60.0, 1)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def verify_liveness(
    frames: list[np.ndarray],
    challenge: ChallengeType = ChallengeType.TURN_LEFT,
    fps: float = 15.0,
    challenges: list[ChallengeType] | None = None,
    mirrored: bool | None = None,
) -> LivenessResult:
    """Run the full Layer 3 active-liveness pipeline over a frame sequence.

    Pass ``challenges`` for an ordered sequence performed in one continuous
    recording; ``challenge`` remains for the single-challenge case. ``mirrored``
    declares the camera's flip state — see :func:`_challenge_stage` for why it
    matters to a left/right sequence.
    """
    sequence = challenges or [challenge]
    frames = sample_frames(
        frames, MAX_FRAMES_SEQUENCE if len(sequence) > 1 else MAX_FRAMES
    )
    if len(frames) < 4:
        return LivenessResult(
            liveness_score=0.0,
            reasons=[rc.LIVENESS_FRAMES_INSUFFICIENT(
                "Need at least 4 frames for liveness verification.")],
        )

    results = [detect_landmarks(f) for f in frames]
    found = sum(1 for r in results if r.found)
    if found < 3:
        return LivenessResult(
            liveness_score=0.0,
            reasons=[rc.FACE_NOT_TRACKED("Face was not reliably detected across the sequence.")],
        )

    position_score, r_pos = _position_stage(results)
    lighting_score, r_light = _lighting_stage(frames, results)
    blink_score, blinked, r_blink = _blink_stage(results)
    (challenge_score, challenge_passed, sequence_passed,
     steps, r_ch) = _challenge_stage(results, sequence, mirrored=mirrored)
    depth_score, depth_passed, r_depth = _depth_stage(results)
    motion_score, motion, r_micro = _micro_movement_stage(results)
    replay_score, r_replay = _replay_stage(motion)
    rppg = _rppg_bpm(frames, results, fps)

    base = (
        0.10 * position_score
        + 0.10 * lighting_score
        + 0.25 * blink_score
        + 0.25 * challenge_score
        + 0.20 * depth_score
        + 0.10 * motion_score
    )
    # Replay resistance gates multiplicatively (anti-spoof), never inflates.
    replay_factor = linear_map(replay_score, 0.0, 100.0, 0.55, 1.0)
    liveness_score = clamp(base * replay_factor)

    reasons = r_pos + r_light + r_blink + r_ch + r_depth + r_micro + r_replay
    if rppg is not None:
        reasons.append(rc.RPPG_ADVISORY(
            f"Advisory rPPG pulse estimate: {rppg:.0f} bpm (not used in scoring)."))

    logger.info(
        "Layer3 liveness={:.1f} pos={:.0f} light={:.0f} blink={:.0f} chal={:.0f} "
        "depth={:.0f} micro={:.0f} replay={:.0f}",
        liveness_score, position_score, lighting_score, blink_score,
        challenge_score, depth_score, motion_score, replay_score,
    )

    return LivenessResult(
        liveness_score=round(liveness_score, 1),
        position_score=round(position_score, 1),
        lighting_score=round(lighting_score, 1),
        blink_score=round(blink_score, 1),
        challenge_score=round(challenge_score, 1),
        depth_score=round(depth_score, 1),
        motion_score=round(motion_score, 1),
        replay_resistance_score=round(replay_score, 1),
        blink_detected=bool(blinked),
        challenge_passed=bool(challenge_passed),
        depth_passed=bool(depth_passed),
        challenge_sequence=[
            ChallengeStepResult(
                challenge=s.challenge,
                passed=s.passed,
                score=round(s.score, 1),
                peak_frame_index=s.peak_index,
            )
            for s in steps
        ],
        challenge_sequence_passed=bool(sequence_passed),
        rppg_bpm=rppg,
        reasons=reasons,
    )
