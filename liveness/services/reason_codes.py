"""Stable machine-readable codes for every reason the pipeline can emit.

Each layer explains its score with human-readable sentences. Those sentences are
deliberately user-facing — ``docs/liveness-api.md`` §7.3 tells integrators to
render them straight back to the candidate on a retry — so they are written for
a person, get reworded when the copy improves, and interpolate live values.

That makes them useless as a key. A reviewer dashboard needs to group, filter
and count "how many verifications failed the blink challenge this week", and a
consumer needs to branch on a condition without string-matching English prose.

So every reason carries BOTH: a ``code`` that never changes, and the ``message``
exactly as it was already written. Adding a code is not a reason to reword the
prose, and rewording the prose is not a reason to change the code.

``severity`` is about how much the reason should worry a human reviewer, NOT how
much it moved the score — the scores already say that:

* ``low``    - informational, or an outright positive signal.
* ``medium`` - degrades confidence; usually a capture problem the candidate
               could fix by retrying (bad light, too few frames, weak turn).
* ``high``   - an active fraud indicator, or a hard failure of the check.

recriauth mirrors this enum on its side; treat the string values as a published
contract and only ever append to them.
"""
from __future__ import annotations

from liveness.schemas.responses import Reason, ReasonSeverity


def _code(code: str, severity: ReasonSeverity):
    """Build a factory for one reason code.

    The message stays a call-site argument because most of them interpolate
    measured values (similarity, fps, the matched virtual-camera signature) and
    would lose all their diagnostic value as a fixed string.
    """

    def make(message: str) -> Reason:
        return Reason(code=code, message=message, severity=severity)

    make.code = code  # type: ignore[attr-defined]
    return make


# --------------------------------------------------------------------------- #
# Layer 1 - capture integrity
# --------------------------------------------------------------------------- #
TIMING_DATA_INSUFFICIENT = _code("TIMING_DATA_INSUFFICIENT", "medium")
TIMING_DATA_DEGENERATE = _code("TIMING_DATA_DEGENERATE", "medium")
CADENCE_DETERMINISTIC = _code("CADENCE_DETERMINISTIC", "high")
TIMING_IRREGULAR = _code("TIMING_IRREGULAR", "high")
FPS_OUT_OF_RANGE = _code("FPS_OUT_OF_RANGE", "medium")
REAL_CAMERA = _code("REAL_CAMERA", "low")
METADATA_MISSING = _code("METADATA_MISSING", "medium")
VIRTUAL_CAMERA_DETECTED = _code("VIRTUAL_CAMERA_DETECTED", "high")
BLACKLISTED_DRIVER = _code("BLACKLISTED_DRIVER", "high")
NO_VIRTUAL_CAMERA = _code("NO_VIRTUAL_CAMERA", "low")
ENTROPY_FRAMES_INSUFFICIENT = _code("ENTROPY_FRAMES_INSUFFICIENT", "medium")
FRAME_DUPLICATES = _code("FRAME_DUPLICATES", "high")
LOW_TEMPORAL_VARIANCE = _code("LOW_TEMPORAL_VARIANCE", "high")
FRAME_VARIATION_OK = _code("FRAME_VARIATION_OK", "low")

# --------------------------------------------------------------------------- #
# Layer 2 - identity
# --------------------------------------------------------------------------- #
NO_FACE_IN_REFERENCE = _code("NO_FACE_IN_REFERENCE", "high")
NO_FACE_IN_PROBE = _code("NO_FACE_IN_PROBE", "high")
REFERENCE_AUTO_ROTATED = _code("REFERENCE_AUTO_ROTATED", "low")
PROBE_AUTO_ROTATED = _code("PROBE_AUTO_ROTATED", "low")
PORTRAIT_EXTRACTED = _code("PORTRAIT_EXTRACTED", "low")
REFERENCE_RESTORED = _code("REFERENCE_RESTORED", "low")
FACE_MATCH_PASS = _code("FACE_MATCH_PASS", "low")
FACE_MATCH_FAIL = _code("FACE_MATCH_FAIL", "high")
FACE_MATCH_GEOMETRY_CORROBORATED = _code("FACE_MATCH_GEOMETRY_CORROBORATED", "low")
FACE_MATCH_PERIOCULAR_CORROBORATED = _code("FACE_MATCH_PERIOCULAR_CORROBORATED", "low")
FORGIVING_THRESHOLD = _code("FORGIVING_THRESHOLD", "medium")
GEOMETRY_CONSISTENT = _code("GEOMETRY_CONSISTENT", "low")
GEOMETRY_DIVERGENT = _code("GEOMETRY_DIVERGENT", "medium")
MULTIPLE_FACES = _code("MULTIPLE_FACES", "high")

# --------------------------------------------------------------------------- #
# Layer 3 - active liveness
# --------------------------------------------------------------------------- #
LIVENESS_FRAMES_INSUFFICIENT = _code("LIVENESS_FRAMES_INSUFFICIENT", "medium")
FACE_NOT_TRACKED = _code("FACE_NOT_TRACKED", "high")
NO_FACE_FOR_POSITION = _code("NO_FACE_FOR_POSITION", "high")
POSITION_POOR = _code("POSITION_POOR", "medium")
POSITION_OK = _code("POSITION_OK", "low")
LIGHTING_UNDEREXPOSED = _code("LIGHTING_UNDEREXPOSED", "medium")
LIGHTING_OK = _code("LIGHTING_OK", "low")
BLINK_FRAMES_INSUFFICIENT = _code("BLINK_FRAMES_INSUFFICIENT", "medium")
BLINK_DETECTED = _code("BLINK_DETECTED", "low")
NO_BLINK = _code("NO_BLINK", "medium")
CHALLENGE_FRAMES_INSUFFICIENT = _code("CHALLENGE_FRAMES_INSUFFICIENT", "medium")
CHALLENGE_AXIS_MISMATCH = _code("CHALLENGE_AXIS_MISMATCH", "medium")
CHALLENGE_PASS = _code("CHALLENGE_PASS", "low")
CHALLENGE_PASS_MIRRORED = _code("CHALLENGE_PASS_MIRRORED", "low")
CHALLENGE_FAIL = _code("CHALLENGE_FAIL", "medium")
CHALLENGE_SEQUENCE_PASS = _code("CHALLENGE_SEQUENCE_PASS", "low")
CHALLENGE_SEQUENCE_OUT_OF_ORDER = _code("CHALLENGE_SEQUENCE_OUT_OF_ORDER", "high")
# Everything was performed, but the camera's mirror state was not declared, so
# left-vs-right ordering could not be checked. Medium, not low: the sequence is
# weaker evidence than a fully-verified one and a reviewer should know that.
CHALLENGE_SEQUENCE_ORDER_PARTIAL = _code("CHALLENGE_SEQUENCE_ORDER_PARTIAL", "medium")
DEPTH_FRAMES_INSUFFICIENT = _code("DEPTH_FRAMES_INSUFFICIENT", "medium")
DEPTH_ROTATION_INSUFFICIENT = _code("DEPTH_ROTATION_INSUFFICIENT", "medium")
DEPTH_MODEL_UNFIT = _code("DEPTH_MODEL_UNFIT", "medium")
DEPTH_PASS = _code("DEPTH_PASS", "low")
DEPTH_PLANAR = _code("DEPTH_PLANAR", "high")
MICRO_FRAMES_INSUFFICIENT = _code("MICRO_FRAMES_INSUFFICIENT", "medium")
MICRO_STATIC = _code("MICRO_STATIC", "high")
MICRO_OK = _code("MICRO_OK", "low")
REPLAY_PERIODIC = _code("REPLAY_PERIODIC", "high")
REPLAY_PASS = _code("REPLAY_PASS", "low")
RPPG_ADVISORY = _code("RPPG_ADVISORY", "low")

# --------------------------------------------------------------------------- #
# Capture quality (utils/quality.py) - surfaced as `quality.issues`
# --------------------------------------------------------------------------- #
BLUR = _code("BLUR", "medium")
LOW_LIGHT = _code("LOW_LIGHT", "medium")
OVEREXPOSED = _code("OVEREXPOSED", "medium")
LOW_CONTRAST = _code("LOW_CONTRAST", "medium")
LOW_RESOLUTION = _code("LOW_RESOLUTION", "medium")
FACE_TOO_SMALL = _code("FACE_TOO_SMALL", "medium")
