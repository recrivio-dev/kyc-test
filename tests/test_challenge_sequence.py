"""Ordered challenge-sequence scoring (Layer 3).

Drives ``_challenge_stage`` with synthetic landmark frames instead of real
video: the point under test is the temporal ordering logic, and a real capture
would make "did it fail because of ordering, or because MediaPipe read the pose
differently today?" unanswerable.

Frames are built by placing landmarks so ``_head_signals`` reports a chosen
yaw/pitch, which lets each test state its intent as a movement script.
"""
from __future__ import annotations

import numpy as np
import pytest

from liveness.schemas.requests import ChallengeType
from liveness.services.layer3_liveness import (
    CHIN,
    FOREHEAD,
    LEFT_CHEEK,
    NOSE_TIP,
    RIGHT_CHEEK,
    _challenge_stage,
    _head_signals,
)
from liveness.utils.face_engine import LandmarkResult

N_POINTS = 478
FRAME_W, FRAME_H = 640, 480


def _frame(yaw: float, blink: float = 0.0) -> LandmarkResult:
    """One synthetic frame whose measured yaw is approximately ``yaw``.

    ``_head_signals`` derives yaw from how much closer the nose sits to one
    cheek than the other, so positioning the nose along the cheek-to-cheek axis
    is enough to drive it. Pitch is pinned at centre.
    """
    lm = np.zeros((N_POINTS, 3), dtype=np.float64)

    # Face box, wide enough that landmark spread doesn't distort the position
    # stage if it ever runs over these frames.
    cx, cy = FRAME_W / 2.0, FRAME_H / 2.0
    half_w = 120.0
    lm[:, 0] = cx
    lm[:, 1] = cy

    lm[LEFT_CHEEK] = [cx - half_w, cy, 0.0]
    lm[RIGHT_CHEEK] = [cx + half_w, cy, 0.0]
    lm[FOREHEAD] = [cx, cy - 100.0, 0.0]
    lm[CHIN] = [cx, cy + 100.0, 0.0]

    # yaw = (d_right - d_left) / (d_right + d_left); with the nose on the axis
    # at offset t from centre this reduces to -t / half_w. Invert for the target.
    lm[NOSE_TIP] = [cx - yaw * half_w, cy, 0.0]

    return LandmarkResult(
        landmarks=lm,
        normalized=lm / np.array([FRAME_W, FRAME_H, 1.0]),
        blendshapes={"eyeBlinkLeft": blink, "eyeBlinkRight": blink},
        pose={"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        image_size=(FRAME_W, FRAME_H),
    )


def _hold(yaw: float, n: int, blink: float = 0.0) -> list[LandmarkResult]:
    return [_frame(yaw, blink) for _ in range(n)]


# Excursions comfortably past _MOTION_PASS (0.09) and _BLINK_PASS (0.30).
LEFT, RIGHT, NEUTRAL = -0.35, 0.35, 0.0
BLINK_ON = 0.9


def test_frame_builder_produces_requested_yaw():
    """Guard the fixture itself — every other test's meaning depends on it."""
    assert _head_signals(_frame(0.0).landmarks)[0] == pytest.approx(0.0, abs=1e-6)
    assert _head_signals(_frame(-0.35).landmarks)[0] == pytest.approx(-0.35, abs=1e-6)
    assert _head_signals(_frame(0.35).landmarks)[0] == pytest.approx(0.35, abs=1e-6)


def _left_then_right_then_blink() -> list[LandmarkResult]:
    """The production script: turn left, turn right, blink — in that order."""
    return (
        _hold(NEUTRAL, 4)
        + _hold(LEFT, 5)
        + _hold(NEUTRAL, 3)
        + _hold(RIGHT, 5)
        + _hold(NEUTRAL, 3)
        + _hold(NEUTRAL, 2, blink=BLINK_ON)
        + _hold(NEUTRAL, 3)
    )


SEQUENCE = [ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT, ChallengeType.BLINK]


def test_correct_order_passes():
    score, all_passed, seq_passed, steps, _ = _challenge_stage(
        _left_then_right_then_blink(), SEQUENCE, mirrored=False
    )
    assert all_passed, [(s.challenge, s.score) for s in steps]
    assert seq_passed
    assert score > 0
    assert [s.challenge for s in steps] == SEQUENCE
    # Peaks must be strictly increasing — that IS the ordering claim.
    peaks = [s.peak_index for s in steps]
    assert peaks == sorted(peaks)
    assert all(p >= 0 for p in peaks)


def test_actions_performed_in_wrong_order_fail_the_sequence():
    """Every requested action happens, but right precedes left.

    This is the spoof shape the check exists for: a stitched or replayed clip
    that contains all the right movements but not on this session's cue order.
    Only detectable because the camera's mirror state is declared — see
    :func:`test_wrong_order_is_unverifiable_when_mirroring_undeclared`.
    """
    frames = (
        _hold(NEUTRAL, 4)
        + _hold(RIGHT, 5)          # <- requested second, performed first
        + _hold(NEUTRAL, 3)
        + _hold(LEFT, 5)
        + _hold(NEUTRAL, 3)
        + _hold(NEUTRAL, 2, blink=BLINK_ON)
        + _hold(NEUTRAL, 3)
    )
    _, all_passed, seq_passed, steps, reasons = _challenge_stage(
        frames, SEQUENCE, mirrored=False
    )

    # Challenges are located strictly after one another, so the right turn is
    # not available in its window and fails there — and the re-check confirms
    # the action DID occur, just out of position, which is the flag that matters.
    assert not all_passed
    assert not seq_passed
    assert any(r.code == "CHALLENGE_SEQUENCE_OUT_OF_ORDER" for r in reasons)
    by_challenge = {s.challenge: s for s in steps}
    assert by_challenge[ChallengeType.TURN_LEFT].passed
    assert not by_challenge[ChallengeType.TURN_RIGHT].passed


def test_out_of_order_scores_below_correct_order():
    """Same actions, same magnitudes — ordering alone must cost score."""
    good = _left_then_right_then_blink()
    bad = (
        _hold(NEUTRAL, 4)
        + _hold(RIGHT, 5)
        + _hold(NEUTRAL, 3)
        + _hold(LEFT, 5)
        + _hold(NEUTRAL, 3)
        + _hold(NEUTRAL, 2, blink=BLINK_ON)
        + _hold(NEUTRAL, 3)
    )
    good_score, _, _, _, _ = _challenge_stage(good, SEQUENCE, mirrored=False)
    bad_score, _, _, _, _ = _challenge_stage(bad, SEQUENCE, mirrored=False)
    assert bad_score < good_score


def test_wrong_order_is_unverifiable_when_mirroring_undeclared():
    """A mirrored left→right and an unmirrored right→left are the SAME signal.

    So with ``mirrored=None`` the service must not claim the sequence failed —
    it cannot know. It reports the sequence as passing but flags that left/right
    ordering went unverified, rather than silently implying it was checked.
    """
    frames = (
        _hold(NEUTRAL, 4)
        + _hold(RIGHT, 5)
        + _hold(NEUTRAL, 3)
        + _hold(LEFT, 5)
        + _hold(NEUTRAL, 3)
        + _hold(NEUTRAL, 2, blink=BLINK_ON)
        + _hold(NEUTRAL, 3)
    )
    _, all_passed, seq_passed, _, reasons = _challenge_stage(frames, SEQUENCE)
    assert all_passed
    assert seq_passed
    codes = {r.code for r in reasons}
    assert "CHALLENGE_SEQUENCE_ORDER_PARTIAL" in codes
    assert "CHALLENGE_SEQUENCE_PASS" not in codes


def test_blink_ordering_holds_even_when_mirroring_undeclared():
    """Mirroring flips left/right — it cannot move a blink.

    So "both turns before the blink" stays verifiable with ``mirrored=None``.
    """
    frames = (
        _hold(NEUTRAL, 3)
        + _hold(NEUTRAL, 2, blink=BLINK_ON)   # blink FIRST, but requested last
        + _hold(NEUTRAL, 3)
        + _hold(LEFT, 5)
        + _hold(NEUTRAL, 3)
        + _hold(RIGHT, 5)
        + _hold(NEUTRAL, 3)
    )
    _, all_passed, seq_passed, steps, reasons = _challenge_stage(frames, SEQUENCE)
    assert not all_passed      # the blink is not there AFTER the turns
    assert not seq_passed
    assert any(r.code == "CHALLENGE_SEQUENCE_OUT_OF_ORDER" for r in reasons)
    # The turns themselves were fine; only the blink landed out of position.
    by_challenge = {s.challenge: s for s in steps}
    assert by_challenge[ChallengeType.TURN_LEFT].passed
    assert by_challenge[ChallengeType.TURN_RIGHT].passed
    assert not by_challenge[ChallengeType.BLINK].passed


def test_spontaneous_early_blink_does_not_break_a_correct_sequence():
    """The regression that made a genuine candidate fail.

    People blink involuntarily every few seconds. Here the candidate blinks once
    while looking straight — harder than their deliberate blink — then performs
    every challenge correctly. A global peak search would award the blink
    challenge to that first involuntary blink, put its peak BEFORE the turns,
    and report the sequence as out of order. Scoring each challenge only after
    the previous one keeps that from happening.
    """
    frames = (
        _hold(NEUTRAL, 2)
        + _hold(NEUTRAL, 2, blink=1.0)      # involuntary, and STRONGER
        + _hold(NEUTRAL, 3)
        + _hold(LEFT, 5)
        + _hold(NEUTRAL, 3)
        + _hold(RIGHT, 5)
        + _hold(NEUTRAL, 2)
        + _hold(NEUTRAL, 2, blink=BLINK_ON)  # the deliberate one, on cue
        + _hold(NEUTRAL, 3)
    )
    _, all_passed, seq_passed, steps, reasons = _challenge_stage(
        frames, SEQUENCE, mirrored=False
    )
    assert all_passed
    assert seq_passed
    assert not any(r.code == "CHALLENGE_SEQUENCE_OUT_OF_ORDER" for r in reasons)
    # The blink credited must be the LATE one, after both turns.
    by_challenge = {s.challenge: s for s in steps}
    assert (by_challenge[ChallengeType.BLINK].peak_index
            > by_challenge[ChallengeType.TURN_RIGHT].peak_index)


def test_missing_challenge_fails_that_step_only():
    """Candidate turns left and blinks but never turns right."""
    frames = (
        _hold(NEUTRAL, 4)
        + _hold(LEFT, 6)
        + _hold(NEUTRAL, 4)
        + _hold(NEUTRAL, 2, blink=BLINK_ON)
        + _hold(NEUTRAL, 4)
    )
    _, all_passed, seq_passed, steps, _ = _challenge_stage(
        frames, SEQUENCE, mirrored=False
    )
    by_challenge = {s.challenge: s for s in steps}

    assert not all_passed
    assert not seq_passed
    assert by_challenge[ChallengeType.TURN_LEFT].passed
    assert by_challenge[ChallengeType.BLINK].passed
    assert not by_challenge[ChallengeType.TURN_RIGHT].passed


def test_no_blink_fails_the_blink_step():
    frames = (
        _hold(NEUTRAL, 4) + _hold(LEFT, 5) + _hold(NEUTRAL, 3)
        + _hold(RIGHT, 5) + _hold(NEUTRAL, 4)   # eyes open throughout
    )
    _, all_passed, _, steps, _ = _challenge_stage(frames, SEQUENCE, mirrored=False)
    by_challenge = {s.challenge: s for s in steps}
    assert not by_challenge[ChallengeType.BLINK].passed
    assert not all_passed


def test_declared_mirrored_camera_passes_in_order():
    """A mirrored front camera swaps left and right in the raw signal.

    The candidate genuinely followed left → right → blink; the camera flipped
    the image. With the flip declared, that must read as a clean pass.
    """
    frames = (
        _hold(NEUTRAL, 4)
        + _hold(RIGHT, 5)          # mirrored: this IS their left turn
        + _hold(NEUTRAL, 3)
        + _hold(LEFT, 5)           # ...and this their right
        + _hold(NEUTRAL, 3)
        + _hold(NEUTRAL, 2, blink=BLINK_ON)
        + _hold(NEUTRAL, 3)
    )
    _, all_passed, seq_passed, _, reasons = _challenge_stage(
        frames, SEQUENCE, mirrored=True
    )
    assert all_passed
    assert seq_passed
    assert any(r.code == "CHALLENGE_SEQUENCE_PASS" for r in reasons)


def test_same_capture_read_with_wrong_mirror_flag_fails():
    """Declaring the flip wrongly inverts the reading — the flag must matter."""
    frames = (
        _hold(NEUTRAL, 4) + _hold(RIGHT, 5) + _hold(NEUTRAL, 3)
        + _hold(LEFT, 5) + _hold(NEUTRAL, 3)
        + _hold(NEUTRAL, 2, blink=BLINK_ON) + _hold(NEUTRAL, 3)
    )
    _, _, mirrored_ok, _, _ = _challenge_stage(frames, SEQUENCE, mirrored=True)
    _, _, unmirrored_ok, _, _ = _challenge_stage(frames, SEQUENCE, mirrored=False)
    assert mirrored_ok
    assert not unmirrored_ok


def test_single_challenge_needs_no_ordering():
    """One challenge degrades to the pre-sequence behaviour."""
    frames = _hold(NEUTRAL, 4) + _hold(LEFT, 6) + _hold(NEUTRAL, 4)
    score, all_passed, seq_passed, steps, reasons = _challenge_stage(
        frames, [ChallengeType.TURN_LEFT], mirrored=False
    )
    assert all_passed
    assert seq_passed          # trivially ordered
    assert score > 0
    assert len(steps) == 1
    # No sequence commentary when there is no sequence.
    assert not any(r.code.startswith("CHALLENGE_SEQUENCE") for r in reasons)


def test_single_challenge_undeclared_mirror_is_not_flagged_partial():
    """One horizontal turn has no left/right ORDERING to leave unverified."""
    frames = _hold(NEUTRAL, 4) + _hold(LEFT, 6) + _hold(NEUTRAL, 4)
    _, _, _, _, reasons = _challenge_stage(frames, [ChallengeType.TURN_LEFT])
    assert not any(r.code == "CHALLENGE_SEQUENCE_ORDER_PARTIAL" for r in reasons)


def test_static_capture_passes_nothing():
    """A held photo: no motion, no blink."""
    _, all_passed, seq_passed, steps, _ = _challenge_stage(
        _hold(NEUTRAL, 20), SEQUENCE, mirrored=False
    )
    assert not all_passed
    assert not seq_passed
    assert not any(s.passed for s in steps)


def test_too_few_frames_is_reported_not_crashed():
    score, all_passed, seq_passed, steps, reasons = _challenge_stage(
        _hold(NEUTRAL, 2), SEQUENCE
    )
    assert score == 0.0
    assert not all_passed and not seq_passed
    assert steps == []
    assert any(r.code == "CHALLENGE_FRAMES_INSUFFICIENT" for r in reasons)


def test_every_reason_carries_a_code_and_message():
    """The contract recriauth mirrors: code for machines, prose for humans."""
    _, _, _, _, reasons = _challenge_stage(
        _left_then_right_then_blink(), SEQUENCE, mirrored=False
    )
    assert reasons
    for r in reasons:
        assert r.code and r.code.isupper()
        assert r.message and r.message[0].isupper()
        assert r.severity in {"low", "medium", "high"}
