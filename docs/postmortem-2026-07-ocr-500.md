# Post-mortem — KYC OCR service returning 500s

- **Status:** Resolved
- **Severity:** SEV-2 (in-house OCR down; no hard user-facing outage — masked by vendor fallback)
- **Broke:** on/around 2026-07-10 (deploy of PR #8, `mask-cropped-dual-image`)
- **Root-caused & permanently fixed:** 2026-07-14
- **Author:** engineering

## Summary

A deploy of the in-house KYC OCR service (`recriviodev/kyc-ocr`) returned HTTP 500 on **every** `/api/v1/ocr/*` request. The container reported **healthy** the entire time, because the healthcheck only checked that models had loaded — not that the pipeline actually ran. It was mitigated by pinning the runtime back to a pre-deploy image snapshot (~2026-07-10); the underlying code was not fixed until 2026-07-14.

## Impact

- In-house OCR unavailable for all document types while the broken image was live.
- **No hard user-facing outage:** recriauth silently falls back to the paid SurePass vendor on any OCR failure, so candidate verifications kept working — at higher vendor cost and with **no alert**.
- The silent fallback is why nothing paged and why the window is only approximately known.

## Root cause — a stack of latent issues, not one bug

1. **The crash.** `estimate_skew_angle` iterated `lines[:, 0]`, assuming `cv2.HoughLinesP` returns shape `(N, 1, 4)`. The OpenCV build baked into the image returned `(N, 4)`, so the unpack raised `TypeError: cannot unpack non-iterable numpy.int32 object` on every request.
2. **Why the OpenCV shape differed.** `rapidocr-onnxruntime` depends on the full `opencv-python` **unpinned** (`>=4.5.1.48`). It floated to a newer major (`5.0.0.93`) whose `HoughLinesP` return shape differs from older builds. Non-reproducible dependency → the effective `cv2` changed between builds.
3. **Why it shipped green.** `/healthz` only reported "models loaded." A container that 500'd on 100% of real requests still passed the healthcheck, so the deploy looked successful.

Two more latent traps surfaced during remediation (each would have blocked a naive fix):

- The base image `python:3.11-slim` is an **unpinned, moving tag**; it had drifted from Debian 12 (bookworm) to 13 (trixie), where `libgl1` no longer provides `libGL.so.1` → `import cv2` crashed at container boot.
- `surya-ocr==0.17.0` **hard-pins** `opencv-python-headless==4.11.0.86`, so simply "bumping OpenCV" is impossible (pip `ResolutionImpossible`).

## Detection

Not detected by monitoring. Healthcheck was green; the silent SurePass fallback hid the user impact. Found only during a manual investigation days later.

## Resolution (2026-07-14)

- **Code fix:** `estimate_skew_angle` now iterates `lines.reshape(-1, 4)` — shape-agnostic, robust to any OpenCV build.
- **Reproducibility:** pinned the base image to `python:3.11-slim-bookworm`; pinned `opencv-python==5.0.0.93` (the full build rapidocr was floating).
- **Detection:** `/healthz` now runs a startup self-test that drives a synthetic document through the **entire** pipeline and returns `503` if it raises. A boots-but-crashes build can no longer report healthy.
- **Ops:** deploy/rollback are now scripted (`deploy/deploy.sh`, `deploy/rollback.sh`) with pre-deploy snapshot, verify-boots-and-serves, and auto-rollback on failure.

## What went well

- Rollback discipline: a `rollback-pre-<date>` image snapshot existed, so recovery was a one-line tag flip.
- Blast radius was contained by the SurePass fallback (which, however, also hid the failure).

## Action items

| # | Action | Status |
|---|--------|--------|
| 1 | Shape-agnostic `estimate_skew_angle` (`reshape(-1,4)`) | ✅ done |
| 2 | Pipeline-exercising `/healthz` startup self-test | ✅ done |
| 3 | Pin base image (bookworm) + pin `opencv-python` | ✅ done |
| 4 | Scripted deploy/rollback with verify + auto-rollback | ✅ done |
| 5 | Weekly `docker builder prune` cron; resize 28 GB root (hit 88%) | ⏳ cron done · resize open |
| 6 | **Alert when recriauth falls back to SurePass** — make the silent fallback visible | ⛔ open (highest-value prevention) |
| 7 | CI: build the image + run the pipeline self-test on every PR, so this is caught pre-merge | ⛔ open |

## Lessons

- A healthcheck must exercise the actual work, not just "did the process start."
- Pin **every** dependency — transitive packages (`opencv-python`) and base images (`python:3.11-slim`) each independently broke a build here.
- A silent vendor fallback prevents user-facing outages *and* hides that your own service is down. It needs its own alert, or failures stay invisible until someone looks.
