"""Face liveness + ID-photo face-match layers.

Ported from the standalone ``live-mini`` service so liveness deploys inside the
same container as the OCR pipeline (one image, one venv, one process). The
algorithms, thresholds and tuning comments are unchanged from the source; only
the logging shim, the settings loader and the InsightFace module set were
rewritten to fit this service's dependency footprint.

Public entry point: ``liveness.router.router`` — mounted by ``api.py``.
"""
