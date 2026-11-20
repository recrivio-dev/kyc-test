"""Score Aadhaar field extraction against hand-labelled ground truth.

    python tests/eval_aadhaar.py            # upright only
    python tests/eval_aadhaar.py --rot      # also 90/180/270 rotations
    python tests/eval_aadhaar.py -v         # print every wrong field

Needs the local sample images (sample/, sample-specifics/ — gitignored).

Names / IDs / dates / zip must match exactly after normalisation (case,
spaces and punctuation ignored). Address is scored by character similarity;
>= 0.90 counts as correct.
"""
from __future__ import annotations

import asyncio
import contextlib
import difflib
import io
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402

from kyc_pipeline import DocumentPipeline  # noqa: E402

TRUTH = json.load(open(os.path.join(ROOT, "tests/fixtures/aadhaar_truth.json")))
ADDR_OK = 0.90
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _addr_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _pick(fields: list, side: str):
    want = "aadhaar_back" if side == "back" else "aadhaar_front"
    for f in fields:
        if f.get("document_type", "").startswith(want):
            return f
    return None


async def main() -> None:
    angles = (0, 90, 180, 270) if "--rot" in sys.argv else (0,)
    verbose = "-v" in sys.argv
    p = DocumentPipeline()
    p._ensure_ocr()
    tally: dict = {}
    addr_sims = []
    tmp = tempfile.mkdtemp()
    for path, sides in TRUTH.items():
        for a in angles:
            src = path
            if a:
                if path.endswith(".pdf"):
                    continue
                src = os.path.join(tmp, f"r{a}_{os.path.basename(path)}.png")
                cv2.imwrite(src, cv2.rotate(cv2.imread(path), _ROT[a]))
            with contextlib.redirect_stdout(io.StringIO()):
                r = await p.process_and_verify(src, "AADHAAR")
            got_all = ((r.get("output_json") or {}).get("data") or {}).get(
                "ocr_fields") or []
            for truth in sides:
                got = _pick(got_all, truth["side"]) or {}
                for key, want in truth.items():
                    if key == "side":
                        continue
                    val = (got.get(key) or {}).get("value", "") if got else ""
                    if key == "address":
                        sim = _addr_sim(val, want)
                        addr_sims.append(sim)
                        ok = sim >= ADDR_OK
                    else:
                        ok = _norm(val) == _norm(want)
                    t = tally.setdefault(key, [0, 0])
                    t[0] += ok
                    t[1] += 1
                    if verbose and not ok:
                        print(f"  x {path}@{a} {truth['side']}.{key}: "
                              f"got {val!r} want {want!r}")
    total = [0, 0]
    for key, (ok, n) in sorted(tally.items()):
        total[0] += ok
        total[1] += n
        print(f"{key:15s} {ok:3d}/{n:<3d}")
    print(f"{'ALL':15s} {total[0]:3d}/{total[1]:<3d} "
          f"({100 * total[0] / total[1]:.0f}%)  "
          f"mean address similarity {sum(addr_sims) / len(addr_sims):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
