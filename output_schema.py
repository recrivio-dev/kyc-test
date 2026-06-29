"""Build the response JSON contract for downstream consumers.

The pipeline gives us per-region OCR results (text + confidence + bbox).
This module turns that into the document-type-specific JSON shape the
frontend expects.

Field extraction here is heuristic: we use regex over the joined text
where possible and fall back to layout-aware label→value rules. Anything
we can't recover confidently is emitted as an empty value with
confidence 0 so the JSON shape itself stays stable.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Envelope helpers
# ──────────────────────────────────────────────────────────────────────────────


def _conf(confidence: Optional[float]) -> int:
    """Scale a 0..1 OCR score to a 0..100 integer.

    The frontend wants the genuine OCR-reported confidence, so this is the
    region-level score that actually produced the value — no fabrication."""
    return int(round(float(confidence or 0.0) * 100.0))


def _field(value: str, confidence: Optional[float],
           **extra) -> Dict[str, Any]:
    out: Dict[str, Any] = {"value": value or "",
                           "confidence": _conf(confidence)}
    out.update(extra)
    return out


def _envelope(data: Dict[str, Any], *, success: bool = True,
              status: int = 200, message: Optional[str] = None,
              message_code: str = "success") -> Dict[str, Any]:
    return {
        "data": data,
        "status_code": status,
        "message_code": message_code,
        "message": message,
        "success": success,
    }


def failure_envelope(message: str, *, status: int = 400) -> Dict[str, Any]:
    return _envelope({}, success=False, status=status,
                     message=message, message_code="failed")


def success_envelope(data: Dict[str, Any], *, status: int = 200) -> Dict[str, Any]:
    """Wrap an arbitrary payload in the standard success envelope — used by
    endpoints (e.g. /api/v1/ocr/mask-identity) whose payload isn't a per-doc OCR shape."""
    return _envelope(data, success=True, status=status,
                     message=None, message_code="success")


# Common Indian surnames — used to recover the first/last-name boundary
# in an all-caps glued OCR token ('RAHULGUPTA' → 'RAHUL GUPTA'), which
# has neither a space nor a case change to split on. Longest entries are
# tried first so 'AGGARWAL' wins over any shorter accidental suffix.
_COMMON_SURNAMES = sorted((
    "SRIVASTAVA", "BHARDWAJ", "AGARWAL", "AGGARWAL", "MALHOTRA", "CHATURVEDI",
    "TRIPATHI", "SACHDEVA", "CHAUHAN", "RATHORE", "KULKARNI", "MUKHERJEE",
    "CHATTERJEE", "BANERJEE", "DESHMUKH", "GAIKWAD", "WADHWA", "KAPOOR",
    "KHANNA", "CHOPRA", "MEHROTRA", "PANDEY", "MISHRA", "TIWARI", "DUBEY",
    "SHUKLA", "DWIVEDI", "SAXENA", "THAKUR", "SHARMA", "VERMA", "GUPTA",
    "YADAV", "PATEL", "REDDY", "NAIDU", "PILLAI", "MEHTA", "BANSAL",
    "MITTAL", "ARORA", "SETHI", "DESAI", "JOSHI", "MALIK", "PRASAD",
    "CHANDRA", "NAHATA", "SINGHAL", "AGRAWAL", "GOENKA", "JAISWAL",
    "SINHA", "GHOSH", "NAIR", "IYER", "MENON", "SHAH", "JAIN", "GOEL",
    "GARG", "KHAN", "BOSE", "DASS", "NEGI", "RAWAT", "BISHT", "SIDHU",
    "DHILLON", "SANDHU", "BAJWA", "GREWAL", "BHATIA", "ANAND", "GROVER",
    "NANDA", "TANEJA", "NAGPAL", "RANA", "MEHRA", "KAUR", "GILL", "BRAR",
    "SOOD", "SETH", "DAS", "SEN", "ROY", "RAO", "BHAT", "BHATT", "KUMAR",
    "SINGH", "NATH",
    # Maharashtra / Marathi surnames — frequent on the sample licences and
    # otherwise absent from the list, so glued tokens like 'MPALANDE' or
    # 'NIVRUTTIBODAKE' never recover their first/last split.
    "DESHPANDE", "DESHMUKH", "KULKARNI", "BHOSALE", "SALUNKE", "PALANDE",
    "KARANDE", "GAIKWAD", "JADHAV", "SHINDE", "CHAVAN", "SAWANT", "BODAKE",
    "THORAT", "NIKAM", "SHELAR", "KSHIRSAGAR", "WAGHMARE", "PAWAR", "PATIL",
    "KADAM", "SHIRKE", "BHOSLE", "MANE", "MORE",
), key=len, reverse=True)


def _split_glued_surname(token: str) -> str:
    """Split an all-caps glued name on a trailing known surname.

    'RAHULGUPTA' → 'RAHUL GUPTA', 'MPALANDE' → 'M PALANDE'. Fires when the
    remaining prefix is a plausible given name (≥2 chars) or a single
    initial sitting before a clearly-surname-length suffix (≥5 chars); a
    token that *is* exactly a surname is left untouched."""
    up = token.upper()
    for sn in _COMMON_SURNAMES:
        if not up.endswith(sn):
            continue
        prefix = len(up) - len(sn)
        if prefix >= 2 or (prefix == 1 and len(sn) >= 5):
            return f"{token[:-len(sn)]} {token[-len(sn):]}"
    return token


def _clean_name(text: str) -> str:
    """Normalise a person-name string.

    OCR mangles the spacing between a first and last name several ways:
      * it splits them across text lines      → 'JAY\\nVERMA'
      * it glues them keeping the case change → 'JayVerma'
      * it glues them in all-caps             → 'RAHULGUPTA'
      * it drops the space after a middle initial → 'POOJA MPALANDE'
    Whitespace runs collapse to one space; a lowercase→uppercase boundary
    becomes a space (a genuine intra-word capital is vanishingly rare in
    Indian names); every all-caps token is then split on a trailing known
    surname — applied per token so a glued surname is recovered even when
    the rest of the name is already spaced."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)
    parts = []
    for tok in t.split(" "):
        if len(tok) >= 5 and tok.isalpha():
            tok = _split_glued_surname(tok)
        parts.append(tok)
    return " ".join(parts).title()


# ──────────────────────────────────────────────────────────────────────────────
# Region helpers
# ──────────────────────────────────────────────────────────────────────────────

def _order(regions: List[Dict]) -> List[Dict]:
    """Reading order — top-to-bottom, then left-to-right."""
    return sorted(regions, key=lambda r: (r["region"].bbox[1],
                                          r["region"].bbox[0]))


def _conf_for(regions: List[Dict], needle: str,
              default: float = 0.0) -> float:
    """Confidence of the region whose text contains `needle`."""
    if not needle:
        return default
    key = re.sub(r"\s+", "", needle).upper()
    for r in regions:
        if key and key in re.sub(r"\s+", "", r["text"]).upper():
            return r["conf"]
    return default


def _find_region_idx(regions: List[Dict], pattern: str) -> int:
    rx = re.compile(pattern, re.I)
    for i, r in enumerate(regions):
        if rx.search(r["text"]):
            return i
    return -1


def _yyyy_mm_dd(dmy: str) -> str:
    """Convert 'DD/MM/YYYY' (or with - / .) to 'YYYY-MM-DD'. Returns input on failure."""
    m = re.match(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", dmy)
    if not m:
        return dmy
    d, mo, y = m.groups()
    return f"{y}-{int(mo):02d}-{int(d):02d}"


def _age_from_dob(iso_dob: str) -> str:
    """Compute age in whole years from an ISO 'YYYY-MM-DD' date of birth.
    Returns '' when the input isn't a usable date."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", iso_dob or "")
    if not m:
        return ""
    y, mo, d = (int(g) for g in m.groups())
    today = date.today()
    age = today.year - y - ((today.month, today.day) < (mo, d))
    return str(age) if 0 <= age < 150 else ""


_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
     "nov", "dec"), 1)}


def _to_iso_date(s: str) -> str:
    """Normalise 'DD/MM/YYYY', 'DD-MM-YYYY' or 'DD-Mon-YYYY' (e.g.
    '09-Jun-2004', '09-August-2022') to ISO 'YYYY-MM-DD'. Returns '' when
    the string isn't a recognisable date."""
    # Numeric form: separators are required so a plain digit run (e.g. a
    # licence number) can't be mistaken for a date.
    m = re.match(r"\s*(\d{1,2})[/\-.\s]+(\d{1,2})[/\-.\s]+(\d{4})", s)
    if not m:
        # Month-name form: OCR routinely fuses day/month/year together
        # ('09JUN2004'), so the separators are optional here.
        m = re.match(r"\s*(\d{1,2})[/\-.\s]*([A-Za-z]{3,})[/\-.\s]*(\d{4})", s)
    if not m:
        return ""
    d, mo, y = m.groups()
    mon = int(mo) if mo.isdigit() else _MONTHS.get(mo[:3].lower(), 0)
    if not (1 <= mon <= 12) or not (1 <= int(d) <= 31):
        return ""
    return f"{y}-{mon:02d}-{int(d):02d}"


def _yymmdd_to_dmy(s: str, *, is_expiry: bool = False) -> str:
    """Convert MRZ 'YYMMDD' to 'DD/MM/YYYY'.

    The MRZ stores only a 2-digit year, so the century must be inferred:

      * expiry dates are always in the future for a usable passport, so a
        2-digit year maps unconditionally to the 2000s (yy '35' -> 2035).
      * birth dates use a pivot at 30 — '00'-'30' -> 2000s,
        '31'-'99' -> 1900s — matching Indian passports issued from the
        2000s onward.
    """
    if not re.fullmatch(r"\d{6}", s):
        return ""
    yy, mo, d = int(s[:2]), int(s[2:4]), int(s[4:6])
    if is_expiry:
        year = 2000 + yy
    else:
        year = 2000 + yy if yy <= 30 else 1900 + yy
    return f"{d:02d}/{mo:02d}/{year}"


# ──────────────────────────────────────────────────────────────────────────────
# PAN
# ──────────────────────────────────────────────────────────────────────────────

PAN_LABEL_TOKENS = re.compile(
    r"INCOME|TAX|DEPART|GOVT|GOVERNMENT|PERMANENT|ACCOUNT|NUMBER|NAME|"
    r"FATHER|PARENT|DATE|BIRTH|SIGNATURE|SAMPLE|IMMIHELP",
    re.I,
)


def _looks_like_name(text: str) -> bool:
    """PAN names print in ALL CAPS, so reject anything mixed-case — that
    filters OCR-garbage tokens like 'Hteak' that happen to fall between
    the real name and father-name lines."""
    t = text.strip()
    if len(t) < 3 or any(ch.isdigit() for ch in t):
        return False
    if PAN_LABEL_TOKENS.search(t):
        return False
    letters = [ch for ch in t if ch.isalpha()]
    if len(letters) < 3:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return upper_ratio > 0.85


def build_pan(regions: List[Dict], full_text: str) -> Dict[str, Any]:
    ordered = _order(regions)

    pan_match = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", full_text.upper())
    pan_num = pan_match.group() if pan_match else ""

    dob_match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", full_text)
    dob = dob_match.group(1) if dob_match else ""

    # Two name lines (Name, Father's Name) appear ABOVE the DOB in reading order.
    full_name = father_name = ""
    fn_conf = par_conf = 0.0
    dob_idx = next((i for i, r in enumerate(ordered) if dob and dob in r["text"]),
                   -1)
    if dob_idx > 0:
        names: List[Dict] = []
        for j in range(dob_idx - 1, -1, -1):
            if _looks_like_name(ordered[j]["text"]):
                names.append(ordered[j])
            if len(names) == 2:
                break
        # names is bottom→top; bottom one (closest to DOB) is father, top one is full
        if len(names) >= 1:
            father_name = _clean_name(names[0]["text"])
            par_conf = names[0]["conf"]
        if len(names) >= 2:
            full_name = _clean_name(names[1]["text"])
            fn_conf = names[1]["conf"]

    data = {
        "ocr_fields": [{
            "document_type": "pan",
            "pan_number": _field(pan_num, _conf_for(regions, pan_num)),
            "full_name": _field(full_name, fn_conf),
            "father_name": _field(father_name, par_conf),
            "dob": _field(dob, _conf_for(regions, dob)),
        }],
    }
    return _envelope(data)


# ──────────────────────────────────────────────────────────────────────────────
# Aadhaar
# ──────────────────────────────────────────────────────────────────────────────

# OCR routinely mangles the 'DOB' label — 'D0B', 'O0B', or even '008'
# (zero-zero-eight, seen in the wild) — so match any D/O/0 + O/0 + B/8
# spelling, plus the spelled-out 'Date of Birth'.
_DOB_LABEL_RX = re.compile(r"[D0O][O0][B8]|DATE\s*OF\s*BIRTH", re.I)
# An Issue / Print / Download / Update date is NEVER the date of birth. A card
# always prints one of these alongside the DOB, so they must be ruled out
# before any "first date wins" fallback.
_ISSUE_DATE_RX = re.compile(
    r"ISSUE|ISSUED|PRINT|DOWNLOAD|UPDATE|GENERAT|VALID", re.I)
_FULL_DATE_RX = re.compile(r"[0-3]?\d[/\-.][01]?\d[/\-.]\d{4}")


def _is_plausible_dob(iso: str) -> bool:
    """True if `iso` ('YYYY-MM-DD' or bare 'YYYY') is a usable birth date:
    real calendar values and a year between 1900 and today — a date of birth
    is never in the future, so an OCR-mangled or stray date is rejected."""
    m = re.match(r"(\d{4})(?:-(\d{2})-(\d{2}))?$", iso)
    if not m:
        return False
    if not (1900 <= int(m.group(1)) <= date.today().year):
        return False
    if m.group(2) is not None:
        mo, d = int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return False
    return True


def _dob_from_digits(chunk: str) -> str:
    """Recover a DOB whose day/month separator OCR dropped, e.g. '2102/2002'
    or '21022002' -> '2002-02-21'. Reads the first 8 digits as DDMMYYYY and
    returns ISO only when the result is a plausible birth date (so a random
    digit run can't masquerade as one)."""
    digits = re.sub(r"\D", "", chunk)
    if len(digits) < 8:
        return ""
    iso = f"{digits[4:8]}-{digits[2:4]}-{digits[0:2]}"
    return iso if _is_plausible_dob(iso) else ""


def _aadhaar_dob(full_text: str) -> Tuple[str, bool, str]:
    """Pick the date of birth from the card text.

    Returns ``(value, yob_only_flag, raw_matched_date)`` — ``value`` is
    ``YYYY-MM-DD`` (or just ``YYYY`` for a year-of-birth-only card) and
    ``raw_matched_date`` is the un-normalised match used for the confidence
    lookup.

    The hard part is NOT reading *a* date — it's reading the *right* one.
    Every Aadhaar prints an Issue/Print date too, and OCR frequently mangles
    the 'DOB' label into '008' / 'D0B' / 'O0B', so both a naive "first date
    wins" and a strict ``DOB:`` anchor land on the issue date. Strategy, in
    order of confidence:

      1. a date whose preceding characters carry a DOB-style label (and not an
         issue label);
      1b. a DOB-labelled date whose day/month separator OCR dropped
         ('D0B: 2102/2002');
      2. a DOB/YOB label followed by a bare 4-digit year (year-of-birth card);
      3. the EARLIEST remaining plausible date — a person is always born
         before their card is issued or printed, so even when the issue
         label is garbled beyond recognition the birth date is still the
         oldest date on the card.

    Every candidate must pass :func:`_is_plausible_dob`, so impossible dates
    (future years, month 13, …) can never surface."""
    dates = [(m.group(0), full_text[max(0, m.start() - 18):m.start()])
             for m in _FULL_DATE_RX.finditer(full_text)]

    # 1) a full date explicitly carrying a DOB-style label (not an issue label)
    for raw, ctx in dates:
        if _DOB_LABEL_RX.search(ctx) and not _ISSUE_DATE_RX.search(ctx):
            iso = _yyyy_mm_dd(raw)
            if _is_plausible_dob(iso):
                return iso, False, raw

    # 1b) a DOB-labelled date that lost its day/month separator in OCR
    #     ('D0B：2102/2002', '21022002'). Anchored to an explicit DOB label and
    #     gated by _is_plausible_dob so a stray digit run can't masquerade as a
    #     birth date. The fullwidth colon '：' is common on phone-photo OCR.
    #     Internal spaces are allowed ('21 / 02 / 2002') but newlines are not,
    #     so the run can't bleed into the Aadhaar number on the next line; only
    #     the first 8 digits are used, so trailing junk is harmless.
    loose = re.search(
        r"(?:[D0O][O0][B8]|DATE\s*OF\s*BIRTH)\s*[:：\-]?\s*(\d[\d/\-.．\t ]{5,14})",
        full_text, re.I)
    if loose:
        iso = _dob_from_digits(loose.group(1))
        if iso:
            return iso, False, loose.group(1)

    # 2) a DOB / YOB label followed by a bare year (year-of-birth-only card)
    ym = re.search(
        r"(?:[D0O][O0][B8]|DATE\s*OF\s*BIRTH|YEAR\s*OF\s*BIRTH|YOB)"
        r"[:\s]*((?:19|20)\d{2})\b", full_text, re.I)
    if ym and _is_plausible_dob(ym.group(1)):
        return ym.group(1), True, ym.group(1)

    # 3) the earliest plausible non-issue date
    plausible = [(_yyyy_mm_dd(raw), raw) for raw, ctx in dates
                 if not _ISSUE_DATE_RX.search(ctx)
                 and _is_plausible_dob(_yyyy_mm_dd(raw))]
    if plausible:
        iso, raw = min(plausible, key=lambda p: p[0])
        return iso, False, raw
    return "", False, ""


# Boilerplate/header tokens a name region must never be. Fragments (GOVE,
# IDENTI, AUTHORI, ADDR …) rather than whole words so OCR garbles still match
# — e.g. 'GOVEMMENT', 'IDENTINICATION', 'ADDRBSS', 'OFINDIA' all leaked
# through as "names" before.
_AADHAAR_LABELS = re.compile(
    r"DOB|YEAR|MALE|FEMALE|GOVE|GOVT|BHARAT|REPUBLIC|OFIND|INDIA|AADH|UIDAI|"
    r"AUTHORI|UNIQUE|IDENTI|ADDR|MOBILE|HELP|INFORMAT|SCANN|VERIFIED|ENROL|"
    r"PROOF|CITIZEN|DATE|BIRTH|ISSUE|PRINT|DOWNLOAD|MERA|PEHCHAN|XML|"
    r"\bCODE\b|GOV\.",
    re.I,
)


def _looks_like_aadhaar_name(text: str) -> bool:
    """A plausible Aadhaar name token. RapidOCR often runs words together
    ('JAYVERMA') so single tokens are accepted as long as they are mostly
    letters and don't hit a known label keyword."""
    t = text.strip()
    if not t or len(t) < 4 or any(ch.isdigit() for ch in t):
        return False
    if _AADHAAR_LABELS.search(t):
        return False
    letters = sum(ch.isalpha() for ch in t)
    return letters >= 4 and letters / len(t) > 0.8


# Same OCR-tolerant DOB label as _DOB_LABEL_RX, but it must be followed (after
# any separators, including the fullwidth colon '：') by a date-like digit run —
# 2 digits then another digit/separator. That requirement keeps a bare '008'
# token or the boilerplate 'date of birth' sentence from anchoring, while still
# matching a separator-mangled date like 'D0B：2102/2002'.
_DOB_ANCHOR_RX = re.compile(
    r"(?:[D0O][O0][B8]|DATE\s*OF\s*BIRTH|YEAR\s*OF\s*BIRTH|YOB)"
    r"[\s:：/.\-]*\d{2}[\d/\-.．]",
    re.I,
)


def _aadhaar_name(ordered: List[Dict]) -> Tuple[str, float]:
    """The cardholder name prints directly above the DOB line.

    Anchor on the region that actually carries 'DOB <date>' — not the
    boilerplate paragraph that merely mentions 'date of birth', and not a
    stray print/issue date — then take the first name-like region above
    it. Only if no real DOB line is found do we fall back to the first
    plausible name in reading order."""
    dob_idx = next((i for i, r in enumerate(ordered)
                    if _DOB_ANCHOR_RX.search(r["text"])), -1)
    if dob_idx > 0:
        for j in range(dob_idx - 1, -1, -1):
            if _looks_like_aadhaar_name(ordered[j]["text"]):
                return _clean_name(ordered[j]["text"]), ordered[j]["conf"]
    for r in ordered:
        if _looks_like_aadhaar_name(r["text"]):
            return _clean_name(r["text"]), r["conf"]
    return "", 0.0


_INDIAN_STATES = (
    "PUNJAB", "HARYANA", "DELHI", "MAHARASHTRA", "GUJARAT", "KARNATAKA",
    "TAMIL NADU", "KERALA", "RAJASTHAN", "UTTAR PRADESH", "WEST BENGAL",
    "ANDHRA PRADESH", "TELANGANA", "MADHYA PRADESH", "BIHAR", "ODISHA",
    "ASSAM", "JHARKHAND", "CHHATTISGARH", "UTTARAKHAND", "GOA", "TRIPURA",
    "MANIPUR", "MEGHALAYA", "MIZORAM", "NAGALAND", "SIKKIM",
    "HIMACHAL PRADESH", "JAMMU AND KASHMIR", "LADAKH", "CHANDIGARH",
    "PUDUCHERRY", "ANDAMAN AND NICOBAR",
)


_AADHAAR_ADDR_MARKERS = re.compile(
    r"C/O|S/O|D/O|W/O|HOUSE\s*NO|\bH\.?\s*NO\b|HNO|FLAT|FLOOR|SECTOR|"
    r"\bVTC\b|\bP\.?\s*O\b|SUB\s*DIST|DISTRICT|\bDIST\b|STATE|PIN\s*CODE|"
    r"\bPIN\b|VILLAGE|ROAD|STREET|NAGAR|COLONY|BLOCK|LANE|MARG|TEHSIL|"
    r"MANDAL|TALUK|GALI|CHOWK|AVENUE|\bAVE\b|APART?MENT|APPARTMENT|ENCLAVE|"
    r"PHASE|PLOT|MARKET|BAZ?AR|GANJ|PURA|BAGH|CROSS|\bEXTN?\b|TOWER|\bZONE\b",
    re.I,
)


def _is_aadhaar_addr_line(text: str) -> bool:
    """A line carrying Aadhaar address content — identified by the
    structured markers Aadhaar prints (C/O, House No, VTC, PO, District,
    State, PIN Code …), a 'CITY-PIN' chunk, or a 6-digit PIN."""
    t = text.strip()
    if not t:
        return False
    if _AADHAAR_ADDR_MARKERS.search(t):
        return True
    if re.search(r"[A-Za-z]{3,}\s*-\s*\d{6}\b", t):
        return True
    # A bare 6-digit PIN, alone or trailing an address line ('… BENGAL 713205').
    return bool(re.search(r"\b\d{6}\b", t))


# Address label, OCR-tolerant: 'ADDRESS', 'ADDRBSS', 'ADORESS' (the second
# letter is often misread, and the tail garbles) or the Hindi पता.
_ADDR_LABEL_RX = re.compile(r"A[D0O]{1,2}R[A-Z]*S{1,2}|पता", re.I)
# Footer / contact lines that print right under the address but are not part
# of it (helpline 1947, email, uidai.gov.in, VID, mobile, enrolment no.).
_ADDR_FOOTER_RX = re.compile(
    r"VID[:\s]|MOBILE|UIDAI|WWW\.|HELP@|@|ENROL|\b1947\b|GOV\.?\s*IN", re.I)


def _aadhaar_address(ordered: List[Dict], aadhaar_num: str = "") -> Tuple[
        str, float, str, str, str, Dict[str, str]]:
    """Return (address_string, conf, zip, care_of, care_of_relation, components).

    Two extraction strategies are computed and the better result wins:

      1. **Label-anchored span** — find the printed 'Address:' label (even when
         OCR mangles it to 'ADDRBSS' or glues the first line onto it) and walk
         downward, accumulating lines until the address clearly ends (an ID
         number, a footer line, or a 6-digit PIN). This handles the common
         real-world back-side photo where address lines carry no structured
         marker words.
      2. **Marker-based clusters** — gather lines that carry structured markers
         (C/O, House No, VTC, District, PIN …), cluster them by x-column (a
         multi-panel e-card prints the address once per panel) and keep the
         richest cluster, dropping the duplicate, lower-quality copies.

    A garbled image can make either strategy wander, so both are scored and we
    keep whichever captured a PIN (a complete address) and is longest — that
    way the label path wins on a clean back photo while the marker path still
    rescues a messy multi-panel e-card whose label landed on a bad panel."""
    digits_aadhaar = re.sub(r"\D", "", aadhaar_num)

    def _is_id_number(t: str) -> bool:
        """Aadhaar / VID numbers print near the address — never part of it."""
        d = re.sub(r"\D", "", t)
        return len(d) >= 11 or bool(digits_aadhaar and digits_aadhaar in d)

    results: List[Tuple] = []

    # ── strategy 1: label-anchored span ──
    label_idx = next((i for i, r in enumerate(ordered)
                      if _ADDR_LABEL_RX.search(r["text"])), -1)
    if label_idx >= 0:
        parts: List[str] = []
        confs: List[float] = []
        # The opening line is often glued onto the label ('ADDRBSS:2B/41,…') —
        # keep whatever follows the label on that same region.
        first = ordered[label_idx]["text"]
        m = _ADDR_LABEL_RX.search(first)
        tail = re.sub(r"^[\s:.\-)）]*", "", first[m.end():]).strip()
        if tail and not _is_id_number(tail):
            parts.append(tail)
            confs.append(ordered[label_idx]["conf"])
        for r in ordered[label_idx + 1:]:
            t = r["text"].strip()
            if not t:
                continue
            if _is_id_number(t) or _ADDR_FOOTER_RX.search(t):
                break
            parts.append(t)
            confs.append(r["conf"])
            if re.search(r"\b\d{6}\b", t):     # a PIN terminates the address
                break
        if parts:
            results.append(_assemble_address(parts, confs, aadhaar_num))

    # ── strategy 2: marker-based clusters ──
    candidates: List[Dict] = []
    for r in ordered:
        t = r["text"].strip()
        if not t or _is_id_number(t) or _ADDR_FOOTER_RX.search(t):
            continue
        if _is_aadhaar_addr_line(t):
            candidates.append(r)
    if candidates:
        # cluster by x-column — a multi-panel e-card prints the address once
        # per panel, each panel being its own x-band.
        clusters: List[List[Dict]] = []
        for r in sorted(candidates, key=lambda c: c["region"].bbox[0]):
            x = r["region"].bbox[0]
            if clusters and x - clusters[-1][-1]["region"].bbox[0] <= 120:
                clusters[-1].append(r)
            else:
                clusters.append([r])
        best = max(clusters, key=lambda cl: (
            sum(_is_aadhaar_addr_line(x["text"]) for x in cl), len(cl)))
        best.sort(key=lambda r: r["region"].bbox[1])
        results.append(_assemble_address([r["text"].strip() for r in best],
                                         [r["conf"] for r in best], aadhaar_num))

    if not results:
        return "", 0.0, "", "", "father", {}

    # Prefer the result that captured a PIN (index 2 of the tuple) — a strong
    # signal of a complete address — then the longest address string.
    return max(results, key=lambda res: (1 if res[2] else 0, len(res[0])))


def _assemble_address(parts: List[str], confs: List[float],
                      aadhaar_num: str = "") -> Tuple[
        str, float, str, str, str, Dict[str, str]]:
    """Join ordered address lines into the address string + sub-components."""
    full = ", ".join(parts).strip(" ,")
    full = re.sub(r"(?:\s*,\s*)+", ", ", full).strip(" ,")
    full = re.sub(r"\s+", " ", full)

    zipm = re.search(r"\b(\d{6})\b", full)
    pin = zipm.group(1) if zipm else ""

    # care_of: relation depends on which prefix appeared (C/O vs S/O vs W/O).
    co_match = re.search(
        r"(C/O|S/O|D/O|W/O)[:\s]*([A-Za-z][A-Za-z\s]+?)(?:,|House|$)",
        full, re.I,
    )
    care_of = ""
    care_of_relation = "father"      # sensible default for empty case
    if co_match:
        care_of = _clean_name(co_match.group(2))
        prefix = co_match.group(1).upper()
        care_of_relation = {
            "C/O": "care_of", "S/O": "father",
            "D/O": "father",  "W/O": "husband",
        }.get(prefix, "care_of")

    # ── sub-components — best-effort ──
    components: Dict[str, str] = {
        "first_line": "", "second_line": "", "locality": "", "landmark": "",
        "house_number": "", "district": "", "city": "", "state": "",
        "country": "", "zip": pin,
    }
    up = full.upper()
    if "INDIA" in up:
        components["country"] = "INDIA"
    for state in _INDIAN_STATES:
        if state in up:
            components["state"] = state.title()
            break

    # House number: the value printed against a 'House No' label, else the
    # leading numeric token of the first non-care-of segment ('B-1203').
    hn = re.search(r"(?:HOUSE\s*NO|H\.?\s*NO|HNO|FLAT)[.:\s]*"
                   r"(\d{1,5}(?:/\d{1,4})?)", full, re.I)
    if not hn:
        first_seg = next((p for p in parts
                          if not re.match(r"C/O|S/O|D/O|W/O", p, re.I)), "")
        hn = re.search(r"\b([A-Za-z]?-?\d{1,5}(?:/\d{1,4})?)\b", first_seg)
    if hn:
        components["house_number"] = re.sub(r"^[A-Za-z]-?", "", hn.group(1))

    # District / city from their labelled segments where present.
    dm = re.search(r"\bDISTRICT[.:\s]+([A-Za-z][A-Za-z\s]*?)\s*[,;]",
                   full, re.I)
    if dm:
        components["district"] = _clean_name(dm.group(1))
    vm = re.search(r"\bVTC[.:\s]*([A-Za-z][A-Za-z\s]*?)\s*[,;]", full, re.I)
    if vm:
        components["city"] = _clean_name(vm.group(1))

    # NOTE: city / district are filled ONLY from their explicit labels (VTC /
    # District). The old "guess the last two comma tokens" heuristic is gone —
    # on a garbled address it produced confidently-wrong values (city="Avenue")
    # and a wrong structured field is worse than an empty one. The full address
    # string is always returned; downstream can re-parse if it needs more.

    conf = sum(confs) / len(confs) if confs else 0.0
    return full, conf, pin, care_of, care_of_relation, components


def build_aadhaar(regions: List[Dict], full_text: str) -> Dict[str, Any]:
    ordered = _order(regions)

    # ── front-side signals (identity) ──
    num_match = re.search(r"(?<!\d)(\d{4}\s?\d{4}\s?\d{4})(?!\d)", full_text)
    aadhaar_num = re.sub(r"\s+", "", num_match.group(1)) if num_match else ""
    a_conf = _conf_for(regions, aadhaar_num)

    # _aadhaar_dob returns the raw matched date so the confidence lookup lands
    # on the real birth-date region, not a stray print/issue date.
    dob_val, yob_only, dob_raw = _aadhaar_dob(full_text)
    dob_conf = _conf_for(regions, dob_raw) if dob_raw else 0.0

    gender = ""
    g_conf = 0.0
    gm = re.search(r"\b(MALE|FEMALE|TRANSGENDER)\b", full_text, re.I)
    if gm:
        gender = {"male": "M", "female": "F",
                  "transgender": "T"}[gm.group(1).lower()]
        g_conf = _conf_for(regions, gm.group(1))
    else:
        # Bilingual cards print the Hindi gender too; fall back to it when the
        # English word was missed. (Devanagari is case-less, so the upper-cased
        # full_text still contains it.)
        for pat, code in ((r"पुरुष", "M"), (r"महिला|स्त्री", "F"),
                          (r"ट्रांसजेंडर|किन्नर", "T")):
            hm = re.search(pat, full_text)
            if hm:
                gender = code
                g_conf = _conf_for(regions, hm.group(0))
                break

    full_name, name_conf = _aadhaar_name(ordered)

    # ── back-side signals (address) ──
    addr, addr_conf, pin, care_of, co_relation, addr_comps = \
        _aadhaar_address(ordered, aadhaar_num)

    # Detect which side(s) the image actually shows. DOB / gender belong
    # to the front; address belongs to the back. uniqueness_id is only
    # meaningful when we actually have the Aadhaar number.
    has_front = bool(dob_val or gender)
    has_back = bool(addr)

    uniq = (hashlib.sha256(aadhaar_num.encode()).hexdigest()
            if aadhaar_num else "")

    front_obj = {
        "document_type": "aadhaar_front_bottom",
        "full_name": _field(full_name, name_conf),
        "gender": _field(gender, g_conf),
        "mother_name": _field("", 0.0),
        "father_name": _field("", 0.0),
        "dob": {"value": dob_val, "confidence": _conf(dob_conf),
                "yob": yob_only},
        "aadhaar_number": {"value": aadhaar_num, "confidence": _conf(a_conf),
                           "is_masked": False, "input_validation": False},
        "image_url": None,
        "uniqueness_id": uniq,
    }
    back_obj = {
        "document_type": "aadhaar_back",
        "address": {"value": addr, "confidence": _conf(addr_conf),
                    **addr_comps},
        "zip": _field(pin, addr_conf),
        "care_of": {"value": care_of, "confidence": _conf(addr_conf),
                    "relation": co_relation},
        "aadhaar_number": {"value": aadhaar_num, "confidence": _conf(a_conf),
                           "is_masked": False, "input_validation": False},
        "image_url": None,
    }

    fields: List[Dict[str, Any]] = []
    if has_front:
        fields.append(front_obj)
    if has_back:
        fields.append(back_obj)
    if not fields:
        fields.append(front_obj)         # fallback so the shape stays stable

    return _envelope({"ocr_fields": fields})


# ──────────────────────────────────────────────────────────────────────────────
# Passport
# ──────────────────────────────────────────────────────────────────────────────

def _mrz_lines(full_text: str) -> Tuple[str, str]:
    """Extract MRZ line 1 (P<...) and line 2 (passport_num<...)."""
    txt = full_text.upper()
    nospace_per_line = [re.sub(r"\s+", "", ln) for ln in txt.split("\n") if ln.strip()]
    # Line 1 starts with the document-type letter ('P' for passport) and
    # carries the '<<' name separator. On a real passport the 3rd char is
    # the first letter of the country code ('P<IND...') — filler '<' only
    # appears on specimens — so don't pin the 3rd char to '<'.
    line1 = next((ln for ln in nospace_per_line
                  if re.match(r"P[A-Z<]", ln) and "<<" in ln
                  and len(ln) >= 28), "")
    # Line 2 begins with the passport number: 1-2 letters then 6-7 digits
    # (Indian passports use both 'A1234567' and 'AH386374' forms).
    line2 = next((ln for ln in nospace_per_line
                  if re.match(r"[A-Z]{1,2}\d{6,7}", ln)
                  and len(ln) >= 28), "")
    return line1, line2


def _split_mrz_line1(line1: str) -> Tuple[str, str, str, str]:
    """Return (type, country_code, surname, given). Handles the case where
    the country-code positions are filler `<<<` (e.g. on specimen passports)
    by skipping leading `<` chars after the type."""
    if not line1 or len(line1) < 5:
        return "", "", "", ""
    type_ = line1[0]
    tail = line1[2:]
    country = ""
    if len(tail) >= 3 and tail[:3].isalpha():
        country, tail = tail[:3], tail[3:]
    else:
        tail = tail.lstrip("<")
    parts = [p for p in tail.split("<<") if p]
    surname = _clean_name(parts[0].replace("<", " ")) if parts else ""
    given = (_clean_name(parts[1].replace("<", " "))
             if len(parts) > 1 else "")
    return type_, country, surname, given


def _split_mrz_line2(line2: str) -> Tuple[str, str, str, str, str]:
    """Return (passport_num, country, dob_yymmdd, sex, doe_yymmdd).
    Tolerant of 0-2 check digits between fields."""
    if not line2:
        return "", "", "", "", ""
    pm = re.match(r"([A-Z]{1,2}\d{6,7})", line2)
    pnum = pm.group(1) if pm else ""
    m = re.search(r"([A-Z]{3})(\d{6})\d?([MF<])(\d{6})", line2)
    if not m:
        return pnum, "", "", "", ""
    return (pnum,) + m.groups()


_BACK_LABEL_RX = re.compile(
    r"NAME\s*OF|ADDRESS|LEGAL\s*GUARDIAN|OLD\s*PASSPORT|FILE\s*NO|"
    r"FATHER|MOTHER|SPOUSE|REPUBLIC|GIVEN\s*NAME|DATE\s*OF|PLACE\s*OF|"
    r"PASSPORT",
    re.I,
)


def _label_value_before(ordered: List[Dict], label_pattern: str
                        ) -> Tuple[str, float]:
    """Passport-back layout prints each name ABOVE its label
    ('SUKESH VERMA' / 'Name of Father'). Only the IMMEDIATELY adjacent
    non-empty line is the value — don't walk past another label, or we'd
    borrow the previous field's value (e.g. give 'Name of Spouse' the
    mother's name)."""
    rx = re.compile(label_pattern, re.I)
    for i, r in enumerate(ordered):
        if rx.search(r["text"]):
            for j in range(i - 1, -1, -1):
                t = ordered[j]["text"].strip()
                if not t:
                    continue
                if _BACK_LABEL_RX.search(t) or any(ch.isdigit() for ch in t):
                    return "", 0.0
                return _clean_name(t), ordered[j]["conf"]
            break
    return "", 0.0


def _label_value_after(ordered: List[Dict], label_pattern: str
                       ) -> Tuple[str, float]:
    """Address / File No / Old Passport: value follows the label.
    Only the immediately adjacent non-empty line is taken."""
    rx = re.compile(label_pattern, re.I)
    for i, r in enumerate(ordered):
        if rx.search(r["text"]):
            for j in range(i + 1, len(ordered)):
                t = ordered[j]["text"].strip()
                if not t:
                    continue
                if _BACK_LABEL_RX.search(t):
                    return "", 0.0
                return t.strip(), ordered[j]["conf"]
            break
    return "", 0.0


def _passport_back_address(ordered: List[Dict]
                           ) -> Tuple[str, float, Dict[str, str]]:
    """Collect every region between the 'Address' label and the
    'Old Passport' / 'File No' markers; parse out city / state / zip.
    Stopper includes 'PASSP' so OCR garbage like 'LDPASSPOWDATANDPLACEFSE'
    (the mangled 'Old Passport No with Date and Place of Issue' label)
    is correctly recognised as the end of the address block."""
    start = next((i for i, r in enumerate(ordered)
                  if re.search(r"ADDRESS", r["text"], re.I)), -1)
    if start < 0:
        return "", 0.0, {}

    stoppers = re.compile(r"OLD\s*PASSPORT|FILE\s*NO|PASSP|FATHER|MOTHER|"
                          r"SPOUSE|LEGAL\s*GUARDIAN", re.I)
    parts, confs = [], []
    for r in ordered[start + 1:]:
        t = r["text"].strip()
        if not t:
            continue
        if stoppers.search(t):
            break
        parts.append(t)
        confs.append(r["conf"])

    full = ", ".join(parts).strip(" ,")
    components: Dict[str, str] = {"house_number": "", "locality": "",
                                  "city": "", "district": "", "state": "",
                                  "country": "", "zip": ""}
    if not full:
        return "", 0.0, components

    pin = re.search(r"PIN[:\s]*([0-9]{6})|\b([0-9]{6})\b", full)
    components["zip"] = (pin.group(1) or pin.group(2)) if pin else ""
    if "INDIA" in full.upper():
        components["country"] = "INDIA"
    for state in ("PUNJAB", "HARYANA", "DELHI", "MAHARASHTRA", "GUJARAT",
                  "KARNATAKA", "TAMIL NADU", "KERALA", "RAJASTHAN",
                  "UTTAR PRADESH", "WEST BENGAL", "ANDHRA PRADESH",
                  "TELANGANA", "MADHYA PRADESH", "BIHAR", "ODISHA",
                  "ASSAM", "JHARKHAND", "CHHATTISGARH", "UTTARAKHAND"):
        if state in full.upper():
            components["state"] = state
            break

    return full, sum(confs) / len(confs), components


def _looks_like_passport_name(text: str) -> bool:
    """ALL-CAPS person-name line on the passport back."""
    t = text.strip()
    if len(t) < 4 or any(ch.isdigit() for ch in t):
        return False
    if re.search(r"NAME\s*OF|ADDRESS|GUARDIAN|FATHER|MOTHER|SPOUSE|"
                 r"FILE\s*NO|OLD\s*PASSPORT|PASSP|REPUBLIC|INDIA|"
                 r"PUNJAB|HARYANA|DELHI|MAHARASHTRA|BLOCK|AEROCITY|"
                 r"NAGAR|MOHALI|PIN", t, re.I):
        return False
    letters = [ch for ch in t if ch.isalpha()]
    if len(letters) < 4:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return upper_ratio > 0.85


def _passport_back_names(ordered: List[Dict]
                          ) -> Tuple[Tuple[str, float],
                                     Tuple[str, float],
                                     Tuple[str, float]]:
    """Father / Mother / Spouse names sit ABOVE the Address block in a
    fixed printed order on the Indian passport. The labels above each
    name are often dropped or garbled by OCR — so we don't anchor on
    them. Take the first 1-3 ALL-CAPS name-like lines that appear
    before the 'Address' marker."""
    addr_idx = next((i for i, r in enumerate(ordered)
                     if re.search(r"ADDRESS", r["text"], re.I)),
                    len(ordered))
    picks: List[Tuple[str, float]] = []
    for r in ordered[:addr_idx]:
        if _looks_like_passport_name(r["text"]):
            picks.append((_clean_name(r["text"]), r["conf"]))
        if len(picks) == 3:
            break
    while len(picks) < 3:
        picks.append(("", 0.0))
    return picks[0], picks[1], picks[2]


def _build_passport_back(regions: List[Dict],
                         full_text: str) -> Dict[str, Any]:
    ordered = _order(regions)
    up = full_text.upper()

    fn = re.search(r"\b([A-Z]{2}\d{13,14})\b", up)
    file_no = fn.group(1) if fn else ""
    f_conf = _conf_for(regions, file_no)

    # The barcode print under the passport number gives us the passport ID.
    pn = re.search(r"\b([A-Z]\d{7}|[A-Z]{2}\d{6,7})\b", up)
    passport_num = pn.group(1) if pn else ""
    p_conf = _conf_for(regions, passport_num)

    (father, fa_conf), (mother, mo_conf), (spouse, sp_conf) = \
        _passport_back_names(ordered)
    old_pass, op_conf = _label_value_after(ordered, r"OLD\s*PASSPORT")

    addr_text, addr_conf, comps = _passport_back_address(ordered)

    pin = comps.get("zip", "")
    data = {
        "ocr_fields": [{
            "document_type": "passport_back",
            "address": {"value": addr_text, "confidence": _conf(addr_conf)},
            "father": _field(father, fa_conf),
            "mother": _field(mother, mo_conf),
            "file_num": _field(file_no, f_conf),
            "old_doi": _field("", 0.0),
            "old_passport_num": _field(old_pass, op_conf),
            "old_place_of_issue": _field("", 0.0),
            "pin": _field(pin, addr_conf if pin else 0.0),
            "spouse": _field(spouse, sp_conf),
        }],
    }
    return _envelope(data)


def build_passport(regions: List[Dict], full_text: str) -> Dict[str, Any]:
    # Back side has no MRZ; identify it by labels unique to the back.
    nospace = re.sub(r"\s+", "", full_text.upper())
    back_hits = sum(k in nospace for k in (
        "NAMEOFFATHER", "NAMEOFMOTHER", "NAMEOFSPOUSE",
        "OLDPASSPORT", "FILENO", "LEGALGUARDIAN",
    ))
    if back_hits >= 2:
        return _build_passport_back(regions, full_text)

    line1, line2 = _mrz_lines(full_text)
    type_of, country_l1, surname, given = _split_mrz_line1(line1)
    passport_num, country_l2, dob_mrz, gender_mrz, doe_mrz = \
        _split_mrz_line2(line2)
    country_code = country_l2 or country_l1
    dob = _yymmdd_to_dmy(dob_mrz)
    doe = _yymmdd_to_dmy(doe_mrz, is_expiry=True)

    # Fallback to labelled fields when the MRZ second line is malformed.
    if not dob:
        m = re.search(r"(?:DATE\s*OF\s*BIRTH|BIRTH)[:\s]*"
                      r"([0-3]?\d[/\-.][01]?\d[/\-.]\d{4})",
                      full_text, re.I)
        if m:
            dob = m.group(1)
    if not dob:
        # DOB is a required field — when both the MRZ and the explicit
        # label are unreadable, fall back to the earliest date printed on
        # the page. Date of issue / expiry always fall after the birth
        # date, so the smallest year is the DOB.
        dated = [(d, int(re.search(r"\d{4}", d).group()))
                 for d in re.findall(
                     r"\b[0-3]?\d[/\-.][01]?\d[/\-.]\d{4}\b", full_text)]
        if dated:
            dob = min(dated, key=lambda x: x[1])[0]
    if not doe:
        m = re.search(r"DATE\s*OF\s*EXPIRY[:\s]*([0-3]?\d[/\-.][01]?\d[/\-.]\d{4})",
                      full_text, re.I)
        if m:
            doe = m.group(1)
    if not gender_mrz:
        m = re.search(r"\b(MALE|FEMALE)\b", full_text, re.I)
        if m:
            gender_mrz = "M" if m.group(1).upper() == "MALE" else "F"

    # ── from labeled lines ──
    nationality = "INDIAN" if country_code == "IND" else ""
    doi_m = re.search(r"DATE\s*OF\s*ISSUE[:\s]*([0-3]?\d[/\-.][01]?\d[/\-.]\d{4})",
                      full_text, re.I)
    doi = doi_m.group(1) if doi_m else ""
    pob_m = re.search(r"PLACE\s*OF\s*BIRTH[:\s]*([A-Z, ]+)", full_text, re.I)
    pob = pob_m.group(1).strip().rstrip(",").upper() if pob_m else ""
    poi_m = re.search(r"PLACE\s*OF\s*ISSUE[:\s]*([A-Z, ]+)", full_text, re.I)
    poi = poi_m.group(1).strip().rstrip(",").upper() if poi_m else ""

    # Fallback for passport number from a token printed in the body —
    # 1-2 letters + 6-7 digits, matching the Indian passport formats.
    if not passport_num:
        bm = re.search(r"\b([A-Z]{1,2}\d{6,7})\b", full_text)
        passport_num = bm.group(1) if bm else ""

    base_conf = _conf_for(regions, passport_num,
                          default=_conf_for(regions, line1[:8]) if line1 else 0.0)
    g = {"M": "M", "F": "F"}.get(gender_mrz, "")

    data = {
        "ocr_fields": [{
            "document_type": "passport_front",
            "country_code": _field(country_code, base_conf),
            "dob": _field(dob, base_conf),
            "doe": _field(doe, base_conf),
            "doi": _field(doi, _conf_for(regions, doi)),
            "gender": _field(g, base_conf),
            "given_name": _field(given, base_conf),
            "nationality": _field(nationality, base_conf),
            "passport_num": _field(passport_num, base_conf),
            "place_of_birth": _field(pob, _conf_for(regions, pob)),
            "place_of_issue": _field(poi, _conf_for(regions, poi)),
            "surname": _field(surname, base_conf),
            "mrz_line_1": _field(line1, base_conf),
            "mrz_line_2": _field(line2, base_conf),
            "type_of_passport": _field(type_of, base_conf),
            "passport_validity": None,
        }],
    }
    return _envelope(data)


# ──────────────────────────────────────────────────────────────────────────────
# Driving License
# ──────────────────────────────────────────────────────────────────────────────

def _license_dob(full_text: str, regions: List[Dict]) -> Tuple[str, float]:
    """Return (YYYY-MM-DD, confidence) for a Driving Licence's DOB.

    DOB is a required field. A DL prints it alongside the issue and
    validity dates — in numeric (09-06-2004) or month-name (09-Jun-2004)
    form — so prefer a label-anchored match; failing that, the earliest
    date on the card is the DOB, but only after issue / validity dates
    have been excluded so the issue date can't be mistaken for the DOB."""
    # DD<sep>MM-or-Mon<sep>YYYY. Numeric months need real separators (so a
    # licence number can't match); month names tolerate the missing
    # separators OCR leaves behind ('09JUN2004', '08JUN-2044').
    date_rx = (r"(?:[0-3]?\d[/\-. ]+[01]?\d[/\-. ]+\d{4}"
               r"|[0-3]?\d[/\-. ]*[A-Za-z]{3,}[/\-. ]*\d{4})")
    m = re.search(rf"(?:DOB|DATE\s*OF\s*BIRTH|BIRTH)[:\s]*({date_rx})",
                  full_text, re.I)
    if m:
        iso = _to_iso_date(m.group(1))
        if iso:
            return iso, _conf_for(regions, m.group(1))

    # Earliest *birth-eligible* date = DOB; ISO strings sort chronologically.
    # A date is skipped when an issue / validity / expiry label sits just
    # before it — those dates are not the DOB and would otherwise win when
    # the DOB itself was missed by OCR.
    issue_rx = re.compile(r"(?:ISSUE|VALID|EXPIR|DOI\b|TILL)", re.I)
    dated = []
    for mm in re.finditer(date_rx, full_text):
        iso = _to_iso_date(mm.group(0))
        if not iso:
            continue
        if issue_rx.search(full_text[max(0, mm.start() - 30):mm.start()]):
            continue
        dated.append((iso, mm.group(0)))
    if dated:
        dated.sort()
        iso, raw = dated[0]
        return iso, _conf_for(regions, raw)
    return "", 0.0


# Regions whose text marks the *end* of the address block — anything from
# the signature / authority / class panels onward is not part of the
# printed address and must terminate the line-gathering walk.
_DL_ADDR_STOP = re.compile(
    r"SIGNATURE|ISSUING|ISSUE|AADHAAR|AUTH|HOLDER|VALID|THUMB|IMPRESSION|"
    r"DATE\s*OF|BADGE|\bBG\b|BLOOD|\bDOB\b|FORM|RULE|\bCOV\b|\bLMV\b|"
    r"\bMCWG\b|DRIVE|UNION|STATE\s*MOTOR",
    re.I,
)


def _clean_swd(value: str) -> str:
    """Normalise the Son/Wife/Daughter-of name.

    The S/W/D label is routinely glued to its value in the OCR output
    ('S/DWOfMILINDPALANDE', 'S/DW OfRANGNATH'); `_dl_pair` strips the
    leading 'S/D' token but a relation remnant ('W Of', 'MW Of', 'Of')
    can survive. Only a trailing 'Of' anchors the strip, so a genuine
    name that merely starts with D/W/M (e.g. 'WILLIAM') is left intact."""
    t = value.strip(" :./")
    t = re.sub(r"^(?:[DWM]\s*)*O[Ff](?=[A-Z]|\s|$)\s*", "", t)
    return _clean_name(t)


# Right-column regions that are never an identity *value* — labels and
# boilerplate that must not be mistaken for a name / address under skew.
_DL_VALUE_NOISE = re.compile(
    r"DLNUMBER|\bDL\s*NO|INV|CARR|VALIDITY|\bVALID\b|FORM|RULE|AUTHORIS|"
    r"AADHAAR|ISSUING|AUTHORITY|HOLDER|SIGNATURE|\bAUTH\b|DRIVE|\bCOV\b|"
    r"\bLMV\b|\bMCWG\b|UNION|STATE\s*MOTOR|BLOOD|\bDOB\b|\bBG\b|\bNAME\b|"
    r"DATE\s*OF|ISSUE|THUMB|IMPRESSION|BADGE",
    re.I,
)


def _dl_is_name_value(t: str) -> bool:
    """Accept a holder / guardian name value: letters-only, no digits, not
    a blood-group token, and not a printed label / boilerplate phrase.
    Rejects a date ('09-Jun-2004'), 'B+', or 'Inv Carr No.' that a skewed
    value column might otherwise place on the name row."""
    t = t.strip()
    if any(ch.isdigit() for ch in t):
        return False
    if re.fullmatch(r"(?:AB|A|B|O)\s*[+\-]?\s*(?:VE)?", t, re.I):
        return False
    if _DL_VALUE_NOISE.search(t):
        return False
    return len(re.sub(r"[^A-Za-z]", "", t)) >= 3


def _dl_is_blood_value(t: str) -> bool:
    """Accept a blood-group value: 'B+', 'AB-', 'O+ve', or 'Not'/'Nil'."""
    t = t.strip()
    return bool(re.match(r"(?:AB|A|B|O)\s*[+\-]", t, re.I)
                or re.match(r"NOT|NIL", t, re.I))


def _dl_is_addr_value(t: str) -> bool:
    """Accept an address line: not a label / signature marker, not a bare
    date / number row, not a stray blood-group token. ('Issue Date',
    '09-Aug-2022' and 'B+' are all rejected.)"""
    t = t.strip()
    if not t or _DL_ADDR_STOP.search(t):
        return False
    if re.fullmatch(r"[\d/\-.\s]+", t) or _to_iso_date(t):
        return False
    if _dl_is_blood_value(t):
        return False
    # a real address line carries a meaningful run of letters or a clear
    # house/door number — reject a lone glyph the skewed column nudged in.
    return len(re.sub(r"[^A-Za-z]", "", t)) >= 4 or bool(re.search(r"\d{3}", t))


def _dl_pair(regions: List[Dict], label_rx: str, value_clean=None,
             accept=None) -> Tuple[str, float, Optional[Dict]]:
    """Resolve a DL 'label → value' pair, returning (value, conf, region).

    A driving licence prints each label in a left column with its value
    either glued onto the same OCR region ('DOB:12-05-1986',
    'S/DWOfMILIND', 'Issue Date 09-Aug-2022') or sitting in a separate
    region to its right on the same row ('NAME' → 'JAY VERMA',
    'Blood Grp' → 'B+'). Both layouts are handled.

    When the value sits in its own region we choose the region whose
    vertical centre is *closest* to the label's — not the leftmost on a
    band — because a photographed card is rarely perfectly deskewed and a
    tall band would otherwise grab the row below. An optional `accept`
    predicate filters out values of the wrong type (a date / blood group
    landing on the name row), which makes the pairing robust to skew.
    `region` is the region carrying the value (the label region when
    glued) so callers can anchor further walks."""
    rx = re.compile(label_rx, re.I)
    label = None
    for r in regions:
        m = rx.search(r["text"])
        if not m:
            continue
        label = r
        tail = r["text"][m.end():].strip(" \t:.-/")
        if tail and any(ch.isalnum() for ch in tail):
            return (value_clean(tail) if value_clean else tail), r["conf"], r
        break
    if label is None:
        return "", 0.0, None

    lb = label["region"].bbox
    lcx = (lb[0] + lb[2]) / 2.0
    lcy = (lb[1] + lb[3]) / 2.0
    lh = max(lb[3] - lb[1], 1)
    band = max(lh * 1.6, 50)          # vertical tolerance, skew-friendly
    best, best_dy = None, None
    for o in regions:
        if o is label:
            continue
        ob = o["region"].bbox
        ocx = (ob[0] + ob[2]) / 2.0
        ocy = (ob[1] + ob[3]) / 2.0
        if ocx <= lcx:                # value sits in the right-hand column
            continue
        dy = abs(ocy - lcy)
        if dy > band:
            continue
        if accept is not None and not accept(o["text"].strip()):
            continue
        if best is None or dy < best_dy:
            best, best_dy = o, dy
    if best is not None:
        v = best["text"].strip()
        return (value_clean(v) if value_clean else v), best["conf"], best
    return "", label["conf"], label


def _dl_find_label(regions: List[Dict], label_rx: str) -> Optional[Dict]:
    rx = re.compile(label_rx, re.I)
    for r in regions:
        if rx.search(r["text"]):
            return r
    return None


def _license_names(regions: List[Dict]) -> Tuple[str, float, str, float]:
    """Resolve (name, name_conf, swd, swd_conf) by ordinal column order.

    On a DL the holder name is always printed *above* the Son/Wife/
    Daughter-of name, in a value column to the right of the labels. Picking
    each by nearest row fails on a skewed photo — the two name rows are
    close enough that the guardian row can win the holder's label. Vertical
    *ordering* within a column survives any rotation, so we instead take the
    first name-like value at/below the NAME label as the holder, then the
    next name-like value below it as the guardian. The glued single-region
    layout some states print ('S/DWOfMILIND') is honoured first."""
    name_lbl = _dl_find_label(regions, r"\bNAME\b")
    swd_lbl = _dl_find_label(
        regions, r"\bS[WD]?\s*/\s*[DWMO]|\b[WD]\s*/\s*O\b")

    def _glued(label, cleaner, rx):
        if label is None:
            return "", 0.0
        m = re.search(rx, label["text"], re.I)
        tail = label["text"][m.end():].strip(" \t:.-/") if m else ""
        if tail and any(c.isalnum() for c in tail):
            return cleaner(tail), label["conf"]
        return "", 0.0

    name_v, name_c = _glued(name_lbl, _clean_name, r"\bNAME\b")
    swd_v, swd_c = _glued(
        swd_lbl, _clean_swd, r"\bS[WD]?\s*/\s*[DWMO]|\b[WD]\s*/\s*O\b")

    def _cx(r):
        b = r["region"].bbox
        return (b[0] + b[2]) / 2.0

    def _cy(r):
        b = r["region"].bbox
        return (b[1] + b[3]) / 2.0

    ref = name_lbl or swd_lbl
    cands: List[Dict] = []
    if ref is not None:
        rcx = _cx(ref)
        cands = sorted(
            (o for o in regions
             if o not in (name_lbl, swd_lbl)
             and _cx(o) > rcx
             and _dl_is_name_value(o["text"].strip())),
            key=_cy)

    name_y = None
    if not name_v and name_lbl is not None and cands:
        # The holder name is the topmost name-like value in the column;
        # boilerplate above it is already filtered out, so a generous
        # row-height floor tolerates the value sitting high or low under
        # an imperfect deskew (skew of either sign).
        nb = name_lbl["region"].bbox
        floor = _cy(name_lbl) - max((nb[3] - nb[1]) * 1.3, 36)
        pick = next((o for o in cands if _cy(o) >= floor), None)
        if pick is not None:
            name_v, name_c = _clean_name(pick["text"].strip()), pick["conf"]
            name_y = _cy(pick)

    if not swd_v and cands:
        # The guardian name is the next name-like value below the holder.
        if name_y is not None:
            base = name_y
        elif swd_lbl is not None:
            sb = swd_lbl["region"].bbox
            base = _cy(swd_lbl) - max((sb[3] - sb[1]) * 1.3, 36)
        else:
            base = -1
        pick = next((o for o in cands if _cy(o) > base + 6), None)
        if pick is not None:
            swd_v, swd_c = _clean_swd(pick["text"].strip()), pick["conf"]

    return name_v, name_c, swd_v, swd_c


def _license_address(regions: List[Dict]) -> Tuple[str, float]:
    """Assemble the multi-line printed address.

    Anchored on the 'Address'/'Add' label: the value column is taken from
    the address-valid region nearest the label, and every address-valid
    line in that column — from the label's row downward — is gathered in
    reading order, stopping at the first signature / issue / class marker
    or a large vertical gap. Working off the label's row (not a single
    matched value) means the opening line is kept even when an imperfect
    deskew nudges the column up relative to the label."""
    first, conf, anchor = _dl_pair(regions, r"\bADDRESS\b|\bADD",
                                   accept=_dl_is_addr_value)
    label = _dl_find_label(regions, r"\bADDRESS\b|\bADD")
    if anchor is None or label is None:
        return (first, conf) if first else ("", 0.0)

    ax = anchor["region"].bbox[0]
    lb = label["region"].bbox
    lh = max(lb[3] - lb[1], 1)

    # Two layouts: the value sits in a separate right-hand column
    # (HR-style — the opening line may be skew-shifted above the label, so
    # extend the top upward) or it is glued into the label's own left
    # column (MH-style — the name / S-W-D rows sit just above, so the block
    # must start at the label row and the glued opening line is prepended).
    separate = ax > lb[2] - 8
    if separate:
        top = lb[1] - max(lh * 1.5, 45)
        lines, confs = [], []
    else:
        top = lb[1] - 5
        lines = [first] if first else []
        confs = [conf] if first else []

    col = sorted(
        (o for o in regions
         if o is not label
         and o["region"].bbox[1] >= top
         and abs(o["region"].bbox[0] - ax) <= 90
         and _dl_is_addr_value(o["text"])),
        key=lambda o: o["region"].bbox[1])

    prev_bottom = None
    for o in col:
        ob = o["region"].bbox
        if prev_bottom is not None and ob[1] - prev_bottom > 55:
            break                          # block ended — large vertical gap
        txt = o["text"].strip()
        if txt:
            lines.append(txt)
            confs.append(o["conf"])
        prev_bottom = ob[3]

    if not lines:
        return (first, conf) if first else ("", 0.0)
    address = re.sub(r"\s*,\s*", ", ", ", ".join(lines))
    address = re.sub(r",\s*,", ",", address).strip(" ,")
    return address, confs[0]


# Issue-date labels, canonical first. Only the genuine 'Date Of Issue'
# labels are accepted — the printed 'Issue Date' / 'Date Of Issue', 'DOI'
# (and the OCR-mangled 'DO/'), and Maharashtra's 'ID' as the last-resort
# fallback. 'DLD' is deliberately excluded: it is a renewal / re-issue
# date, not the licence's date of issue. The colon is never required as a
# trailing boundary — OCR fuses it onto the date ('ID:12-12-2005',
# 'DO/:14-05-2001').
_DL_ISSUE_LABELS = (r"ISSUE\s*DATE", r"DATE\s*OF\s*ISSUE", r"\bDOI\b",
                    r"\bDO[I/]", r"\bID\b")
_DL_DATE_TOKEN = r"([0-3]?\d[/\-. ]*[A-Za-z0-9]{2,}[/\-. ]*\d{4})"


def _license_issue_date(regions: List[Dict],
                        dob_iso: str) -> Tuple[str, float]:
    """Return (YYYY-MM-DD, conf) for the licence's date of issue.

    Tries each issue-date label in priority order, first as a value glued
    into the label's region ('Issue Date 09-Aug-2022', 'ID:12-12-2005')
    and then as the region to the label's right. The first candidate that
    parses to a real date other than the DOB wins — so a class / validity
    date, or the DOB itself, can never be mistaken for the issue date."""
    for lbl in _DL_ISSUE_LABELS:
        glued = re.compile(lbl + r"\s*[:\-]?\s*" + _DL_DATE_TOKEN, re.I)
        for r in regions:
            m = glued.search(r["text"])
            if m:
                iso = _to_iso_date(m.group(1))
                if iso and iso != dob_iso:
                    return iso, r["conf"]
    for lbl in _DL_ISSUE_LABELS:
        v, c, _ = _dl_pair(regions, lbl)
        iso = _to_iso_date(re.sub(r"^[:\s]+", "", v)) if v else ""
        if iso and iso != dob_iso:
            return iso, c
    return "", 0.0


def build_license(regions: List[Dict], full_text: str) -> Dict[str, Any]:
    """Driving Licence response — licence number, holder identity and the
    printed card fields the frontend consumes: name, S/W/D (guardian),
    DOB, blood group, address and issue date.

    A Vehicle Registration Certificate routed to this endpoint still
    yields a stable, valid response — its number lands in
    `license_number` via the fallback pattern, and the DL-only fields
    simply come back empty rather than mixing in the RC schema."""
    up = full_text.upper()

    # Real driving licence number — e.g. 'HR41 20220002435'. No \b
    # boundary: DL numbers are often glued to their label in the OCR
    # output ('DLNOHR4120220002435'). The state code is routinely printed
    # with a hyphen/space separator ('MH-1220050000188'), so a `[\s-]?`
    # separator is allowed between every token group.
    lm = re.search(r"([A-Z]{2}[\s-]?\d{2}[\s-]?\d{11})", up)
    license_raw = lm.group(1) if lm else ""

    # Vehicle Registration Certificate fallback — e.g. 'CH01CY1547'.
    if not license_raw:
        rcm = re.search(
            r"\b([A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,2}[\s-]?\d{3,5})\b", up)
        if rcm:
            license_raw = rcm.group(1)
    # Confidence is looked up against the raw match (the region text still
    # carries the separator); the response value itself is the bare token.
    l_conf = _conf_for(regions, license_raw)
    # The state code is routinely printed with a hyphen/space separator
    # ('JK-1120030025206', 'MH 12 20050000188'); the response contract
    # wants those stripped so the value is a single concatenated token.
    license_num = re.sub(r"[^A-Z0-9]", "", license_raw)

    dob_val, dob_conf = _license_dob(full_text, regions)

    # ── label-anchored card fields (name / guardian / blood group / address
    #    / issue date) — robust to both the separate-region layout (HR-style:
    #    'NAME' → 'JAY VERMA') and the glued layout (MH-style: 'DOB:..') ──
    name_val, name_conf, swd_val, swd_conf = _license_names(regions)

    bg_val, bg_conf, _ = _dl_pair(
        regions, r"BLOOD\s*GRP|BLOOD\s*GROUP|\bBG\b", accept=_dl_is_blood_value)
    bg_val = re.sub(r"^[:\s.]+", "", bg_val).strip()

    address_val, addr_conf = _license_address(regions)

    issue_val, issue_conf = _license_issue_date(regions, dob_val)

    data = {
        "document_type": None,
        "license_number": _field(license_num, l_conf),
        "name": _field(name_val, name_conf),
        "swd": _field(swd_val, swd_conf),
        "dob": {"value": dob_val, "confidence": _conf(dob_conf),
                "yob": False},
        "blood_group": _field(bg_val, bg_conf),
        "address": _field(address_val, addr_conf),
        "issue_date": _field(issue_val, issue_conf),
        "image_url": None,
    }
    return _envelope(data)


# ──────────────────────────────────────────────────────────────────────────────
# Voter ID (kept for parity with the pipeline's doc-type list)
# ──────────────────────────────────────────────────────────────────────────────

def _build_voter_back(regions: List[Dict], full_text: str,
                      ordered: List[Dict]) -> Dict[str, Any]:
    """Address side of a Voter ID — same shape as the address half of an
    Aadhaar back, minus the Aadhaar-specific fields."""
    addr, addr_conf, pin, _co, _rel, comps = _aadhaar_address(ordered)
    return _envelope({
        "ocr_fields": [{
            "document_type": "voterid_back",
            "address": {"value": addr, "confidence": _conf(addr_conf),
                        **comps},
            "zip": _field(pin, addr_conf),
        }],
    })


_VOTER_HEADER_RX = re.compile(
    r"ELECTOR|PHOTO|IDENTITY|CARD|COMMISSION|EPIC|FATHER|HUSBAND",
    re.I,
)


def _looks_like_voter_name(text: str) -> bool:
    if _VOTER_HEADER_RX.search(text):
        return False
    return _looks_like_aadhaar_name(text)


def _build_voter_front(regions: List[Dict], full_text: str,
                       ordered: List[Dict]) -> Dict[str, Any]:
    up = full_text.upper()

    epic_m = re.search(r"(?<![A-Z0-9])([A-Z]{3}\d{7})(?![A-Z0-9])", up)
    epic = epic_m.group(1) if epic_m else ""
    epic_conf = _conf_for(regions, epic)

    # DOB — the label on the card reads 'Date of Birth / Age : DD-MM-YYYY',
    # so allow non-digit filler (e.g. '/ Age :') between the label and the
    # actual date.
    dob_val = ""
    dob_conf = 0.0
    dm = re.search(
        r"(?:DOB|DATE\s*OF\s*BIRTH|BIRTH)[^\d]{0,20}"
        r"([0-3]?\d[/\-.][01]?\d[/\-.]\d{4})",
        full_text, re.I,
    )
    if dm:
        dob_val = _yyyy_mm_dd(dm.group(1))
        dob_conf = _conf_for(regions, dm.group(1))

    # Gender
    gender = ""
    g_conf = 0.0
    gm = re.search(r"\b(MALE|FEMALE|TRANSGENDER)\b", full_text, re.I)
    if gm:
        gender = {"male": "M", "female": "F",
                  "transgender": "T"}[gm.group(1).lower()]
        g_conf = _conf_for(regions, gm.group(1))

    # Age — never read off the card; compute it from the DOB so it stays
    # current. Confidence mirrors the DOB it was derived from.
    age = _age_from_dob(dob_val)
    age_conf = dob_conf if age else 0.0

    # Father / Husband / Mother name → care_of. The label and value sit
    # in the SAME OCR region ("Father'sName:SUKESHVERMA"), so the value
    # is taken from that region's own text — matching against the joined
    # full_text bleeds the next line's tokens (e.g. a trailing 'F' from
    # the Gender line) into the value.
    care_of = ""
    co_conf = 0.0
    for r in ordered:
        cm = re.search(
            r"(?:FATHER|HUSBAND|MOTHER)(?:['’]?S)?\s*NAME"
            r"\s*[:/.\-]?\s*(\S.*)",
            r["text"], re.I,
        )
        if cm and cm.group(1).strip():
            care_of = _clean_name(cm.group(1))
            co_conf = r["conf"]
            break

    # Full name — the holder's own 'Name:' line. The value shares the
    # region with its label, so strip the label off; skip the father's /
    # mother's name region so its value isn't picked up by mistake.
    full_name = ""
    name_conf = 0.0
    for r in ordered:
        t = r["text"].strip()
        if re.search(r"FATHER|HUSBAND|MOTHER", t, re.I):
            continue
        nm = re.search(r"\bNAME\s*[:/.\-]?\s*(\S.*)", t, re.I)
        if nm and nm.group(1).strip():
            full_name = _clean_name(nm.group(1))
            name_conf = r["conf"]
            break

    # Fallback: region just above EPIC, else first plausible name region.
    if not full_name:
        anchor_idx = next((i for i, r in enumerate(ordered)
                           if epic and epic in r["text"].upper()), -1)
        if anchor_idx > 0:
            for j in range(anchor_idx - 1, -1, -1):
                if _looks_like_voter_name(ordered[j]["text"]):
                    full_name = _clean_name(ordered[j]["text"])
                    name_conf = ordered[j]["conf"]
                    break
    if not full_name:
        for r in ordered:
            if _looks_like_voter_name(r["text"]):
                full_name = _clean_name(r["text"])
                name_conf = r["conf"]
                break

    # 'doc' — date-of-card / issuance. Best-effort: any DD/MM/YYYY that
    # isn't the DOB we already captured.
    doc_val = ""
    doc_conf = 0.0
    for d in re.findall(r"\b([0-3]?\d[/\-.][01]?\d[/\-.]\d{4})\b", full_text):
        if d and (not dm or d != dm.group(1)):
            doc_val = _yyyy_mm_dd(d)
            doc_conf = _conf_for(regions, d)
            break

    return _envelope({
        "ocr_fields": [{
            "document_type": "voterid_front",
            "full_name": _field(full_name, name_conf),
            "age": _field(age, age_conf),
            "care_of": _field(care_of, co_conf),
            "dob": {"value": dob_val, "confidence": _conf(dob_conf)},
            "doc": {"value": doc_val, "confidence": _conf(doc_conf)},
            "gender": _field(gender, g_conf),
            "epic_number": _field(epic, epic_conf),
        }],
    })


def build_voter(regions: List[Dict], full_text: str) -> Dict[str, Any]:
    """Route by side: 'Address' without EPIC/elector signals → back."""
    ordered = _order(regions)
    up = full_text.upper()
    is_back = ("ADDRESS" in up
               and not re.search(r"ELECTOR|EPIC|PHOTO\s*IDENTITY", up))
    if is_back:
        return _build_voter_back(regions, full_text, ordered)
    return _build_voter_front(regions, full_text, ordered)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level dispatch
# ──────────────────────────────────────────────────────────────────────────────

_BUILDERS = {
    "PAN": build_pan,
    "AADHAAR": build_aadhaar,
    "PASSPORT": build_passport,
    "DRIVING_LICENSE": build_license,
    "VOTER_ID": build_voter,
}


def build_output_json(doc_type: str, regions: List[Dict],
                      full_text: str) -> Dict[str, Any]:
    builder = _BUILDERS.get(doc_type)
    if builder is None:
        return failure_envelope(f"unsupported document type: {doc_type}")
    return builder(regions, full_text)
