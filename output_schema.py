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
), key=len, reverse=True)


def _split_glued_surname(token: str) -> str:
    """Split an all-caps glued name on a trailing known surname.

    'RAHULGUPTA' → 'RAHUL GUPTA'. Only fires when the remaining prefix is
    a plausible given name (≥2 chars); a token that *is* exactly a
    surname is left untouched."""
    up = token.upper()
    for sn in _COMMON_SURNAMES:
        if up.endswith(sn) and len(up) - len(sn) >= 2:
            return f"{token[:-len(sn)]} {token[-len(sn):]}"
    return token


def _clean_name(text: str) -> str:
    """Normalise a person-name string.

    OCR mangles the spacing between a first and last name three ways:
      * it splits them across text lines      → 'JAY\\nVERMA'
      * it glues them keeping the case change → 'JayVerma'
      * it glues them in all-caps             → 'RAHULGUPTA'
    Whitespace runs collapse to one space; a lowercase→uppercase boundary
    becomes a space (a genuine intra-word capital is vanishingly rare in
    Indian names); an all-caps glued token is split on a trailing known
    surname since it offers no other boundary."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)
    if " " not in t and len(t) >= 5 and t.isalpha():
        t = _split_glued_surname(t)
    return t.title()


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


_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
     "nov", "dec"), 1)}


def _to_iso_date(s: str) -> str:
    """Normalise 'DD/MM/YYYY', 'DD-MM-YYYY' or 'DD-Mon-YYYY' (e.g.
    '09-Jun-2004', '09-August-2022') to ISO 'YYYY-MM-DD'. Returns '' when
    the string isn't a recognisable date."""
    m = re.match(r"\s*(\d{1,2})[/\-.\s]+([A-Za-z]{3,}|\d{1,2})[/\-.\s]+"
                 r"(\d{4})", s)
    if not m:
        return ""
    d, mo, y = m.groups()
    mon = int(mo) if mo.isdigit() else _MONTHS.get(mo[:3].lower(), 0)
    if not (1 <= mon <= 12) or not (1 <= int(d) <= 31):
        return ""
    return f"{y}-{mon:02d}-{int(d):02d}"


def _yymmdd_to_dmy(s: str) -> str:
    """Convert MRZ 'YYMMDD' to 'DD/MM/YYYY'. The MRZ century pivot at 30
    matches Indian passports issued from the 2000s onward."""
    if not re.fullmatch(r"\d{6}", s):
        return ""
    yy, mo, d = int(s[:2]), int(s[2:4]), int(s[4:6])
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

def _aadhaar_dob(full_text: str) -> Tuple[str, bool]:
    """Return (YYYY-MM-DD, yob_only_flag)."""
    m = re.search(r"DOB[:\s]*([0-3]?\d[/\-.][01]?\d[/\-.]\d{4})", full_text, re.I)
    if m:
        return _yyyy_mm_dd(m.group(1)), False
    m = re.search(r"(?:YEAR\s*OF\s*BIRTH|YOB)[:\s]*(\d{4})", full_text, re.I)
    if m:
        return m.group(1), True
    m = re.search(r"\b([0-3]?\d[/\-.][01]?\d[/\-.]\d{4})\b", full_text)
    if m:
        return _yyyy_mm_dd(m.group(1)), False
    return "", False


_AADHAAR_LABELS = re.compile(
    r"DOB|YEAR|MALE|FEMALE|GOVERNMENT|INDIA|AADHAAR|UIDAI|AUTHORITY|UNIQUE|"
    r"IDENTIFICATION|ADDRESS|MOBILE|HELP|INFORMATION|SCANNING|VERIFIED|"
    r"ENROL|PROOF|CITIZENSHIP|DATE|BIRTH|ISSUED|MERA|XML|CODE|SCANNED",
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


_DOB_ANCHOR_RX = re.compile(
    r"(?:DOB|DATE\s*OF\s*BIRTH|YEAR\s*OF\s*BIRTH|YOB)"
    r"[:\s/]*(?:[0-3]?\d[/\-.][01]?\d[/\-.]\d{4}|\d{4}\b)",
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
    r"MANDAL|TALUK|GALI|CHOWK",
    re.I,
)


def _is_aadhaar_addr_line(text: str) -> bool:
    """A line carrying Aadhaar address content — identified by the
    structured markers Aadhaar prints (C/O, House No, VTC, PO, District,
    State, PIN Code …), a 'CITY-PIN' chunk, or a bare 6-digit PIN."""
    t = text.strip()
    if not t:
        return False
    if _AADHAAR_ADDR_MARKERS.search(t):
        return True
    if re.search(r"[A-Za-z]{3,}\s*-\s*\d{6}\b", t):
        return True
    return bool(re.fullmatch(r"\d{6}", re.sub(r"\s+", "", t)))


def _aadhaar_address(ordered: List[Dict], aadhaar_num: str = "") -> Tuple[
        str, float, str, str, str, Dict[str, str]]:
    """Return (address_string, conf, zip, care_of, care_of_relation, components).

    Aadhaar address extraction has to survive two awkward layouts:

      * the boilerplate paragraph contains the word 'address', so the
        printed 'Address:' label is found by being a SHORT region — not
        any region merely mentioning the word; and
      * a full e-card prints all four panels on one image, so the address
        appears twice. Address-content lines are gathered by marker,
        clustered into x-columns (one column per panel) and the richest
        cluster wins — that drops the duplicate, lower-quality copy."""
    # ── locate the printed 'Address:' label (a short region) ──
    label_idx = -1
    for i, r in enumerate(ordered):
        t = r["text"].strip()
        if re.search(r"\bADDRESS\b|पता", t, re.I) and len(t.split()) <= 3:
            label_idx = i
            break

    digits_aadhaar = re.sub(r"\D", "", aadhaar_num)

    def _is_id_number(t: str) -> bool:
        """Aadhaar / VID numbers print near the address — never part of it."""
        d = re.sub(r"\D", "", t)
        return len(d) >= 11 or bool(digits_aadhaar and digits_aadhaar in d)

    # ── candidate address regions: marker lines, plus the line right
    #    after the label (the opening address line often has no marker) ──
    candidates: List[Dict] = []
    for i, r in enumerate(ordered):
        t = r["text"].strip()
        if not t or _is_id_number(t):
            continue
        if re.search(r"VID[:\s]|MOBILE|UIDAI|WWW\.|HELP@|ENROL", t, re.I):
            continue
        if _is_aadhaar_addr_line(t) or (label_idx >= 0 and i == label_idx + 1):
            candidates.append(r)
    if not candidates:
        return "", 0.0, "", "", "father", {}

    # ── cluster by x-column — a multi-panel e-card prints the address
    #    once per panel, each panel being its own x-band ──
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

    parts = [r["text"].strip() for r in best]
    confs = [r["conf"] for r in best]

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

    # Fallback city / district: last two purely-alphabetic comma tokens.
    if not components["city"] or not components["district"]:
        tokens = [t.strip() for t in full.split(",") if t.strip()]
        alpha = [t for t in tokens
                 if re.fullmatch(r"[A-Za-z][A-Za-z\s\-]*", t)
                 and t.upper() not in {components["state"].upper(), "INDIA"}]
        if not components["city"] and alpha:
            components["city"] = alpha[-1].strip().title()
        if not components["district"] and len(alpha) >= 2:
            components["district"] = alpha[-2].strip().title()

    conf = sum(confs) / len(confs) if confs else 0.0
    return full, conf, pin, care_of, care_of_relation, components


def build_aadhaar(regions: List[Dict], full_text: str) -> Dict[str, Any]:
    ordered = _order(regions)

    # ── front-side signals (identity) ──
    num_match = re.search(r"(?<!\d)(\d{4}\s?\d{4}\s?\d{4})(?!\d)", full_text)
    aadhaar_num = re.sub(r"\s+", "", num_match.group(1)) if num_match else ""
    a_conf = _conf_for(regions, aadhaar_num)

    # Anchor on a labelled DOB so the confidence lookup lands on the real
    # birth-date region, not a stray print/issue date elsewhere on the card.
    dob_raw_match = re.search(
        r"(?:DOB|DATE\s*OF\s*BIRTH)[:\s/]*"
        r"([0-3]?\d[/\-.][01]?\d[/\-.]\d{4})", full_text, re.I)
    dob_raw = dob_raw_match.group(1) if dob_raw_match else ""
    dob_val, yob_only = _aadhaar_dob(full_text)
    dob_conf = _conf_for(regions, dob_raw) if dob_raw else 0.0

    gender = ""
    g_conf = 0.0
    gm = re.search(r"\b(MALE|FEMALE|TRANSGENDER)\b", full_text, re.I)
    if gm:
        gender = {"male": "M", "female": "F",
                  "transgender": "T"}[gm.group(1).lower()]
        g_conf = _conf_for(regions, gm.group(1))

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
    line1 = next((ln for ln in nospace_per_line
                  if re.match(r"P[A-Z<]<", ln) and "<" in ln), "")
    line2 = next((ln for ln in nospace_per_line
                  if re.match(r"[A-Z]\d{6,7}<", ln)), "")
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
    pm = re.match(r"([A-Z]\d{6,7})", line2)
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
    doe = _yymmdd_to_dmy(doe_mrz)

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

    # Fallback for passport number from a labelled token in the body.
    if not passport_num:
        bm = re.search(r"\b([A-Z]\d{7})\b", full_text)
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
    date on the card is the DOB (issue / validity fall after it)."""
    # DD<sep>MM-or-Mon<sep>YYYY, tolerating numeric and named months.
    date_rx = r"[0-3]?\d[/\-. ]+(?:[A-Za-z]{3,}|[01]?\d)[/\-. ]+\d{4}"
    m = re.search(rf"(?:DOB|DATE\s*OF\s*BIRTH|BIRTH)[:\s]*({date_rx})",
                  full_text, re.I)
    if m:
        iso = _to_iso_date(m.group(1))
        if iso:
            return iso, _conf_for(regions, m.group(1))

    # earliest date = DOB; ISO strings sort chronologically.
    dated = []
    for raw in re.findall(rf"\b{date_rx}\b", full_text):
        iso = _to_iso_date(raw)
        if iso:
            dated.append((iso, raw))
    if dated:
        dated.sort()
        iso, raw = dated[0]
        return iso, _conf_for(regions, raw)
    return "", 0.0


def build_license(regions: List[Dict], full_text: str) -> Dict[str, Any]:
    """Driving Licence response — the minimal identity contract the
    frontend consumes: licence number + DOB.

    A Vehicle Registration Certificate routed to this endpoint still
    yields a stable, valid response — its number lands in
    `license_number` via the fallback pattern. RC-specific fields
    (chassis / engine / address) are deliberately not emitted here: this
    contract is DL-shaped and must not be mixed with the RC schema."""
    up = full_text.upper()

    # Real driving licence number — e.g. 'HR41 20220002435'. No \b
    # boundary: DL numbers are often glued to their label in the OCR
    # output ('DLNOHR4120220002435').
    lm = re.search(r"([A-Z]{2}\d{2}\s?\d{11})", up)
    license_num = lm.group(1) if lm else ""

    # Vehicle Registration Certificate fallback — e.g. 'CH01CY1547'.
    if not license_num:
        rcm = re.search(r"\b([A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,5})\b", up)
        if rcm:
            license_num = rcm.group(1)
    l_conf = _conf_for(regions, license_num)

    dob_val, dob_conf = _license_dob(full_text, regions)

    data = {
        "document_type": None,
        "license_number": _field(license_num, l_conf),
        "dob": {"value": dob_val, "confidence": _conf(dob_conf),
                "yob": False},
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

    # DOB
    dob_val = ""
    dob_conf = 0.0
    dm = re.search(
        r"(?:DOB|DATE\s*OF\s*BIRTH|BIRTH)[:\s]*([0-3]?\d[/\-.][01]?\d[/\-.]\d{4})",
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

    # Age (labelled)
    age = ""
    age_conf = 0.0
    am = re.search(r"\bAGE[:\s]+(\d{1,3})\b", full_text, re.I)
    if am:
        age = am.group(1)
        age_conf = _conf_for(regions, age)

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
