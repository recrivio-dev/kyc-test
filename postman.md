# Recrivio KYC OCR — API Reference

Base URL: `http://127.0.0.1:8000`

All OCR endpoints accept `multipart/form-data` with an image file (`.jpg`, `.png`, `.jpeg`). The response envelope is consistent across endpoints:

```json
{
  "data": { ... },
  "status_code": 200,
  "message_code": "success",
  "message": null,
  "success": true
}
```

On failure:

```json
{
  "data": {},
  "status_code": 400,
  "message_code": "failed",
  "message": "<error description>",
  "success": false
}
```

---

## 1. Health Check

**Endpoint:** `GET /healthz`

Returns whether the OCR engine is initialized and ready.

### Request
```
GET http://127.0.0.1:8000/healthz
```

### Expected Response — `200 OK`
```json
{
  "ok": true
}
```

### Error Response
If OCR engine is not available:
```json
{
  "ok": false
}
```

---

## 2. Generic OCR Endpoint

**Endpoint:** `POST /api/v1/ocr`

Runs the full locate → read → mask pipeline. Caller chooses the document type via the `doc_type` form field.

### Request
```
POST http://127.0.0.1:8000/api/v1/ocr
Content-Type: multipart/form-data
```

| Field      | Type   | Required | Description |
|------------|--------|----------|-------------|
| `file`     | File   | Yes      | Document image file |
| `doc_type` | String | Yes      | One of: `PAN`, `AADHAAR`, `PASSPORT`, `VOTER_ID`, `DRIVING_LICENSE` |

### cURL Example
```bash
curl -F "file=@sample/pan1.png" \
     -F "doc_type=PAN" \
     http://127.0.0.1:8000/api/v1/ocr
```

### Error Responses

**`422 Unprocessable Entity`** — Missing or invalid `doc_type`:
```json
{
  "detail": [
    {
      "loc": ["body", "doc_type"],
      "msg": "value is not a valid enumeration member; permitted: 'PAN', 'AADHAAR', 'PASSPORT', 'VOTER_ID', 'DRIVING_LICENSE'",
      "type": "type_error.enum"
    }
  ]
}
```

**`400 Bad Request`** — Pipeline failure (corrupt file, no document detected, OCR error):
```json
{
  "data": {},
  "status_code": 400,
  "message_code": "failed",
  "message": "processing failed",
  "success": false
}
```

---

## 3. PAN Card OCR

**Endpoint:** `POST /api/v1/ocr/pan`

### Request
```
POST http://127.0.0.1:8000/api/v1/ocr/pan
Content-Type: multipart/form-data
```

| Field  | Type | Required | Description |
|--------|------|----------|-------------|
| `file` | File | Yes      | PAN card image |

### cURL Example
```bash
curl -F "file=@sample/pan1.png" http://127.0.0.1:8000/api/v1/ocr/pan
```

### Expected Response — `200 OK`
```json
{
  "data": {
    "ocr_fields": [
      {
        "document_type": "pan",
        "pan_number": { "value": "ABCDE1234F", "confidence": 96 },
        "full_name":  { "value": "Ravi Kumar Sharma", "confidence": 92 },
        "father_name": { "value": "Suresh Kumar Sharma", "confidence": 88 },
        "dob":        { "value": "12/05/1990", "confidence": 95 }
      }
    ]
  },
  "status_code": 200,
  "message_code": "success",
  "message": null,
  "success": true
}
```

### Error Response — `400 Bad Request`
```json
{
  "data": {},
  "status_code": 400,
  "message_code": "failed",
  "message": "processing failed",
  "success": false
}
```

---

## 4. Aadhaar OCR

**Endpoint:** `POST /api/v1/ocr/aadhaar`

Auto-detects whether the uploaded image is the front (identity) or back (address) side and returns the appropriate object(s) in `ocr_fields`.

### Request
```
POST http://127.0.0.1:8000/api/v1/ocr/aadhaar
Content-Type: multipart/form-data
```

| Field  | Type | Required | Description |
|--------|------|----------|-------------|
| `file` | File | Yes      | Aadhaar card image (front or back) |

### cURL Example
```bash
curl -F "file=@sample/ad1.jpeg" http://127.0.0.1:8000/api/v1/ocr/aadhaar
```

### Expected Response — Front Side
```json
{
  "data": {
    "ocr_fields": [
      {
        "document_type": "aadhaar_front_bottom",
        "full_name":  { "value": "Jay Verma", "confidence": 91 },
        "gender":     { "value": "M", "confidence": 90 },
        "mother_name": { "value": "", "confidence": 0 },
        "father_name": { "value": "", "confidence": 0 },
        "dob": {
          "value": "1995-07-14",
          "confidence": 93,
          "yob": false
        },
        "aadhaar_number": {
          "value": "123456789012",
          "confidence": 97,
          "is_masked": false,
          "input_validation": false
        },
        "image_url": null,
        "uniqueness_id": "9b74c9897bac770ffc029102a200c5de..."
      }
    ]
  },
  "status_code": 200,
  "message_code": "success",
  "message": null,
  "success": true
}
```

### Expected Response — Back Side
```json
{
  "data": {
    "ocr_fields": [
      {
        "document_type": "aadhaar_back",
        "address": {
          "value": "S/O Suresh Kumar, House 12, Sector 21, Chandigarh, 160021",
          "confidence": 88,
          "first_line": "",
          "second_line": "",
          "locality": "",
          "landmark": "",
          "house_number": "12",
          "district": "Sector 21",
          "city": "Chandigarh",
          "state": "Chandigarh",
          "country": "India",
          "zip": "160021"
        },
        "zip": { "value": "160021", "confidence": 88 },
        "care_of": {
          "value": "Suresh Kumar",
          "confidence": 88,
          "relation": "father"
        },
        "aadhaar_number": {
          "value": "123456789012",
          "confidence": 96,
          "is_masked": false,
          "input_validation": false
        },
        "image_url": null
      }
    ]
  },
  "status_code": 200,
  "message_code": "success",
  "message": null,
  "success": true
}
```

### Error Response — `400 Bad Request`
```json
{
  "data": {},
  "status_code": 400,
  "message_code": "failed",
  "message": "processing failed",
  "success": false
}
```

---

## 5. Passport OCR

**Endpoint:** `POST /api/v1/ocr/passport`

Auto-detects front vs. back side based on presence of MRZ versus address/family labels.

### Request
```
POST http://127.0.0.1:8000/api/v1/ocr/passport
Content-Type: multipart/form-data
```

| Field  | Type | Required | Description |
|--------|------|----------|-------------|
| `file` | File | Yes      | Passport image (front or back) |

### cURL Example
```bash
curl -F "file=@sample/Passport_front.jpeg" http://127.0.0.1:8000/api/v1/ocr/passport
```

### Expected Response — Front Side
```json
{
  "data": {
    "ocr_fields": [
      {
        "document_type": "passport_front",
        "country_code":  { "value": "IND", "confidence": 95 },
        "dob":           { "value": "14/07/1995", "confidence": 95 },
        "doe":           { "value": "13/07/2035", "confidence": 95 },
        "doi":           { "value": "14/07/2025", "confidence": 92 },
        "gender":        { "value": "M", "confidence": 95 },
        "given_name":    { "value": "Jay", "confidence": 95 },
        "nationality":   { "value": "INDIAN", "confidence": 95 },
        "passport_num":  { "value": "M1234567", "confidence": 96 },
        "place_of_birth":{ "value": "DELHI", "confidence": 90 },
        "place_of_issue":{ "value": "DELHI", "confidence": 90 },
        "surname":       { "value": "Verma", "confidence": 95 },
        "mrz_line_1":    { "value": "P<INDVERMA<<JAY<<<<<<<<<<<<<<<<<<<<<<<<<<<<", "confidence": 95 },
        "mrz_line_2":    { "value": "M1234567<7IND9507147M3507139<<<<<<<<<<<<<<<<", "confidence": 95 },
        "type_of_passport": { "value": "P", "confidence": 95 },
        "passport_validity": null
      }
    ]
  },
  "status_code": 200,
  "message_code": "success",
  "message": null,
  "success": true
}
```

### Expected Response — Back Side
```json
{
  "data": {
    "ocr_fields": [
      {
        "document_type": "passport_back",
        "address": {
          "value": "House 45, Sector 17, Chandigarh, 160017, INDIA",
          "confidence": 90
        },
        "father":  { "value": "Suresh Verma", "confidence": 92 },
        "mother":  { "value": "Anita Verma",  "confidence": 92 },
        "file_num": { "value": "DL1320250012345", "confidence": 89 },
        "old_doi":  { "value": "", "confidence": 0 },
        "old_passport_num": { "value": "K9876543", "confidence": 88 },
        "old_place_of_issue": { "value": "", "confidence": 0 },
        "pin":     { "value": "160017", "confidence": 90 },
        "spouse":  { "value": "", "confidence": 0 }
      }
    ]
  },
  "status_code": 200,
  "message_code": "success",
  "message": null,
  "success": true
}
```

### Error Response — `400 Bad Request`
```json
{
  "data": {},
  "status_code": 400,
  "message_code": "failed",
  "message": "processing failed",
  "success": false
}
```

---

## 6. Voter ID OCR

**Endpoint:** `POST /api/v1/ocr/voter-id`

Auto-detects front (identity) vs. back (address) side.

### Request
```
POST http://127.0.0.1:8000/api/v1/ocr/voter-id
Content-Type: multipart/form-data
```

| Field  | Type | Required | Description |
|--------|------|----------|-------------|
| `file` | File | Yes      | Voter ID image (front or back) |

### cURL Example
```bash
curl -F "file=@sample/vote1.png" http://127.0.0.1:8000/api/v1/ocr/voter-id
```

### Expected Response — Front Side
```json
{
  "data": {
    "ocr_fields": [
      {
        "document_type": "voterid_front",
        "full_name":   { "value": "Ravi Kumar Sharma", "confidence": 92 },
        "age":         { "value": "30", "confidence": 88 },
        "care_of":     { "value": "Suresh Kumar Sharma", "confidence": 89 },
        "dob":         { "value": "1995-07-14", "confidence": 93 },
        "doc":         { "value": "2018-01-01", "confidence": 80 },
        "gender":      { "value": "M", "confidence": 91 },
        "epic_number": { "value": "ABC1234567", "confidence": 96 }
      }
    ]
  },
  "status_code": 200,
  "message_code": "success",
  "message": null,
  "success": true
}
```

### Expected Response — Back Side
```json
{
  "data": {
    "ocr_fields": [
      {
        "document_type": "voterid_back",
        "address": {
          "value": "House 12, Sector 21, Chandigarh, 160021",
          "confidence": 88,
          "first_line": "",
          "second_line": "",
          "locality": "",
          "landmark": "",
          "house_number": "12",
          "district": "Sector 21",
          "city": "Chandigarh",
          "state": "Chandigarh",
          "country": "India",
          "zip": "160021"
        },
        "zip": { "value": "160021", "confidence": 88 }
      }
    ]
  },
  "status_code": 200,
  "message_code": "success",
  "message": null,
  "success": true
}
```

### Error Response — `400 Bad Request`
```json
{
  "data": {},
  "status_code": 400,
  "message_code": "failed",
  "message": "processing failed",
  "success": false
}
```

---

## 7. Driving License OCR

**Endpoint:** `POST /api/v1/ocr/driving-license`

Returns the licence number plus the holder's printed card fields — `name`, `swd` (Son/Wife/Daughter-of), `dob`, `blood_group`, `address` and `issue_date`. A Vehicle Registration Certificate sent to this endpoint still returns a valid, DL-shaped response (its number lands in `license_number`; the DL-only fields come back empty).

### Request
```
POST http://127.0.0.1:8000/api/v1/ocr/driving-license
Content-Type: multipart/form-data
```

| Field  | Type | Required | Description |
|--------|------|----------|-------------|
| `file` | File | Yes      | Driving Licence image |

### cURL Example
```bash
curl -F "file=@sample/dl1.png" http://127.0.0.1:8000/api/v1/ocr/driving-license
```

### Expected Response — Driving Licence
```json
{
  "data": {
    "document_type": null,
    "license_number": {
      "value": "HR4120220002435",
      "confidence": 98
    },
    "name": {
      "value": "Jay Verma",
      "confidence": 97
    },
    "swd": {
      "value": "Sukesh Verma",
      "confidence": 96
    },
    "dob": {
      "value": "2004-06-09",
      "confidence": 92,
      "yob": false
    },
    "blood_group": {
      "value": "B+",
      "confidence": 98
    },
    "address": {
      "value": "150 KHADAK SINGH FARM, KURUKSHETRA ROAD, PEHOWA, KURUKSHETRA, HR 136128",
      "confidence": 93
    },
    "issue_date": {
      "value": "2022-08-09",
      "confidence": 96
    },
    "image_url": null
  },
  "status_code": 200,
  "message_code": "success",
  "message": null,
  "success": true
}
```

`dob` is a required field — taken from a labelled `DOB` line when present, otherwise from the earliest date on the card.

### Error Response — `400 Bad Request`
```json
{
  "data": {},
  "status_code": 400,
  "message_code": "failed",
  "message": "processing failed",
  "success": false
}
```

---

## Common Error Scenarios

| Status | When It Occurs | Body |
|--------|----------------|------|
| `400`  | OCR pipeline could not extract any usable text or detect a document | `{"data": {}, "status_code": 400, "message_code": "failed", "message": "processing failed", "success": false}` |
| `422`  | A required form field is missing or `doc_type` is not in the allowed list | FastAPI validation error object with `detail` array |
| `500`  | Unhandled server exception (model load failure, disk write failure on temp file) | FastAPI default `{"detail": "Internal Server Error"}` |

---

## Notes

- All field values follow `{ "value": "<string>", "confidence": <int 0-100> }` shape unless otherwise noted.
- `confidence` is the OCR-reported region-level score, scaled to `0-100`. A value of `0` indicates the field was not extracted.
- Dates may be in `DD/MM/YYYY` (PAN, Passport) or `YYYY-MM-DD` (Aadhaar, Voter ID, Driving Licence DOB) format depending on document type.
- `ocr_fields` is always an array — Aadhaar uploads showing both sides may return two entries (front + back).
- CORS is enabled for all origins (`*`) — tighten in production before deploy.
