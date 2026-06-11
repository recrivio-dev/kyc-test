"""Quick Streamlit harness for the POST /mask-identity endpoint.

Run the API first, then this UI:

    uvicorn api:app --reload --port 8000
    streamlit run mask_identity_app.py

Upload a document, pick its type, and inspect the masked image, the
masked-region geometry, and the raw JSON envelope.
"""
from __future__ import annotations

import base64
import io
import json

import requests
import streamlit as st
from PIL import Image, ImageDraw

DOC_TYPES = ["aadhaar", "pan", "voter-id", "passport", "driving-license"]

st.set_page_config(page_title="Mask Identity Tester", layout="wide")
st.title("🛡️  /mask-identity tester")

with st.sidebar:
    base_url = st.text_input("API base URL", "http://127.0.0.1:8000")
    document_type = st.selectbox("document_type", DOC_TYPES)
    client_id = st.text_input("client_id (optional)", "")
    draw_boxes = st.checkbox("Outline masked_regions", value=True)

uploaded = st.file_uploader(
    "Upload document (image / PDF)",
    type=["png", "jpg", "jpeg", "webp", "pdf"],
)

if uploaded is not None and st.button("Mask identity", type="primary"):
    files = {"file": (uploaded.name, uploaded.getvalue(),
                      uploaded.type or "application/octet-stream")}
    data = {"document_type": document_type}
    if client_id.strip():
        data["client_id"] = client_id.strip()

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/mask-identity",
            files=files, data=data, timeout=120,
        )
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        st.stop()

    try:
        payload = resp.json()
    except ValueError:
        st.error(f"Non-JSON response (HTTP {resp.status_code}): {resp.text[:500]}")
        st.stop()

    if resp.ok:
        st.success(f"HTTP {resp.status_code} — success")
    else:
        st.error(f"HTTP {resp.status_code} — {payload.get('message')}")

    body = payload.get("data") or {}
    regions = body.get("masked_regions") or []
    b64_img = body.get("masked_image")

    left, right = st.columns(2)

    with left:
        st.subheader("Masked image")
        if b64_img:
            img = Image.open(io.BytesIO(base64.b64decode(b64_img))).convert("RGB")
            if draw_boxes and regions:
                draw = ImageDraw.Draw(img)
                for r in regions:
                    x, y, w, h = r["bbox"]
                    draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
            st.image(img, use_container_width=True)
            st.caption(f"document_detected={body.get('document_detected')} · "
                       f"mime={body.get('mime_type')} · "
                       f"{len(regions)} masked region(s)")
        else:
            st.warning("No masked_image returned — endpoint failed closed.")

    with right:
        st.subheader("masked_regions")
        if regions:
            st.dataframe(
                [{"field": r["field"], "bbox": r["bbox"],
                  "confidence": r["confidence"]} for r in regions],
                use_container_width=True,
            )
        else:
            st.write("—")

    st.subheader("Raw JSON envelope")
    shown = json.loads(json.dumps(payload))  # deep copy
    if shown.get("data", {}).get("masked_image"):
        shown["data"]["masked_image"] = (
            f"<{len(shown['data']['masked_image'])} base64 chars>")
    st.json(shown)
