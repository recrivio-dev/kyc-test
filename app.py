import os
import streamlit as st
from kyc_pipeline import DocumentPipeline

# Ensure storage workspace exists
os.makedirs("sample-docs", exist_ok=True)

# Configure wide dashboard workspace views
st.set_page_config(
    page_title="KYC Verification Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimalistic container padding
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Cache pipeline instantiation to keep execution fast
@st.cache_resource
def get_pipeline():
    return DocumentPipeline()

pipeline = get_pipeline()

try:
    pipeline._ensure_ocr()

    if not pipeline.ocr_available:
        st.warning(
            "OCR backend unavailable"
        )

except Exception as e:

    st.error(
        f"OCR init failed: {e}"
    )

# Title Header Section
st.title("🛡️ Secure KYC Verification Pipeline")
st.markdown("Automated classification, validation, text parsing, and dynamic physical redaction engine.")
st.divider()

# Setup interactive dashboard columns
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("1. Setup & Upload")
    
    # Encapsulating controls in a form layout stops dynamic automatic script reruns
    with st.form(key="kyc_execution_form"):
        intended_type = st.selectbox(
            "Select Intended Document Type",
            options=["AADHAAR", "PAN", "VOTER_ID", "DRIVING_LICENSE", "PASSPORT"],
            format_func=lambda x: pipeline.get_printable_name(x)
        )
        
        uploaded_file = st.file_uploader(
            "Upload Source Document File", 
            type=["jpg", "jpeg", "png", "webp", "pdf"],
            help="Images and PDF arrays are fully supported."
        )
        
        submit_button = st.form_submit_button(label="Execute KYC Pipeline", use_container_width=True)

with col2:
    st.subheader("2. Processing Output Console")
    
    if submit_button:
        if uploaded_file is None:
            st.warning("⚠️ Please provide a source file payload to proceed.")
        else:
            with st.spinner("Extracting multi-angle OCR features and assessing boundaries..."):
                # Save input temporarily for OpenCV ingestion mapping
                temp_path = f"sample-docs/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Execute primary parsing classes
                result = pipeline.process_and_verify(temp_path, intended_type)
                
                # Render payload outcomes visually
                actual_name = pipeline.get_printable_name(result.get('actual_type'))
                
                if result.get("status") == "SUCCESS":
                    st.success("✅ **Verification Verified Successfully**")

                    # OCR metadata
                    with st.expander("OCR details"):
                        st.write(f"Engine: **{result.get('ocr_engine')}**")
                        st.write(f"Avg confidence: **{result.get('ocr_avg_confidence')}**")
                        st.write(f"Decision: `{result.get('ocr_decision_reason')}`")
                    
                    # Display structured output parameters cleanly
                    st.metric("Detected Classification", actual_name)
                    st.text_input("Raw Target Extraction ID", result.get("extracted_id"), disabled=True)
                    st.text_input("Secured Masked Output", result.get("masked_id"), disabled=True)
                    
                    # Present visual redacted image file output mapping
                    if "masked_image_file" in result and os.path.exists(result["masked_image_file"]):
                        st.caption("Redacted Document Preview (Solid Bounding Layers Applied)")
                        st.image(result["masked_image_file"], use_container_width=True)
                else:
                    st.error("❌ **Pipeline Verification Blocked**")
                    st.metric("Detected Classification", actual_name)
                    st.warning(result.get("message"))

                    with st.expander("OCR details"):
                        st.write(f"Engine: **{result.get('ocr_engine')}**")
                        st.write(f"Avg confidence: **{result.get('ocr_avg_confidence')}**")
                        st.write(f"Decision: `{result.get('ocr_decision_reason')}`")

            # Provide visibility into the complete raw string output layout logs
            with st.expander("Show Raw OCR Processing Data Log"):
                st.code(result.get("extracted_text", "[No string content located]"), language="text")

            # Debug preview images for preprocessing stages
            with st.expander("Debug: preprocessing previews"):
                try:
                    dbg = pipeline.build_preprocess_debug(temp_path)
                    st.caption("Base (cropped + deskewed) image")
                    st.image(dbg["base_color"], channels="BGR", use_container_width=True)

                    v = dbg.get("variants", {})
                    cols = st.columns(3)
                    if "color" in v:
                        cols[0].caption("Variant: color")
                        cols[0].image(v["color"], channels="BGR", use_container_width=True)
                    if "gray" in v:
                        cols[1].caption("Variant: gray")
                        cols[1].image(v["gray"], clamp=True, use_container_width=True)
                    if "bin" in v:
                        cols[2].caption("Variant: bin")
                        cols[2].image(v["bin"], clamp=True, use_container_width=True)

                    st.caption("Crop / Deskew debug")
                    st.json({"crop": dbg.get("crop"), "deskew": dbg.get("deskew")})
                except Exception as e:
                    st.warning(f"Could not build debug previews: {e}")
                
            # Clean up target local file cache cleanly post execution flow
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        st.info("Awaiting verification execution triggers. Configure your payload targets on the left console.")