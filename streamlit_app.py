import os
import io
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pydicom
import streamlit as st
import tensorflow as tf
from PIL import Image  # untuk ikon gambar

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="Low-Dose CT Medical Image Denoising App",
    page_icon=Image.open("TPU-logo.jpg"),  # gunakan logo terlampir
    layout="wide",
)

st.title("🏥 Low-Dose CT Medical Image Denoising (IMA / DICOM)")
st.markdown("---")

# =========================
# Sidebar: Input
# =========================
st.sidebar.header("⚙️ Settings")

# Noisy images uploader
st.sidebar.subheader("📁 Input Noisy Images")
uploaded_imas = st.sidebar.file_uploader(
    "Upload noisy IMA/DICOM files (multi-select supported)",
    type=["ima", "dcm"],
    accept_multiple_files=True,
    help="Pilih banyak berkas sekaligus (Ctrl/Shift klik) untuk batch upload.",
)

zip_folder = st.sidebar.file_uploader(
    "Atau upload folder sebagai .zip",
    type=["zip"],
    help="Unggah folder yang dikompres ZIP untuk memproses seluruh isinya.",
)

# Model uploader
st.sidebar.subheader("🤖 Model File")
uploaded_model = st.sidebar.file_uploader(
    "Upload model (.h5)",
    type=["h5"],
    help="Unggah file model Keras/TensorFlow berformat .h5.",
)

# =========================
# Helpers
# =========================
def list_ima_dcm_recursive(root_dir: str):
    paths = []
    for r, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith((".ima", ".dcm")):
                paths.append(os.path.join(r, name))
    paths.sort()
    return paths

@st.cache_resource
def load_denoising_model(model_path: str):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, True
    except Exception as e:
        return str(e), False

def read_dicom_from_path(path: str):
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32) / 255.0
        return img, dcm, True
    except Exception as e:
        return str(e), None, False

def read_dicom_from_bytes(b: bytes):
    try:
        dcm = pydicom.dcmread(io.BytesIO(b))
        img = dcm.pixel_array.astype(np.float32) / 255.0
        return img, dcm, True
    except Exception as e:
        return str(e), None, False

def dicom_bytes(original_dcm, denoised_array: np.ndarray) -> bytes:
    """Return DICOM bytes for inclusion in ZIP (file-like write)."""
    denoised_scaled = (denoised_array * 255.0).astype(np.uint16)
    dcm_out = original_dcm.copy()
    dcm_out.PixelData = denoised_scaled.tobytes()
    mem = io.BytesIO()
    dcm_out.save_as(mem)  # pydicom mendukung file-like object
    mem.seek(0)
    return mem.read()

# =========================
# UI: Processing & Info
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Processing Status")
    validation_status = st.container()

    files_ready = (uploaded_imas and len(uploaded_imas) > 0) or (zip_folder is not None)
    model_ready = uploaded_model is not None

    start_disabled = not (files_ready and model_ready)

    if st.button("🚀 Start Denoising and Prepare ZIP", type="primary", disabled=start_disabled):
        with validation_status:
            # Tentukan sumber input
            input_mode = None
            input_file_items = []

            if uploaded_imas and len(uploaded_imas) > 0:
                input_mode = "uploaded_files"
                input_file_items = uploaded_imas
            elif zip_folder is not None:
                input_mode = "zip"
            else:
                st.error("❌ No noisy images provided!")
                st.stop()

            # Simpan model sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model:
                tmp_model.write(uploaded_model.getbuffer())
                model_tmp_path = tmp_model.name

            with st.spinner("⏳ Loading model..."):
                model, ok = load_denoising_model(model_tmp_path)
                if not ok:
                    st.error(f"❌ Failed to load model: {model}")
                    st.stop()
                st.success("✅ Model successfully loaded!")

            # Ekstrak ZIP bila perlu
            extracted_dir = None
            path_list = None
            if input_mode == "zip":
                try:
                    extracted_dir = tempfile.mkdtemp(prefix="noisy_zip_")
                    with zipfile.ZipFile(zip_folder, "r") as z:
                        z.extractall(extracted_dir)
                    path_list = list_ima_dcm_recursive(extracted_dir)
                    if not path_list:
                        st.error("❌ No .IMA/.DCM file found in the uploaded ZIP!")
                        st.stop()
                    st.success(f"✅ Found {len(path_list)} files in ZIP")
                except Exception as e:
                    st.error(f"❌ Failed to extract ZIP: {e}")
                    st.stop()

            # Siapkan ZIP buffer
            zip_buffer = io.BytesIO()
            zf = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED)

            # Progress
            total = len(input_file_items) if input_mode == "uploaded_files" else len(path_list)
            progress_bar = st.progress(0)
            status_text = st.empty()

            success_count, failed_count = 0, 0
            failed_files = []

            def process_one(img_norm, dcm_obj, fname):
                try:
