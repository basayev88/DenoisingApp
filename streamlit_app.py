import os
import io
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pydicom
import streamlit as st
import tensorflow as tf

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="Low-Dose CT Medical Image Denoising App",
    page_icon="🏥",
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
    help="Pilih banyak berkas sekaligus (Ctrl/Shift klik) untuk mengunggah batch.",
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

# Output folder (server-side)
st.sidebar.subheader("💾 Folder Output")
output_folder = st.sidebar.text_input(
    "Type output folder name:",
    placeholder="Example: UNet Denoised Results",
)

# =========================
# State & Helpers
# =========================
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "total_files" not in st.session_state:
    st.session_state.total_files = 0

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

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
        img = dcm.pixel_array.astype(np.float32)
        img = img / 255.0
        return img, dcm, True
    except Exception as e:
        return str(e), None, False

def read_dicom_from_bytes(b: bytes):
    try:
        dcm = pydicom.dcmread(io.BytesIO(b))
        img = dcm.pixel_array.astype(np.float32)
        img = img / 255.0
        return img, dcm, True
    except Exception as e:
        return str(e), None, False

def save_dicom(original_dcm, denoised_array: np.ndarray, save_path: str):
    try:
        denoised_scaled = (denoised_array * 255.0).astype(np.uint16)
        dcm_out = original_dcm.copy()
        dcm_out.PixelData = denoised_scaled.tobytes()
        ensure_dir(os.path.dirname(save_path))
        dcm_out.save_as(save_path)
        return True, "Successfully saved"
    except Exception as e:
        return False, str(e)

# =========================
# UI: Processing & Info
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Processing Status")
    validation_status = st.container()

    files_ready = (uploaded_imas and len(uploaded_imas) > 0) or (zip_folder is not None)
    model_ready = uploaded_model is not None
    output_ready = bool(output_folder)

    start_disabled = not (files_ready and model_ready and output_ready)

    if st.button(
        "🚀 Start Denoising Process",
        type="primary",
        disabled=start_disabled,
    ):
        with validation_status:
            # Validasi file input
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

            # Validasi model
            if uploaded_model is None:
                st.error("❌ Model file not uploaded!")
                st.stop()

            # Siapkan model ke file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model:
                tmp_model.write(uploaded_model.getbuffer())
                model_tmp_path = tmp_model.name

            with st.spinner("⏳ Loading model..."):
                model, model_loaded = load_denoising_model(model_tmp_path)
                if not model_loaded:
                    st.error(f"❌ Failed to load model: {model}")
                    st.stop()
                else:
                    st.success("✅ Model successfully loaded!")

            # Siapkan input dari ZIP (jika dipakai)
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

            # Siapkan output folder
            ensure_dir(output_folder)
            st.info(f"📁 Output folder: {output_folder}")

            # Progress
            if input_mode == "uploaded_files":
                total = len(input_file_items)
            else:
                total = len(path_list)

            progress_bar = st.progress(0)
            status_text = st.empty()

            success_count = 0
            failed_count = 0
            failed_files = []

            # Proses
            if input_mode == "uploaded_files":
                for idx, f in enumerate(input_file_items):
                    fname = f.name
                    status_text.text(f"Denoising: {fname} ({idx + 1}/{total})")

                    img_norm, dcm, ok = read_dicom_from_bytes(f.getbuffer())
                    if not ok:
                        failed_count += 1
                        failed_files.append((fname, img_norm))
                        st.warning(f"⚠️ Failed to read {fname}: {img_norm}")
                        progress_bar.progress((idx + 1) / total)
                        continue

                    try:
                        img_input = np.expand_dims(img_norm, axis=(0, -1))
                        denoised = model.predict(img_input, verbose=0)[0, :, :, 0]
                        out_path = os.path.join(output_folder, fname)
                        save_ok, save_msg = save_dicom(dcm, denoised, out_path)
                        if save_ok:
                            success_count += 1
                            st.success(f"✅ {fname} successfully denoised")
                        else:
                            failed_count += 1
                            failed_files.append((fname, save_msg))
                            st.error(f"❌ Failed to save {fname}: {save_msg}")
                    except Exception as e:
                        failed_count += 1
                        failed_files.append((fname, str(e)))
                        st.error(f"❌ Error processing {fname}: {e}")

                    progress_bar.progress((idx + 1) / total)

            else:  # input_mode == "zip"
                for idx, p in enumerate(path_list):
                    fname = os.path.basename(p)
                    status_text.text(f"Denoising: {fname} ({idx + 1}/{total})")

                    img_norm, dcm, ok = read_dicom_from_path(p)
                    if not ok:
                        failed_count += 1
                        failed_files.append((fname, img_norm))
                        st.warning(f"⚠️ Failed to read {fname}: {img_norm}")
                        progress_bar.progress((idx + 1) / total)
                        continue

                    try:
                        img_input = np.expand_dims(img_norm, axis=(0, -1))
                        denoised = model.predict(img_input, verbose=0)[0, :, :, 0]
                        out_path = os.path.join(output_folder, fname)
                        save_ok, save_msg = save_dicom(dcm, denoised, out_path)
                        if save_ok:
                            success_count += 1
                            st.success(f"✅ {fname} successfully denoised")
                        else:
                            failed_count += 1
                            failed_files.append((fname, save_msg))
                            st.error(f"❌ Failed to save {fname}: {save_msg}")
                    except Exception as e:
                        failed_count += 1
                        failed_files.append((fname, str(e)))
                        st.error(f"❌ Error processing {fname}: {e}")

                    progress_bar.progress((idx + 1) / total)

            # Summary
            st.markdown("---")
            st.header("📊 Resume")
            col_success, col_failed = st.columns(2)
            with col_success:
                st.metric("✅ Successfully", success_count)
            with col_failed:
                st.metric("❌ Failed", failed_count)

            if failed_files:
                st.subheader("⚠️ Failed to denoise:")
                for fname, err in failed_files:
                    st.error(f"**{fname}**: {err}")

            if success_count > 0:
                st.success(
                    f"🎉 Denoising done! {success_count} files successfully saved to: **{output_folder}**"
                )

with col2:
    st.header("ℹ️ Information")
    info = st.container()
    with info:
        # Info jumlah file input
        if uploaded_imas and len(uploaded_imas) > 0:
            st.info(f"📄 **Number of uploaded files:** {len(uploaded_imas)}")
            st.subheader("📋 Preview file list:")
            for f in uploaded_imas[:5]:
                st.text(f"• {f.name}")
            if len(uploaded_imas) > 5:
                st.text(f"... and {len(uploaded_imas) - 5} others")
        elif zip_folder is not None:
            st.info("📦 **ZIP uploaded** (akan diekstrak saat proses)")

        # Info model
        if uploaded_model is not None:
            model_size_mb = uploaded_model.size / (1024 * 1024)
            st.info(f"🤖 **Model:** {uploaded_model.name}")
            st.info(f"📊 **Model Size:** {model_size_mb:.2f} MB")

        # Info output
        if output_folder:
            st.info(f"💾 **Output Folder:** {output_folder}")

# Footer
st.markdown("---")
