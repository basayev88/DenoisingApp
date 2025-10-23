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

# Opsi tujuan output
st.sidebar.subheader("📤 Tujuan Output")
output_mode = st.sidebar.radio(
    "Pilih mode output:",
    options=["Server folder", "Download ZIP"],
    horizontal=False,
    help="Aplikasi web tidak dapat menyimpan langsung ke folder lokal pengguna; gunakan Download ZIP untuk memilih lokasi simpan di perangkat Anda.",
)

output_folder = ""
if output_mode == "Server folder":
    output_folder = st.sidebar.text_input(
        "Nama/Path folder output (server-side):",
        placeholder="Example: UNet Denoised Results",
    )

# =========================
# Helpers
# =========================
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

def save_dicom_to_disk(original_dcm, denoised_array: np.ndarray, save_path: str):
    try:
        denoised_scaled = (denoised_array * 255.0).astype(np.uint16)
        dcm_out = original_dcm.copy()
        dcm_out.PixelData = denoised_scaled.tobytes()
        ensure_dir(os.path.dirname(save_path))
        dcm_out.save_as(save_path)
        return True, "Saved"
    except Exception as e:
        return False, str(e)

def dicom_bytes(original_dcm, denoised_array: np.ndarray) -> bytes:
    """Return DICOM bytes for inclusion in ZIP."""
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
    out_ready = (output_mode == "Download ZIP") or (
        output_mode == "Server folder" and bool(output_folder)
    )

    start_disabled = not (files_ready and model_ready and out_ready)

    if st.button("🚀 Start Denoising Process", type="primary", disabled=start_disabled):
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

            # Validasi model
            if uploaded_model is None:
                st.error("❌ Model file not uploaded!")
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

            # Siapkan output
            if output_mode == "Server folder":
                ensure_dir(output_folder)

            # Siapkan ZIP buffer bila mode unduh
            zip_buffer = None
            if output_mode == "Download ZIP":
                zip_buffer = io.BytesIO()
                zf = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED)

            # Progress
            total = len(input_file_items) if input_mode == "uploaded_files" else len(path_list)
            progress_bar = st.progress(0)
            status_text = st.empty()

            success_count, failed_count = 0, 0
            failed_files = []

            # Proses
            def process_one(img_norm, dcm_obj, fname):
                nonlocal success_count, failed_count, failed_files
                try:
                    img_input = np.expand_dims(img_norm, axis=(0, -1))
                    denoised = model.predict(img_input, verbose=0)[0, :, :, 0]
                    if output_mode == "Server folder":
                        out_path = os.path.join(output_folder, fname)
                        ok_save, msg = save_dicom_to_disk(dcm_obj, denoised, out_path)
                        if ok_save:
                            success_count += 1
                            st.success(f"✅ {fname} saved to server")
                        else:
                            failed_count += 1
                            failed_files.append((fname, msg))
                            st.error(f"❌ Failed to save {fname}: {msg}")
                    else:
                        dbytes = dicom_bytes(dcm_obj, denoised)
                        zf.writestr(fname, dbytes)
                        success_count += 1
                        st.success(f"✅ {fname} added to ZIP")
                except Exception as e:
                    failed_count += 1
                    failed_files.append((fname, str(e)))
                    st.error(f"❌ Error processing {fname}: {e}")

            if input_mode == "uploaded_files":
                for idx, f in enumerate(input_file_items):
                    fname = f.name
                    status_text.text(f"Denoising: {fname} ({idx + 1}/{total})")
                    img_norm, dcm, okread = read_dicom_from_bytes(f.getbuffer())
                    if not okread:
                        failed_count += 1
                        failed_files.append((fname, img_norm))
                        st.warning(f"⚠️ Failed to read {fname}: {img_norm}")
                        progress_bar.progress((idx + 1) / total)
                        continue
                    process_one(img_norm, dcm, fname)
                    progress_bar.progress((idx + 1) / total)
            else:
                for idx, p in enumerate(path_list):
                    fname = os.path.basename(p)
                    status_text.text(f"Denoising: {fname} ({idx + 1}/{total})")
                    img_norm, dcm, okread = read_dicom_from_path(p)
                    if not okread:
                        failed_count += 1
                        failed_files.append((fname, img_norm))
                        st.warning(f"⚠️ Failed to read {fname}: {img_norm}")
                        progress_bar.progress((idx + 1) / total)
                        continue
                    process_one(img_norm, dcm, fname)
                    progress_bar.progress((idx + 1) / total)

            # Tutup ZIP jika dibuat
            if output_mode == "Download ZIP" and zip_buffer is not None:
                zf.close()
                zip_buffer.seek(0)

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
                    st.error(f"{fname}: {err}")

            if success_count > 0 and output_mode == "Server folder":
                st.success(f"🎉 Done! Saved {success_count} files to: {output_folder}")
            if success_count > 0 and output_mode == "Download ZIP":
                st.download_button(
                    "📦 Download denoised.zip",
                    data=zip_buffer,
                    file_name="denoised_results.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

with col2:
    st.header("ℹ️ Information")
    info = st.container()
    with info:
        # Info jumlah file input
        if uploaded_imas and len(uploaded_imas) > 0:
            st.info(f"📄 Number of uploaded files: {len(uploaded_imas)}")
            st.subheader("📋 Preview file list:")
            for f in uploaded_imas[:5]:
                st.text(f"• {f.name}")
            if len(uploaded_imas) > 5:
                st.text(f"... and {len(uploaded_imas) - 5} others")
        elif zip_folder is not None:
            st.info("📦 ZIP uploaded (akan diekstrak saat proses)")

        # Info model
        if uploaded_model is not None:
            model_size_mb = uploaded_model.size / (1024 * 1024)
            st.info(f"🤖 Model: {uploaded_model.name}")
            st.info(f"📊 Model Size: {model_size_mb:.2f} MB")

        # Info output
        st.info(f"📤 Output mode: {output_mode}")
        if output_mode == "Server folder" and output_folder:
            st.info(f"💾 Output Folder (server): {output_folder}")

# Footer
st.markdown("---")
