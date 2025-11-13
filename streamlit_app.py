import os
import io
import zipfile
import tempfile
from pathlib import Path
import gc

import numpy as np
import pydicom
import streamlit as st
import tensorflow as tf
from PIL import Image  # ikon & logo

APP_DIR = Path(__file__).parent

# 1) Tentukan path aset
FAVICON = APP_DIR / "assets" / "TPU_new_logo_en.png"   # ikon tab
HEADER_IMG = APP_DIR / "assets" / "Logo_TPU.jpg"  # gambar di atas judul
# Opsi fallback jika folder assets belum dipakai
if not FAVICON.exists():
    alt = APP_DIR / "TPU_new_logo_en.png"
    FAVICON = alt if alt.exists() else None
if not HEADER_IMG.exists():
    alt = APP_DIR / "Logo_TPU.jpg"
    HEADER_IMG = alt if alt.exists() else None

# 2) WAJIB: set konfigurasi halaman paling awal
st.set_page_config(
    page_title="Low-Dose CT Medical Image Denoising App",
    page_icon=Image.open(FAVICON) if FAVICON and FAVICON.exists() else "🏥",
    layout="wide",
)  # [web:165][web:95]

# 3) Render gambar header di atas judul (jika ada)
if HEADER_IMG and HEADER_IMG.exists():
    left, mid, right = st.columns([1, 2, 1])  # pusatkan
    with mid:
        st.image(str(HEADER_IMG), use_container_width=True)  # [web:106]
st.title("Low-Dose CT Medical Image Denoising (IMA/DICOM)")
st.markdown("---")

# =========================
# Sidebar: Input
# =========================
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("📁 Input Noisy Images")
uploaded_imas = st.sidebar.file_uploader(
    "Upload noisy IMA/DICOM files (multi-select supported)",
    type=["ima", "dcm"],
    accept_multiple_files=True,
    help="Select multiple files at once (Ctrl/Shift click) for batch upload.",
)

zip_folder = st.sidebar.file_uploader(
    "Or upload the folder as .zip",
    type=["zip"],
    help="Upload the ZIP folder to process its entire contents.",
)

st.sidebar.subheader("🤖 Model File")
uploaded_model = st.sidebar.file_uploader(
    "Upload model (.h5)",
    type=["h5"],
    help="Upload the Keras/TensorFlow model file in .h5 format.",
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
    """
    Tulis DICOM ke buffer memori (file-like), siap dimasukkan ke ZIP.
    write_like_original=False memastikan meta DICOM standar.
    """
    denoised_scaled = (denoised_array * 255.0).astype(np.uint16)
    dcm_out = original_dcm.copy()
    dcm_out.PixelData = denoised_scaled.tobytes()
    mem = io.BytesIO()
    dcm_out.save_as(mem, write_like_original=False)
    mem.seek(0)
    return mem.getvalue()

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
            if uploaded_imas and len(uploaded_imas) > 0:
                input_mode = "uploaded_files"
                input_items = uploaded_imas
            elif zip_folder is not None:
                input_mode = "zip"
                input_items = None
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
            else:
                path_list = None

            # Siapkan ZIP di DISK (hindari konsumsi RAM besar)
            zip_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            zip_tmp_path = zip_tmp.name
            zip_tmp.close()
            zf = zipfile.ZipFile(zip_tmp_path, "w", zipfile.ZIP_DEFLATED)

            # Progress
            total = len(input_items) if input_mode == "uploaded_files" else len(path_list)
            progress_bar = st.progress(0)
            status_text = st.empty()

            success_count, failed_count = 0, 0
            failed_files = []

            def process_one(img_norm, dcm_obj, fname):
                try:
                    img_input = np.expand_dims(img_norm, axis=(0, -1))
                    denoised = model.predict(img_input, verbose=0)[0, :, :, 0]
                    dbytes = dicom_bytes(dcm_obj, denoised)
                    zf.writestr(fname, dbytes)
                    return 1, 0, None
                except Exception as e:
                    return 0, 1, (fname, str(e))

            # Proses
            if input_mode == "uploaded_files":
                for idx, f in enumerate(input_items):
                    fname = f.name
                    status_text.text(f"Denoising: {fname} ({idx + 1}/{total})")
                    img_norm, dcm, okread = read_dicom_from_bytes(f.getbuffer())
                    if not okread:
                        failed_count += 1
                        failed_files.append((fname, img_norm))
                        progress_bar.progress((idx + 1) / total)
                        continue
                    s_inc, f_inc, fail_rec = process_one(img_norm, dcm, fname)
                    success_count += s_inc
                    failed_count += f_inc
                    if fail_rec:
                        failed_files.append(fail_rec)
                    progress_bar.progress((idx + 1) / total)
            else:
                for idx, p in enumerate(path_list):
                    fname = os.path.basename(p)
                    status_text.text(f"Denoising: {fname} ({idx + 1}/{total})")
                    img_norm, dcm, okread = read_dicom_from_path(p)
                    if not okread:
                        failed_count += 1
                        failed_files.append((fname, img_norm))
                        progress_bar.progress((idx + 1) / total)
                        continue
                    s_inc, f_inc, fail_rec = process_one(img_norm, dcm, fname)
                    success_count += s_inc
                    failed_count += f_inc
                    if fail_rec:
                        failed_files.append(fail_rec)
                    progress_bar.progress((idx + 1) / total)

            # Tutup ZIP & siapkan link unduh via path (tanpa memuat ke memori)
            zf.close()

            # Bebaskan memori model & grafik TF untuk menstabilkan proses unduhan
            try:
                del model
                tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()

            # Simpan path ZIP agar persist saat rerun dan JANGAN hapus dulu
            st.session_state["denoised_zip_path"] = zip_tmp_path

            # Ringkasan
            st.markdown("---")
            st.header("📊 Resume")
            c_ok, c_fail = st.columns(2)
            with c_ok:
                st.metric("✅ Successfully", success_count)
            with c_fail:
                st.metric("❌ Failed", failed_count)
            if failed_files:
                st.subheader("⚠️ Failed to denoise:")
                for fname, err in failed_files:
                    st.error(f"{fname}: {err}")

# Tampilkan tombol download: selalu buka ulang file tiap render
if "denoised_zip_path" in st.session_state and os.path.exists(st.session_state["denoised_zip_path"]):
    # Hindari with ... yang segera menutup handle; biarkan Streamlit membaca selama request
    fzip = open(st.session_state["denoised_zip_path"], "rb")
    st.download_button(
        "📦 Download denoised_results.zip",
        data=fzip,  # file-like; Streamlit mengalirkan konten saat klik
        file_name="denoised_results.zip",
        mime="application/zip",
        key="download_denoised_zip",
        use_container_width=True,
    )
    # Tampilkan tombol bersih manual setelah unduhan selesai
    def _cleanup():
        try:
            fzip.close()
        except Exception:
            pass
        p = st.session_state.get("denoised_zip_path")
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
        st.session_state.pop("denoised_zip_path", None)
    st.button("🧹 Delete temporary ZIP", on_click=_cleanup)

with col2:
    st.header("ℹ️ Information")
    info = st.container()
    with info:
        if uploaded_imas and len(uploaded_imas) > 0:
            st.info(f"📄 Number of uploaded files: {len(uploaded_imas)}")
            st.subheader("📋 Preview file list:")
            for f in uploaded_imas[:5]:
                st.text(f"• {f.name}")
            if len(uploaded_imas) > 5:
                st.text(f"... and {len(uploaded_imas) - 5} others")
        elif zip_folder is not None:
            st.info("📦 ZIP uploaded (will be extracted during the process)")

        if uploaded_model is not None:
            model_size_mb = uploaded_model.size / (1024 * 1024)
            st.info(f"🤖 Model: {uploaded_model.name}")
            st.info(f"📊 Model Size: {model_size_mb:.2f} MB")

st.markdown("---")
st.markdown("To evaluate the quality of denoised images, you can use evaluation metrics in this apps: https://metreval.streamlit.app/")




