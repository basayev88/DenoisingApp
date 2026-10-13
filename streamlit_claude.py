import os
import io
import json
import zipfile
import tempfile
from pathlib import Path
import gc

import cv2
import numpy as np
import pydicom
import streamlit as st
from PIL import Image  # ikon & logo
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

# TensorFlow di-import secara "lazy" (hanya saat menu Denoising dijalankan),
# sehingga menu Deteksi Tepi tetap ringan dan cepat dibuka.

APP_DIR = Path(__file__).parent

# 1) Tentukan path aset
FAVICON = APP_DIR / "assets" / "TPU_new_logo_en.png"  # ikon tab
HEADER_IMG = APP_DIR / "assets" / "Logo_TPU.jpg"      # gambar di atas judul
# Opsi fallback jika folder assets belum dipakai
if not FAVICON.exists():
    alt = APP_DIR / "TPU_new_logo_en.png"
    FAVICON = alt if alt.exists() else None
if not HEADER_IMG.exists():
    alt = APP_DIR / "Logo_TPU.jpg"
    HEADER_IMG = alt if alt.exists() else None

# 2) WAJIB: set konfigurasi halaman paling awal
st.set_page_config(
    page_title="Low-Dose CT Denoising & Edge Detection App",
    page_icon=Image.open(FAVICON) if FAVICON and FAVICON.exists() else "🏥",
    layout="wide",
)

# 3) Render gambar header di atas judul (jika ada)
if HEADER_IMG and HEADER_IMG.exists():
    left, mid, right = st.columns([1, 2, 1])  # pusatkan
    with mid:
        st.image(str(HEADER_IMG), use_container_width=True)


# =========================================================
# Helpers umum
# =========================================================
DICOM_EXT = (".ima", ".dcm")


def list_ima_dcm_recursive(root_dir: str):
    paths = []
    for r, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(DICOM_EXT):
                paths.append(os.path.join(r, name))
    paths.sort()
    return paths


def collect_sources(uploaded_files, zip_file, zip_path=None):
    """
    Satukan semua sumber input menjadi daftar (nama, fungsi_baca_bytes).
    Bytes dibaca hanya saat dibutuhkan agar hemat memori untuk batch besar.
    """
    items = []
    if uploaded_files:
        for f in uploaded_files:
            items.append((f.name, f.getvalue))
        return items

    zsrc = zip_file if zip_file is not None else zip_path
    if zsrc is not None:
        with zipfile.ZipFile(zsrc, "r") as z:
            names = sorted(
                n for n in z.namelist()
                if n.lower().endswith(DICOM_EXT) and not n.startswith("__MACOSX")
            )

        def make_reader(name):
            def _read():
                with zipfile.ZipFile(zsrc, "r") as zz:
                    return zz.read(name)
            return _read

        for n in names:
            items.append((n, make_reader(n)))
    return items


def ensure_uncompressed(ds):
    """Data terkompresi perlu didekompres dulu agar PixelData bisa ditulis ulang."""
    try:
        if hasattr(ds, "file_meta") and ds.file_meta.TransferSyntaxUID.is_compressed:
            ds.decompress()
    except Exception:
        pass
    return ds


def dataset_to_bytes(ds) -> bytes:
    """Tulis dataset DICOM ke buffer memori dengan meta DICOM standar (pydicom 2.x & 3.x)."""
    mem = io.BytesIO()
    try:
        ds.save_as(mem, enforce_file_format=True)       # pydicom >= 3.0
    except TypeError:
        ds.save_as(mem, write_like_original=False)      # pydicom 2.x
    mem.seek(0)
    return mem.getvalue()


def window_to_uint8(arr, ds, window=None, use_header_window=True):
    """
    Terapkan rescale (slope/intercept -> HU) + windowing, kembalikan citra 8-bit.
    Dipakai untuk pratinjau (denoising) dan sebagai input deteksi tepi Canny.
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[-1] == 3:          # sudah RGB
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if arr.ndim == 3:                                  # multi-frame
        arr = arr[0]

    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept

    if window is not None:
        wc, ww = float(window[0]), float(window[1])
    elif use_header_window:
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
    else:
        wc = ww = None

    if wc is not None and ww is not None:
        wc = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
        ww = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)
        lo, hi = wc - ww / 2.0, wc + ww / 2.0
    else:
        lo, hi = np.percentile(arr, (0.5, 99.5))

    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max()) or 1.0

    out = np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0

    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        out = 255.0 - out       # putih = nilai rendah, perlu dibalik

    return out.astype(np.uint8)


def dicom_to_uint8(ds, window=None, use_header_window=True):
    return window_to_uint8(ds.pixel_array, ds, window, use_header_window)


# =========================================================
# Helpers: Denoising (mengikuti Denoising_Script_fixed)
# =========================================================
@st.cache_resource
def load_denoising_model(model_path: str):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, True
    except Exception as e:
        return str(e), False


def read_max_val_from_json(uploaded_json):
    """Baca max_val dari <nama_model>_normalization.json yang disimpan saat training."""
    data = json.loads(uploaded_json.getvalue().decode("utf-8"))
    return float(data["max_val"])


def read_dicom_normalized(b: bytes, max_val: float):
    """Baca DICOM lalu normalisasi ke [0,1] memakai basis yang SAMA dengan saat training."""
    try:
        dcm = pydicom.dcmread(io.BytesIO(b))
        ensure_uncompressed(dcm)
        img = dcm.pixel_array.astype(np.float32)
        img = np.clip(img / max_val, 0.0, 1.0)
        return img, dcm, True
    except Exception as e:
        return str(e), None, False


def denoised_raw(original_dcm, denoised_array: np.ndarray, max_val: float) -> np.ndarray:
    """Kembalikan hasil denoising ke skala piksel CT asli (clip dulu agar tidak wraparound)."""
    denoised_clipped = np.clip(denoised_array, 0.0, 1.0)
    out_dtype = original_dcm.pixel_array.dtype
    if not np.issubdtype(out_dtype, np.integer):
        out_dtype = np.uint16
    return (denoised_clipped * max_val).astype(out_dtype)


def dicom_bytes(original_dcm, denoised_scaled: np.ndarray) -> bytes:
    """Tulis DICOM hasil denoising ke buffer memori, siap dimasukkan ke ZIP."""
    dcm_out = original_dcm.copy()
    dcm_out.PixelData = np.ascontiguousarray(denoised_scaled).tobytes()
    return dataset_to_bytes(dcm_out)


# =========================================================
# Helpers: Deteksi Tepi Canny (mengikuti dicom_edge_batch)
# =========================================================
def detect_edges(gray, blur=5, low=55, high=150, thickness=2, min_length=30, l2=True):
    """Kembalikan mask tepi (uint8, bernilai 0 atau 255)."""
    blur = blur if blur % 2 == 1 else blur + 1
    smooth = cv2.GaussianBlur(gray, (blur, blur), 0) if blur > 1 else gray

    edges = cv2.Canny(smooth, low, high, L2gradient=l2)

    # sambung tepi yang sedikit terputus
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # buang serpihan tepi pendek (sisa noise)
    if min_length > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_length
        edges = np.where(keep[labels], 255, 0).astype(np.uint8)

    if thickness > 1:
        edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8))

    return edges


def make_overlay(gray, edges, color_rgb):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb[edges > 0] = color_rgb
    return rgb


def mark_as_derived(ds, description):
    """Tandai citra sebagai turunan supaya tidak tertukar dengan citra asli."""
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    if hasattr(ds, "file_meta"):
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.ImageType = ["DERIVED", "SECONDARY", "EDGE"]
    ds.SeriesDescription = description
    if "SeriesNumber" in ds and ds.SeriesNumber is not None:
        try:
            ds.SeriesNumber = int(ds.SeriesNumber) + 900
        except (TypeError, ValueError):
            pass


def write_color(ds, gray, edges, color_rgb):
    """Simpan sebagai DICOM RGB 8-bit dengan garis berwarna."""
    rgb = make_overlay(gray, edges, color_rgb)

    ds.PhotometricInterpretation = "RGB"
    ds.SamplesPerPixel = 3
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.Rows, ds.Columns = rgb.shape[:2]

    # tag khusus grayscale tidak lagi berlaku untuk data RGB
    for tag in ("RescaleSlope", "RescaleIntercept", "RescaleType",
                "WindowCenter", "WindowWidth", "WindowCenterWidthExplanation",
                "VOILUTSequence", "SmallestImagePixelValue",
                "LargestImagePixelValue", "NumberOfFrames"):
        if tag in ds:
            delattr(ds, tag)

    ds.PixelData = np.ascontiguousarray(rgb).tobytes()
    mark_as_derived(ds, "Edge overlay (color)")
    return ds


def write_gray(ds, edges):
    """Simpan tetap MONOCHROME2, garis digambar dengan nilai maksimum."""
    arr = ds.pixel_array
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.copy()

    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        line_value = arr.min()
    else:
        bits = int(getattr(ds, "BitsStored", 16))
        line_value = arr.max() if arr.max() > 0 else (2 ** bits - 1)

    arr[edges > 0] = line_value
    ds.PixelData = np.ascontiguousarray(arr).tobytes()
    if "NumberOfFrames" in ds:
        del ds.NumberOfFrames
    ds.Rows, ds.Columns = arr.shape[:2]
    mark_as_derived(ds, "Edge overlay (gray)")
    return ds


def edge_process_bytes(b: bytes, p: dict):
    """Baca satu DICOM (bytes), deteksi tepi, kembalikan (bytes_dicom_hasil, gray, edges)."""
    ds = pydicom.dcmread(io.BytesIO(b))
    ensure_uncompressed(ds)

    gray = dicom_to_uint8(ds, window=p["window"], use_header_window=p["use_header"])
    edges = detect_edges(gray, blur=p["blur"], low=p["low"], high=p["high"],
                         thickness=p["thickness"], min_length=p["min_length"],
                         l2=p["l2"])

    ds = write_color(ds, gray, edges, p["color"]) if p["mode"] == "color" else write_gray(ds, edges)

    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    try:
        ds.is_little_endian = True
        ds.is_implicit_VR = False
    except Exception:
        pass
    return dataset_to_bytes(ds), gray, edges


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def show_zip_download(state_key: str, label: str, file_name: str, key: str):
    """Tombol unduh ZIP + tombol hapus ZIP sementara (file dibuka ulang tiap render)."""
    p = st.session_state.get(state_key)
    if not (p and os.path.exists(p)):
        return
    with open(p, "rb") as fzip:
        st.download_button(
            label,
            data=fzip.read(),
            file_name=file_name,
            mime="application/zip",
            key=key,
            use_container_width=True,
        )

    def _cleanup():
        q = st.session_state.get(state_key)
        if q and os.path.exists(q):
            try:
                os.remove(q)
            except Exception:
                pass
        st.session_state.pop(state_key, None)

    st.button("🧹 Delete temporary ZIP", on_click=_cleanup, key=f"cleanup_{key}")


# State
for k, v in (("preview_pairs", []), ("edge_previews", [])):
    if k not in st.session_state:
        st.session_state[k] = v


# =========================================================
# Sidebar: Menu
# =========================================================
st.sidebar.header("🧭 Menu")
menu = st.sidebar.radio(
    "Select process",
    ["🧹 Denoising (RED-CNN)", "✏️ Edge Detection (Canny)"],
    key="menu",
)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")


# =========================================================
# MENU 1: DENOISING
# =========================================================
def page_denoising():
    st.title("Low-Dose CT Medical Image Denoising (IMA/DICOM)")
    st.markdown("---")

    st.sidebar.subheader("📁 Input Noisy Images")
    uploaded_imas = st.sidebar.file_uploader(
        "Upload noisy IMA/DICOM files (multi-select supported)",
        type=["ima", "dcm"],
        accept_multiple_files=True,
        help="Select multiple files at once (Ctrl/Shift click) for batch upload.",
        key="dn_files",
    )
    zip_folder = st.sidebar.file_uploader(
        "Or upload the folder as .zip",
        type=["zip"],
        help="Upload the ZIP folder to process its entire contents.",
        key="dn_zip",
    )

    st.sidebar.subheader("🤖 Model File")
    uploaded_model = st.sidebar.file_uploader(
        "Upload model (.h5)",
        type=["h5"],
        help="Upload the Keras/TensorFlow model file in .h5 format.",
        key="dn_model",
    )

    st.sidebar.subheader("📏 Normalization Basis (max_val)")
    norm_source = st.sidebar.radio(
        "Source of max_val",
        ["JSON file from training (recommended)", "Auto from BitsStored (fallback)", "Manual value"],
        help="The scale used at inference must be IDENTICAL to the scale used when "
             "the model was trained. Use the '<model>_normalization.json' file saved "
             "next to the model during training.",
        key="dn_norm_source",
    )
    uploaded_json = None
    manual_max = None
    if norm_source.startswith("JSON"):
        uploaded_json = st.sidebar.file_uploader(
            "Upload <model>_normalization.json", type=["json"], key="dn_json")
    elif norm_source.startswith("Manual"):
        manual_max = st.sidebar.number_input(
            "max_val", min_value=1.0, value=4095.0, step=1.0, key="dn_manual_max",
            help="Example: 4095 for 12-bit CT data. Only use this if you know the value used in training.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📋 Processing Status")
        validation_status = st.container()

    files_ready = (uploaded_imas and len(uploaded_imas) > 0) or (zip_folder is not None)
    model_ready = uploaded_model is not None
    norm_ready = (not norm_source.startswith("JSON")) or (uploaded_json is not None)
    start_disabled = not (files_ready and model_ready and norm_ready)

    with col1:
        if files_ready and model_ready and not norm_ready:
            st.warning("⚠️ Upload the normalization JSON file, or choose another max_val source.")

    if st.button("🚀 Start Denoising and Prepare ZIP", type="primary", disabled=start_disabled):
        st.session_state["preview_pairs"] = []

        with validation_status:
            # Tentukan sumber input
            try:
                items = collect_sources(uploaded_imas, zip_folder)
            except Exception as e:
                st.error(f"❌ Failed to read ZIP: {e}")
                st.stop()
            if not items:
                st.error("❌ No .IMA/.DCM file found in the input!")
                st.stop()
            if zip_folder is not None and not uploaded_imas:
                st.success(f"✅ Found {len(items)} files in ZIP")

            # Tentukan basis normalisasi
            try:
                if uploaded_json is not None:
                    max_val = read_max_val_from_json(uploaded_json)
                    st.success(f"✅ Normalization basis read from JSON: max_val = {max_val:g}")
                elif manual_max is not None:
                    max_val = float(manual_max)
                    st.info(f"ℹ️ Using manual normalization basis: max_val = {max_val:g}")
                else:
                    sample = pydicom.dcmread(io.BytesIO(items[0][1]()), stop_before_pixels=True)
                    max_val = float((2 ** int(sample.BitsStored)) - 1)
                    st.warning(
                        f"⚠️ Guessing max_val = {max_val:g} from BitsStored of the input image. "
                        "Verify that this is the same basis used when training the model!")
            except Exception as e:
                st.error(f"❌ Failed to determine max_val: {e}")
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

            # Siapkan ZIP di DISK (hindari konsumsi RAM besar)
            zip_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            zip_tmp_path = zip_tmp.name
            zip_tmp.close()

            total = len(items)
            progress_bar = st.progress(0)
            status_text = st.empty()
            success_count, failed_files = 0, []

            with zipfile.ZipFile(zip_tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, (name, reader) in enumerate(items):
                    fname = os.path.basename(name)
                    status_text.text(f"Denoising: {fname} ({idx + 1}/{total})")
                    try:
                        img_norm, dcm, okread = read_dicom_normalized(reader(), max_val)
                        if not okread:
                            raise RuntimeError(img_norm)

                        img_input = np.expand_dims(img_norm, axis=(0, -1))
                        denoised = model.predict(img_input, verbose=0)[0, :, :, 0]
                        den_raw = denoised_raw(dcm, denoised, max_val)
                        zf.writestr(fname, dicom_bytes(dcm, den_raw))

                        # simpan maksimal 5 pasangan untuk preview (windowing dari header)
                        if len(st.session_state["preview_pairs"]) < 5:
                            noisy_raw = np.clip(img_norm, 0, 1) * max_val
                            st.session_state["preview_pairs"].append({
                                "fname": fname,
                                "noisy": Image.fromarray(window_to_uint8(noisy_raw, dcm)),
                                "denoised": Image.fromarray(window_to_uint8(den_raw, dcm)),
                            })
                        success_count += 1
                    except Exception as e:
                        failed_files.append((fname, str(e)))
                    progress_bar.progress((idx + 1) / total)

            status_text.text("Done.")

            # Bebaskan memori model & grafik TF untuk menstabilkan proses unduhan
            try:
                import tensorflow as tf
                del model
                tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()

            # hapus ZIP lama (jika ada) lalu simpan path ZIP baru
            old = st.session_state.get("denoised_zip_path")
            if old and old != zip_tmp_path and os.path.exists(old):
                try:
                    os.remove(old)
                except Exception:
                    pass
            st.session_state["denoised_zip_path"] = zip_tmp_path
            st.session_state["denoised_summary"] = (success_count, failed_files)

    # Ringkasan & preview (tetap tampil setelah rerun)
    if "denoised_summary" in st.session_state and st.session_state.get("denoised_zip_path"):
        success_count, failed_files = st.session_state["denoised_summary"]
        st.markdown("---")
        st.header("📊 Resume")
        c_ok, c_fail = st.columns(2)
        c_ok.metric("✅ Successfully", success_count)
        c_fail.metric("❌ Failed", len(failed_files))
        if failed_files:
            st.subheader("⚠️ Failed to denoise:")
            for fname, err in failed_files:
                st.error(f"{fname}: {err}")

        if st.session_state.get("preview_pairs"):
            st.markdown("---")
            st.header("🔍 Preview Noisy vs Denoised (maksimal 5 contoh)")
            for pair in st.session_state["preview_pairs"]:
                st.subheader(pair["fname"])
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Before (Noisy)")
                    st.image(pair["noisy"], use_container_width=True)
                with c2:
                    st.caption("After (Denoised)")
                    st.image(pair["denoised"], use_container_width=True)

        st.info("💡 Denoised results can be passed directly to the **✏️ Edge Detection (Canny)** menu.")

    show_zip_download("denoised_zip_path", "📦 Download denoised_results.zip",
                      "denoised_results.zip", "download_denoised_zip")

    with col2:
        st.header("ℹ️ Information")
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


# =========================================================
# MENU 2: DETEKSI TEPI CANNY
# =========================================================
WINDOW_PRESETS = {
    "DICOM header (WindowCenter/Width)": "header",
    "Soft tissue (WC 40 / WW 400)": (40, 400),
    "Lung (WC -600 / WW 1500)": (-600, 1500),
    "Bone (WC 300 / WW 1500)": (300, 1500),
    "Brain (WC 40 / WW 80)": (40, 80),
    "Percentile stretch 0.5–99.5% (ignore header)": "percentile",
    "Custom": "custom",
}


def page_edge():
    st.title("Edge Detection with Canny Algorithm (IMA/DICOM)")
    st.markdown("---")

    # ---------- Sidebar: sumber input ----------
    st.sidebar.subheader("📁 Input Images")
    has_denoised = bool(st.session_state.get("denoised_zip_path")) and \
        os.path.exists(st.session_state.get("denoised_zip_path", ""))
    src_options = ["Upload files / ZIP"]
    if has_denoised:
        src_options.insert(0, "Denoising results (this session)")
    src = st.sidebar.radio("Image source", src_options, key="ed_src")

    uploaded_imas, zip_folder = None, None
    if src == "Upload files / ZIP":
        uploaded_imas = st.sidebar.file_uploader(
            "Upload IMA/DICOM files (multi-select supported)",
            type=["ima", "dcm"], accept_multiple_files=True, key="ed_files")
        zip_folder = st.sidebar.file_uploader(
            "Or upload the folder as .zip", type=["zip"], key="ed_zip")

    # ---------- Sidebar: parameter Canny ----------
    st.sidebar.subheader("🪟 Windowing")
    win_label = st.sidebar.selectbox(
        "Window preset", list(WINDOW_PRESETS.keys()), key="ed_win",
        help="Determines which tissues are visible. Tissue outside the window range "
             "becomes flat, so its edges will not be detected.")
    preset = WINDOW_PRESETS[win_label]
    window, use_header = None, True
    if preset == "header":
        window, use_header = None, True
    elif preset == "percentile":
        window, use_header = None, False
    elif preset == "custom":
        wc = st.sidebar.number_input("Window Center (HU)", value=40.0, step=10.0, key="ed_wc")
        ww = st.sidebar.number_input("Window Width (HU)", min_value=1.0, value=400.0,
                                     step=10.0, key="ed_ww")
        window = (wc, ww)
    else:
        window = preset

    st.sidebar.subheader("✏️ Canny Parameters")
    blur = st.sidebar.slider(
        "Gaussian blur kernel (odd)", 1, 15, 5, step=2, key="ed_blur",
        help="Initial smoothing. 1 = no blur. Increase if the image is very noisy/speckled.")
    low, high = st.sidebar.slider(
        "Hysteresis thresholds (LOW, HIGH)", 0, 500, (55, 150), key="ed_thr",
        help="Gradients above HIGH are strong edges; between LOW and HIGH are kept only if "
             "connected to a strong edge. Smaller values = more edges.")
    l2 = st.sidebar.checkbox("L2 gradient (more accurate)", value=True, key="ed_l2")

    st.sidebar.subheader("🧽 Post-processing")
    min_length = st.sidebar.slider(
        "Minimum edge length (pixels)", 0, 300, 30, step=5, key="ed_minlen",
        help="Edge components smaller than this value are removed (noise fragments).")
    thickness = st.sidebar.slider("Line thickness (pixels)", 1, 6, 2, key="ed_thick")

    st.sidebar.subheader("💾 Output Mode")
    mode_label = st.sidebar.radio(
        "Output DICOM", ["Color (RGB 8-bit)", "Gray (MONOCHROME2, original tags)"],
        key="ed_mode",
        help="Color: colored lines, saved as 8-bit RGB DICOM (HU values are lost).\n"
             "Gray: white lines, keeps MONOCHROME2 with all original tags.")
    mode = "color" if mode_label.startswith("Color") else "gray"
    color_hex = st.sidebar.color_picker("Line color", "#145AC8", key="ed_color")
    color = hex_to_rgb(color_hex)

    params = dict(window=window, use_header=use_header, blur=blur, low=low, high=high,
                  l2=l2, thickness=thickness, min_length=min_length, mode=mode, color=color)

    # ---------- Kumpulkan file ----------
    try:
        if src == "Denoising results (this session)":
            items = collect_sources(None, None, st.session_state["denoised_zip_path"])
        else:
            items = collect_sources(uploaded_imas, zip_folder)
    except Exception as e:
        st.error(f"❌ Failed to read input: {e}")
        items = []

    col1, col2 = st.columns([2, 1])

    with col2:
        st.header("ℹ️ Information")
        if items:
            st.info(f"📄 Number of files: {len(items)}")
            st.subheader("📋 Preview file list:")
            for n, _ in items[:5]:
                st.text(f"• {os.path.basename(n)}")
            if len(items) > 5:
                st.text(f"... and {len(items) - 5} others")
        else:
            st.info("Upload images (or run Denoising first) to start.")
        with st.expander("📖 Parameter guide"):
            st.markdown(
                "- **Too many lines/noise?** Raise LOW/HIGH, increase blur, or raise the minimum edge length.\n"
                "- **Edges missing?** Lower LOW/HIGH or reduce blur.\n"
                "- A HIGH:LOW ratio of about **2:1 to 3:1** is a common starting point.\n"
                "- **Windowing** determines which tissue boundaries are visible to Canny.\n"
                "- Edge-overlay images are for image processing purposes, **not for diagnostic reading**."
            )

    with col1:
        # ---------- Live preview (ikut berubah saat parameter digeser) ----------
        st.header("🔬 Live Preview")
        if items:
            names = [n for n, _ in items]
            sel = st.selectbox("Select image for preview", names,
                               format_func=os.path.basename, key="ed_prev_sel")
            reader = dict(items)[sel]
            try:
                ds_prev = pydicom.dcmread(io.BytesIO(reader()))
                ensure_uncompressed(ds_prev)
                gray = dicom_to_uint8(ds_prev, window=window, use_header_window=use_header)
                edges = detect_edges(gray, blur=blur, low=low, high=high,
                                     thickness=thickness, min_length=min_length, l2=l2)
                overlay = make_overlay(gray, edges, color)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.caption("Input (after windowing)")
                    st.image(gray, use_container_width=True, clamp=True)
                with c2:
                    st.caption("Edge mask")
                    st.image(edges, use_container_width=True, clamp=True)
                with c3:
                    st.caption("Overlay")
                    st.image(overlay, use_container_width=True)
                frac = 100.0 * (edges > 0).mean()
                st.caption(f"Edge pixels: {int((edges > 0).sum()):,} ({frac:.2f}% of image)")
            except Exception as e:
                st.error(f"❌ Failed to preview {os.path.basename(sel)}: {e}")
        else:
            st.caption("No image yet.")

        st.markdown("---")
        st.header("📋 Processing Status")
        status_box = st.container()

    if st.button("🚀 Start Edge Detection and Prepare ZIP", type="primary",
                 disabled=not items, key="ed_start"):
        st.session_state["edge_previews"] = []
        with status_box:
            zip_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            zip_tmp_path = zip_tmp.name
            zip_tmp.close()

            total = len(items)
            progress_bar = st.progress(0)
            status_text = st.empty()
            success_count, failed_files = 0, []

            with zipfile.ZipFile(zip_tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, (name, reader) in enumerate(items):
                    fname = os.path.basename(name)
                    status_text.text(f"Edge detection: {fname} ({idx + 1}/{total})")
                    try:
                        out_bytes, gray, edges = edge_process_bytes(reader(), params)
                        zf.writestr(name, out_bytes)   # nama & struktur folder dipertahankan
                        if len(st.session_state["edge_previews"]) < 5:
                            st.session_state["edge_previews"].append({
                                "fname": fname,
                                "input": Image.fromarray(gray),
                                "overlay": Image.fromarray(make_overlay(gray, edges, color)),
                                "n_edge": int((edges > 0).sum()),
                            })
                        success_count += 1
                    except Exception as e:
                        failed_files.append((fname, f"{type(e).__name__}: {e}"))
                    progress_bar.progress((idx + 1) / total)
            status_text.text("Done.")

            old = st.session_state.get("edge_zip_path")
            if old and old != zip_tmp_path and os.path.exists(old):
                try:
                    os.remove(old)
                except Exception:
                    pass
            st.session_state["edge_zip_path"] = zip_tmp_path
            st.session_state["edge_summary"] = (success_count, failed_files, dict(
                params, window=str(window if window else ("header" if use_header else "percentile")),
                color=color_hex))

    if "edge_summary" in st.session_state and st.session_state.get("edge_zip_path"):
        success_count, failed_files, used = st.session_state["edge_summary"]
        st.markdown("---")
        st.header("📊 Resume")
        c_ok, c_fail = st.columns(2)
        c_ok.metric("✅ Successfully", success_count)
        c_fail.metric("❌ Failed", len(failed_files))
        st.caption(
            f"Parameters used — window: {used['window']}, blur: {used['blur']}, "
            f"LOW/HIGH: {used['low']}/{used['high']}, L2: {used['l2']}, "
            f"min length: {used['min_length']}, thickness: {used['thickness']}, "
            f"mode: {used['mode']}, color: {used['color']}")
        if failed_files:
            st.subheader("⚠️ Failed to process:")
            for fname, err in failed_files:
                st.error(f"{fname}: {err}")

        if st.session_state.get("edge_previews"):
            st.markdown("---")
            st.header("🔍 Preview Input vs Edge Overlay (maksimal 5 contoh)")
            for pv in st.session_state["edge_previews"]:
                st.subheader(f"{pv['fname']}  ·  {pv['n_edge']:,} edge pixels")
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Input (after windowing)")
                    st.image(pv["input"], use_container_width=True)
                with c2:
                    st.caption("Edge overlay")
                    st.image(pv["overlay"], use_container_width=True)

    show_zip_download("edge_zip_path", "📦 Download edge_results.zip",
                      "edge_results.zip", "download_edge_zip")


# =========================================================
# Router
# =========================================================
if menu.startswith("🧹"):
    page_denoising()
else:
    page_edge()

st.sidebar.markdown("---")
st.sidebar.markdown("Give feedback to: ism22@tpu.ru")
