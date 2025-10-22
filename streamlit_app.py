
import os
import streamlit as st
import numpy as np
import pydicom
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Low-Dose CT Medical Image Denoising App",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Low-Dose CT Medical Image Denoising (IMA / DICOM)")
st.markdown("---")

# Sidebar untuk input parameter
st.sidebar.header("⚙️ Settings")

# Input folder noisy
st.sidebar.subheader("📁 Input Folder Noisy Images")
noisy_folder = st.sidebar.text_input(
    "Input folder path contain noisy images (.IMA):",
    placeholder="Example: quarter_1mm_sharp_L109"
)

# File browser alternatif menggunakan selectbox jika folder ada di directory yang sama
if st.sidebar.checkbox("Select from local folder"):
    current_dir = Path(".")
    folders = [d.name for d in current_dir.iterdir() if d.is_dir()]
    if folders:
        noisy_folder = st.sidebar.selectbox("Select folder:", options=folders)

# Input model path
st.sidebar.subheader("🤖 Model File")
model_path = st.sidebar.text_input(
    "Input local model file (.h5):",
    placeholder="Example: UNet denoising 3 layers E20B4_model.h5"
)

# File browser untuk model
if st.sidebar.checkbox("Select model from local file"):
    current_dir = Path(".")
    model_files = list(current_dir.glob("*.h5"))
    if model_files:
        model_path = st.sidebar.selectbox(
            "Select model file:", 
            options=[str(f) for f in model_files]
        )

# Input output folder
st.sidebar.subheader("💾 Folder Output")
output_folder = st.sidebar.text_input(
    "Type output folder name:",
    placeholder="Example: UNet Denoised Results"
)

# Progress tracking
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'total_files' not in st.session_state:
    st.session_state.total_files = 0

# Fungsi helper
@st.cache_resource
def load_denoising_model(model_path):
    """Load model ..."""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, True
    except Exception as e:
        return str(e), False

def read_dicom(path):
    """Read DICOM file (.IMA) and return to array numpy 2D"""
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        # Normalisasi ke [0,1]
        img = img / 255.0
        return img, dcm, True
    except Exception as e:
        return str(e), None, False

def save_dicom(original_dcm, denoised_array, save_path):
    """Save denoised result as new DICOM file """
    try:
        denoised_scaled = (denoised_array * 255).astype(np.uint16)
        dcm = original_dcm.copy()  # Buat copy untuk menghindari modifikasi original
        dcm.PixelData = denoised_scaled.tobytes()
        dcm.save_as(save_path)
        return True, "Successfully saved"
    except Exception as e:
        return False, str(e)

def get_ima_files(folder_path):
    """Get all .IMA files from folder"""
    if not os.path.exists(folder_path):
        return []
    
    ima_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.ima'):
            ima_files.append(filename)
    
    return sorted(ima_files)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Processing Status")
    
    # Validasi input
    validation_status = st.container()
    
    # Tombol untuk memulai proses
    if st.button("🚀 Start Denoising Process", type="primary", disabled=not all([noisy_folder, model_path, output_folder])):
        
        with validation_status:
            # Validasi folder noisy
            if not noisy_folder or not os.path.exists(noisy_folder):
                st.error("❌ Noisy Folder not found!")
                st.stop()
            
            # Validasi model
            if not model_path or not os.path.exists(model_path):
                st.error("❌ Model File not found!")
                st.stop()
            
            # Validasi file .IMA
            ima_files = get_ima_files(noisy_folder)
            if not ima_files:
                st.error("❌ No .IMA file found in input folder!")
                st.stop()
            
            st.success(f"✅ Found {len(ima_files)} .IMA files for processing")
        
        # Load model
        with st.spinner("⏳ Loading model..."):
            model, model_loaded = load_denoising_model(model_path)
            
            if not model_loaded:
                st.error(f"❌ Failed to load model: {model}")
                st.stop()
            else:
                st.success("✅ Model successfully loaded!")
        
        # Buat folder output
        os.makedirs(output_folder, exist_ok=True)
        st.info(f"📁 Output folder: {output_folder}")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Counter untuk file yang berhasil dan gagal
        success_count = 0
        failed_count = 0
        failed_files = []
        
        # Proses setiap file
        for idx, filename in enumerate(ima_files):
            input_path = os.path.join(noisy_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Update status
            status_text.text(f"Denoising: {filename} ({idx + 1}/{len(ima_files)})")
            
            # Baca DICOM
            img_norm, dcm, read_success = read_dicom(input_path)
            
            if not read_success:
                failed_count += 1
                failed_files.append((filename, img_norm))
                st.warning(f"⚠️ Failed to read {filename}: {img_norm}")
                continue
            
            try:
                # Ubah ke bentuk (1, 512, 512, 1)
                img_input = np.expand_dims(img_norm, axis=(0, -1))
                
                # Prediksi denoising
                denoised = model.predict(img_input, verbose=0)[0, :, :, 0]
                
                # Simpan hasil ke DICOM baru
                save_success, save_message = save_dicom(dcm, denoised, output_path)
                
                if save_success:
                    success_count += 1
                    st.success(f"✅ {filename} successfully denoised")
                else:
                    failed_count += 1
                    failed_files.append((filename, save_message))
                    st.error(f"❌ Failed to save {filename}: {save_message}")
                    
            except Exception as e:
                failed_count += 1
                failed_files.append((filename, str(e)))
                st.error(f"❌ Error to process {filename}: {str(e)}")
            
            # Update progress
            progress = (idx + 1) / len(ima_files)
            progress_bar.progress(progress)
        
        # Summary hasil
        st.markdown("---")
        st.header("📊 Resume")
        
        col_success, col_failed = st.columns(2)
        with col_success:
            st.metric("✅ Successfully", success_count)
        with col_failed:
            st.metric("❌ Failed", failed_count)
        
        if failed_files:
            st.subheader("⚠️ Failed to denoised:")
            for filename, error in failed_files:
                st.error(f"**{filename}**: {error}")
        
        if success_count > 0:
            st.success(f"🎉 Denoising done! {success_count} files successfully saved to: **{output_folder}**")

with col2:
    st.header("ℹ️ Information")
    
    # Informasi folder dan model
    info_container = st.container()
    
    with info_container:
        if noisy_folder:
            ima_files = get_ima_files(noisy_folder)
            if ima_files:
                st.info(f"📁 **Input Folder:** {noisy_folder}")
                st.info(f"📄 **Number of .IMA files:** {len(ima_files)}")
                
                # Tampilkan beberapa nama file pertama
                st.subheader("📋 List of File:")
                for i, file in enumerate(ima_files[:5]):  # Tampilkan 5 file pertama
                    st.text(f"• {file}")
                if len(ima_files) > 5:
                    st.text(f"... and {len(ima_files) - 5} others")
        
        if model_path and os.path.exists(model_path):
            st.info(f"🤖 **Model:** {model_path}")
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            st.info(f"📊 **Model Size:** {file_size:.2f} MB")
        
        if output_folder:
            st.info(f"💾 **Output Folder:** {output_folder}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>
        🏥 <strong>Low-Dose CT Medical Image Denoising App</strong><br>
        Deep Learning based to denoise Low-Dose CT Medical Images in DICOM (.IMA) format
    </small>
</div>
""", unsafe_allow_html=True)

# Instruksi penggunaan
with st.expander("📖 Instruction for Use"):
    st.markdown("""
    ### Steps:
    1. **Select input folder** contains Low-Dose CT noisy images in .IMA format
    2. **Select model** (.h5) have been trained for denoising
    3. **Type output folder** to saving the results
    4. **Press "Start Denoising Process" button**
    
    ### Importan Notes:
    - Make sure the input folder contains the .IMA file.
    - Model files must be in .h5 format (Keras/TensorFlow)
    - The results will be saved with the same name and format as the input.
    - The normalization and denormalization processes are performed automatically.
    
    ### Supported Formats:
    - **Input**: File DICOM (.IMA) 
    - **Model**: Keras model (.h5)
    - **Output**: File DICOM (.IMA) denoised
    """)
