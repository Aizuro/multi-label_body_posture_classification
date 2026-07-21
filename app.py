import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
from PIL import Image
from feature_engineer import FeatureEngineer

# =============================================================================
# 1. KONFIGURASI AWAL & LOAD ASSETS
# =============================================================================
LABELS = ["forward_head", "postural_kyphosis", "rounded_shoulder", "normal"]

st.set_page_config(
    page_title="Sistem Deteksi Permasalahan Postur Tubuh (SVM)",
    page_icon="🧍",
    layout="centered"
)

@st.cache_resource
def load_pipeline_assets():
    """Memuat model SVM, Scaler, dan Indeks Fitur Terpilih menggunakan joblib"""
    # SVM menggunakan joblib.load, berbeda dengan ANN yang menggunakan tf.keras
    model = joblib.load('best_model_svm.pkl')
    scaler = joblib.load('scaler.pkl')
    selected_indices = np.load('selected_features.npy')
    return model, scaler, selected_indices

try:
    svm_model, data_scaler, selected_features_idx = load_pipeline_assets()
    fe = FeatureEngineer()
except Exception as e:
    st.error(f"Gagal memuat berkas model/pipeline. Pastikan file 'best_model_svm.pkl', 'scaler.pkl', dan 'selected_features.npy' ada di folder aplikasi. Error: {e}")
    st.stop()

# Setup MediaPipe untuk visualisasi skeleton di GUI
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_visualizer = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# =============================================================================
# 2. SELEKSI INPUT GAMBAR (URUTAN AWAL: UNGGAH FOTO)
# =============================================================================
st.title("Multi-label Classification Body Posture")

with st.expander("Photo Guideline (Must Read)"):
    st.markdown("""
    To ensure that the MediaPipe model can accurately extract joint points, make sure your photos meet the following standards:
    1. The photo must be a **side-view** (taken from the side).
    2. Body posture must be **standing** position.
    3. Photo must **not be blurry** (make sure the lighting is bright and the photo is in focus).
    4. Make sure your **upperbody (from the top of your head to your hips)** is clearly visible and not cut off by the camera frame.
    """)

    try:
        st.image("a.png", caption="Examples of correct (Left) and Incorrect (Right) Photos", use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ File gambar panduan (a.jpg) tidak ditemukan di folder.")

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
            max-width: 1000px;  /* atur sesuai kebutuhan, default centered ~730px */
        }
    </style>
""", unsafe_allow_html=True)

# Pilihan awal sudah otomatis diatur ke "Unggah Foto dari Folder"
input_option = st.radio(
    "Select a Photo Input Method:",
    ("Upload Photos from a Folder (Gallery)", "Take a Photo Directly (Cell Phone Camera / Webcam)")
)

image_file = None
if input_option == "Upload Photos from a Folder (Gallery)":
    image_file = st.file_uploader("Select a photo from a folder on your computer or phone", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input("Please take a photo of yourself standing upright from the side (side-view)")

# =============================================================================
# 3. PROSES INFERENSI DAN VISUALISASI
# =============================================================================
if image_file is not None:
    # Konversi file gambar Streamlit ke format OpenCV (BGR)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    st.info("Currently processing and analyzing body posture...")
    
    # Ekstrak Landmark & Fitur Tambahan (Sama seperti saat training)
    raw_landmarks = fe.extract_landmarks(img_bgr)
    
    if raw_landmarks is None:
        st.error("⚠️ MediaPipe failed to detect the silhouette of a human body in the photo. Make sure the entire body, from head to hips, is clearly visible from the side with sufficient lighting.")
    else:
        # Hitung 54 fitur engineered
        engineered_features = fe.calculate_enhanced_features(raw_landmarks)
        
        # Gabungkan raw + engineered menjadi 186 fitur
        total_features = raw_landmarks.tolist() + engineered_features
        
        # Lakukan scaling menggunakan RobustScaler yang sudah dilatih sebelumnya
        features_scaled = data_scaler.transform([total_features])
        
        # Seleksi 60 fitur terbaik berdasarkan indeks yang sudah disimpan
        features_selected = features_scaled[:, selected_features_idx]
        
        # --- PERUBAHAN KHUSUS UNTUK SVM (MultiOutputClassifier) ---
        # predict_proba() pada MultiOutputClassifier menghasilkan list of arrays berukuran (1, 2) untuk tiap kelas.
        # Kita ambil probabilitas kelas positif (indeks [0][1]) untuk setiap label.
        probabilities_raw = svm_model.predict_proba(features_selected)
        probabilities = [float(prob[0][1]) for prob in probabilities_raw]
        # -----------------------------------------------------------
        
        # Pembuatan Visualisasi Skeleton menggunakan MediaPipe Drawing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_results = pose_visualizer.process(img_rgb)
        
        annotated_image = img_rgb.copy()
        if mp_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, 
                mp_results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3), # Titik landmark
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=2)  # Garis tulang
            )
        
        # Tampilkan Hasil Visualisasi
        st.subheader("Analysis Visualization Results")
        st.image(annotated_image, caption="Visualization of the Body's Skeletal Structure", use_container_width=True)
        
        # Menentukan status permasalahan postur berdasarkan threshold 0.5
        detected_problems = []
        st.subheader("Posture Assessment Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            for i in range(3): # Loop untuk FHP, Kyphosis, dan RSP
                label_clean = LABELS[i].replace("_", " ").title()
                prob_percentage = probabilities[i] * 100
                
                st.write(f"**{label_clean}**")
                st.progress(probabilities[i]) # Sudah aman dalam bentuk float bawaan Python
                st.write(f"Confidence Level: `{prob_percentage:.2f}%`")
                
                if probabilities[i] >= 0.5:
                    detected_problems.append(LABELS[i])
        
        with col2:
            # Tampilkan skor untuk kondisi normal tersendiri
            norm_percentage = probabilities[3] * 100
            st.write(f"**Normal Posture**")
            st.progress(probabilities[3])
            st.write(f"Confidence Level: `{norm_percentage:.2f}%`")
            
            if len(detected_problems) == 0:
                detected_problems.append("normal")

        # =============================================================================
        # 4. PENJELASAN EDUKASI REKOMENDASI POSTUR
        # =============================================================================
        st.markdown("---")
        st.subheader("Conclusion & Educational Information")
        
        if "normal" in detected_problems and len(detected_problems) == 1:
            st.success("🎉 **Congratulations! Your Posture Has Been Detected as Normal.**")
            st.markdown("""
            **Explanation:** The structure of your spine and the alignment of your head, shoulders, and pelvis are in good alignment.
            
            **Recommendation:** * Make this a habit by stretching periodically every 30–45 minutes during long periods of sitting.
            * Adjust the monitor so that the screen is at eye level to prevent neck muscle strain.
            """)
        else:
            st.write("Based on the analysis, you are indicated to have the following condition:")
            
            if "forward_head" in detected_problems:
                with st.expander(":red[Forward Head Posture (FHP)]"):
                    st.markdown("""
                    **What is FHP?** A condition in which the head is hyperextended or tilted too far forward beyond the vertical line of the shoulders. It is often referred to as *Text Neck*.
                    
                    **Causes:** Staring at a computer monitor or smartphone that is positioned too low for long periods of time forces the muscles at the back of the neck to work extra hard to support the weight of the head.
                    
                    **Correction Tips:** Adjust your seating position for ergonomic comfort, raise the monitor to eye level, and do *chin tucks* to strengthen your front neck muscles.
                    """)
                    
            if "postural_kyphosis" in detected_problems:
                with st.expander(":red[Postural Kyphosis]"):
                    st.markdown("""
                    **What is Postural Kyphosis?** A condition characterized by excessive curvature of the upper spine (thoracic spine), causing the upper back to appear hunched backward.
                    
                    **Causes:** The habit of *slouching* in a chair without proper lower back support causes the spine to lose its natural curvature.
                    
                    **Correction Tips:** Use a chair with *lumbar support* (a cushion for the lower back), avoid slouching while typing, and stretch your chest periodically.
                    """)
                    
            if "rounded_shoulder" in detected_problems:
                with st.expander(":red[Rounded Shoulder Posture (RSP)]"):
                    st.markdown("""
                    **What is Rounded Shoulder?** A position in which the shoulder joint shifts or curves too far forward and rotates inward, causing the upper body to appear hunched.
                    
                    **Causes:** A desk that is set too high or the habit of leaning your arms forward while typing without leaning back can cause the chest muscles (*pectoralis*) to shorten and the upper back muscles to weaken.
                    
                    **Correction Tips:** Move your seat closer to the table so that your forearms can rest comfortably on the armrests or the table at a 90-degree angle. Strengthen your shoulder blade muscles with *scapular retractions*.
                    """)