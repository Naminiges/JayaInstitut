import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration with a premium icon and layout
st.set_page_config(
    page_title="Jaya Jaya Institut - Student Dropout Early Warning System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS for a modern, glassmorphic look
st.markdown("""
<style>
    /* Main App Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Card styling */
    .header-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .header-card::after {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 50%;
        top: -100px;
        right: -100px;
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Styled Sub-sections */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3c72;
        border-left: 5px solid #2a5298;
        padding-left: 0.75rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Info Card styling */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Result styling */
    .result-box-dropout {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(255, 75, 43, 0.3);
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-box-graduate {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(56, 239, 125, 0.3);
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .result-text {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and preprocessor
@st.cache_resource
def load_model_artifacts():
    if not os.path.exists('model/model.joblib') or not os.path.exists('model/preprocessor.joblib'):
        return None, None
    model = joblib.load('model/model.joblib')
    preprocessor = joblib.load('model/preprocessor.joblib')
    return model, preprocessor

model, preprocessor = load_model_artifacts()

# Render Header
st.markdown("""
<div class="header-card">
    <h1 class="header-title">Jaya Jaya Institut</h1>
    <p class="header-subtitle">Early Warning System: Deteksi Dini & Pencegahan Mahasiswa Dropout</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("File model atau preprocessor tidak ditemukan di folder `model/`. Silakan jalankan `notebook.ipynb` atau script pelatihan terlebih dahulu untuk menghasilkan file model.")
    st.stop()

# Sidebar Information
st.sidebar.markdown("### Ringkasan Model")
st.sidebar.info(
    f"**Model:** Gradient Boosting Classifier\n\n"
    f"**ROC-AUC Score:** 95.64%\n\n"
    f"**Accuracy:** 91.00%\n\n"
    f"**Decision Threshold:** {preprocessor['threshold']:.2f}\n\n"
    f"Model dioptimasi menggunakan threshold kustom untuk memaksimalkan *Recall* (kemampuan mendeteksi mahasiswa yang benar-benar berisiko dropout)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Cara Menggunakan")
st.sidebar.write(
    "1. Gunakan salah satu tombol **Template Profil** untuk memuat sampel secara cepat.\n"
    "2. Atau sesuaikan manual form input di sebelah kanan.\n"
    "3. Hasil analisis, probabilitas, dan rekomendasi aksi akan diperbarui secara real-time di bagian bawah."
)

# Mappings for categorical inputs
marital_status_options = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto Union",
    6: "Legally Separated"
}

course_options = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}

qualification_options = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
    38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)"
}

application_mode_options = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)"
}

nationality_options = {
    1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch",
    14: "English", 17: "Lithuanian", 21: "Angolan", 22: "Cape Verdean",
    24: "Guinean", 25: "Mozambican", 26: "Santomean", 32: "Turkish",
    41: "Brazilian", 62: "Romanian", 100: "Moldova (Republic of)",
    101: "Mexican", 103: "Ukrainian", 105: "Russian", 108: "Cuban", 109: "Colombian"
}

# Template definitions
templates = {
    "Penerima Beasiswa Berprestasi (Low Risk)": {
        "Marital_status": 1, "Application_mode": 1, "Application_order": 1, "Course": 9500,
        "Daytime_evening_attendance": 1, "Previous_qualification": 1, "Previous_qualification_grade": 145.0,
        "Nacionality": 1, "Mothers_qualification": 1, "Fathers_qualification": 1,
        "Mothers_occupation": 5, "Fathers_occupation": 5, "Admission_grade": 140.0,
        "Displaced": 1, "Educational_special_needs": 0, "Debtor": 0, "Tuition_fees_up_to_date": 1,
        "Gender": 0, "Scholarship_holder": 1, "Age_at_enrollment": 18, "International": 0,
        "Curricular_units_1st_sem_credited": 0, "Curricular_units_1st_sem_enrolled": 6,
        "Curricular_units_1st_sem_evaluations": 6, "Curricular_units_1st_sem_approved": 6,
        "Curricular_units_1st_sem_grade": 15.5, "Curricular_units_1st_sem_without_evaluations": 0,
        "Curricular_units_2nd_sem_credited": 0, "Curricular_units_2nd_sem_enrolled": 6,
        "Curricular_units_2nd_sem_evaluations": 6, "Curricular_units_2nd_sem_approved": 6,
        "Curricular_units_2nd_sem_grade": 16.0, "Curricular_units_2nd_sem_without_evaluations": 0,
        "Unemployment_rate": 10.8, "Inflation_rate": 1.4, "GDP": 1.74
    },
    "Mahasiswa Berisiko Tinggi (SPP Menunggak & Akademik Kurang)": {
        "Marital_status": 1, "Application_mode": 17, "Application_order": 2, "Course": 9119,
        "Daytime_evening_attendance": 1, "Previous_qualification": 1, "Previous_qualification_grade": 115.0,
        "Nacionality": 1, "Mothers_qualification": 19, "Fathers_qualification": 19,
        "Mothers_occupation": 10, "Fathers_occupation": 10, "Admission_grade": 110.0,
        "Displaced": 0, "Educational_special_needs": 0, "Debtor": 1, "Tuition_fees_up_to_date": 0,
        "Gender": 1, "Scholarship_holder": 0, "Age_at_enrollment": 24, "International": 0,
        "Curricular_units_1st_sem_credited": 0, "Curricular_units_1st_sem_enrolled": 5,
        "Curricular_units_1st_sem_evaluations": 8, "Curricular_units_1st_sem_approved": 1,
        "Curricular_units_1st_sem_grade": 10.0, "Curricular_units_1st_sem_without_evaluations": 0,
        "Curricular_units_2nd_sem_credited": 0, "Curricular_units_2nd_sem_enrolled": 5,
        "Curricular_units_2nd_sem_evaluations": 10, "Curricular_units_2nd_sem_approved": 0,
        "Curricular_units_2nd_sem_grade": 0.0, "Curricular_units_2nd_sem_without_evaluations": 0,
        "Unemployment_rate": 12.4, "Inflation_rate": 0.5, "GDP": -1.2
    },
    "Mahasiswa Rata-rata (Moderate Risk)": {
        "Marital_status": 1, "Application_mode": 1, "Application_order": 1, "Course": 9238,
        "Daytime_evening_attendance": 1, "Previous_qualification": 1, "Previous_qualification_grade": 130.0,
        "Nacionality": 1, "Mothers_qualification": 1, "Fathers_qualification": 1,
        "Mothers_occupation": 9, "Fathers_occupation": 9, "Admission_grade": 125.0,
        "Displaced": 1, "Educational_special_needs": 0, "Debtor": 0, "Tuition_fees_up_to_date": 1,
        "Gender": 1, "Scholarship_holder": 0, "Age_at_enrollment": 20, "International": 0,
        "Curricular_units_1st_sem_credited": 0, "Curricular_units_1st_sem_enrolled": 6,
        "Curricular_units_1st_sem_evaluations": 7, "Curricular_units_1st_sem_approved": 4,
        "Curricular_units_1st_sem_grade": 12.2, "Curricular_units_1st_sem_without_evaluations": 0,
        "Curricular_units_2nd_sem_credited": 0, "Curricular_units_2nd_sem_enrolled": 6,
        "Curricular_units_2nd_sem_evaluations": 7, "Curricular_units_2nd_sem_approved": 3,
        "Curricular_units_2nd_sem_grade": 11.5, "Curricular_units_2nd_sem_without_evaluations": 0,
        "Unemployment_rate": 11.1, "Inflation_rate": 1.4, "GDP": 0.5
    }
}

# Template Selection Buttons
st.markdown("<div class='section-title'>Pilih Profil Contoh (Template)</div>", unsafe_allow_html=True)
col_t1, col_t2, col_t3 = st.columns(3)

# Use session state to handle form updates from templates
if "active_profile" not in st.session_state:
    st.session_state["active_profile"] = templates["Mahasiswa Rata-rata (Moderate Risk)"]

if col_t1.button("🟢 Penerima Beasiswa (Low Risk)"):
    st.session_state["active_profile"] = templates["Penerima Beasiswa Berprestasi (Low Risk)"]

if col_t2.button("🔴 SPP Menunggak & Nilai Rendah (High Risk)"):
    st.session_state["active_profile"] = templates["Mahasiswa Berisiko Tinggi (SPP Menunggak & Akademik Kurang)"]

if col_t3.button("🟡 Profil Rata-rata (Moderate Risk)"):
    st.session_state["active_profile"] = templates["Mahasiswa Rata-rata (Moderate Risk)"]

active_data = st.session_state["active_profile"]

# Render Input Form
st.markdown("<div class='section-title'>Data & Profil Lengkap Mahasiswa</div>", unsafe_allow_html=True)

with st.form("dropout_prediction_form"):
    tab1, tab2, tab3 = st.tabs(["Demografi & Sosial-Ekonomi", "Pendaftaran & Akademik", "Kondisi Ekonomi Makro"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox(
                "Jenis Kelamin",
                options=[0, 1],
                format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki",
                index=[0, 1].index(active_data["Gender"])
            )
            age = st.number_input(
                "Usia saat Pendaftaran",
                min_value=15, max_value=100,
                value=int(active_data["Age_at_enrollment"])
            )
            marital_status = st.selectbox(
                "Status Pernikahan",
                options=list(marital_status_options.keys()),
                format_func=lambda x: marital_status_options[x],
                index=list(marital_status_options.keys()).index(active_data["Marital_status"])
            )
            nationality = st.selectbox(
                "Kewarganegaraan",
                options=list(nationality_options.keys()),
                format_func=lambda x: nationality_options[x],
                index=list(nationality_options.keys()).index(active_data["Nacionality"])
            )
        with c2:
            displaced = st.selectbox(
                "Apakah Mahasiswa Rantau? (Displaced)",
                options=[0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                index=[0, 1].index(active_data["Displaced"])
            )
            international = st.selectbox(
                "Apakah Mahasiswa Internasional?",
                options=[0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                index=[0, 1].index(active_data["International"])
            )
            special_needs = st.selectbox(
                "Kebutuhan Pendidikan Khusus?",
                options=[0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                index=[0, 1].index(active_data["Educational_special_needs"])
            )
        with c3:
            debtor = st.selectbox(
                "Memiliki Hutang/Tunggakan Lain?",
                options=[0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                index=[0, 1].index(active_data["Debtor"])
            )
            tuition_up_to_date = st.selectbox(
                "Pembayaran SPP Lancar (Up to Date)?",
                options=[0, 1],
                format_func=lambda x: "Tidak/Menunggak" if x == 0 else "Lancar",
                index=[0, 1].index(active_data["Tuition_fees_up_to_date"])
            )
            scholarship = st.selectbox(
                "Penerima Beasiswa?",
                options=[0, 1],
                format_func=lambda x: "Bukan" if x == 0 else "Ya",
                index=[0, 1].index(active_data["Scholarship_holder"])
            )
            
        st.markdown("**Detail Orang Tua**")
        c1_p, c2_p = st.columns(2)
        with c1_p:
            m_qual = st.number_input("Kualifikasi Ibu (Kode 1-44)", min_value=1, max_value=44, value=int(active_data["Mothers_qualification"]))
            m_occ = st.number_input("Pekerjaan Ibu (Kode 1-100)", min_value=1, max_value=100, value=int(active_data["Mothers_occupation"]))
        with c2_p:
            f_qual = st.number_input("Kualifikasi Ayah (Kode 1-44)", min_value=1, max_value=44, value=int(active_data["Fathers_qualification"]))
            f_occ = st.number_input("Pekerjaan Ayah (Kode 1-100)", min_value=1, max_value=100, value=int(active_data["Fathers_occupation"]))

    with tab2:
        st.markdown("#### Informasi Akademik Pendaftaran")
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            course = st.selectbox(
                "Program Studi / Jurusan",
                options=list(course_options.keys()),
                format_func=lambda x: course_options[x],
                index=list(course_options.keys()).index(active_data["Course"])
            )
            app_mode = st.selectbox(
                "Jalur Pendaftaran (Application Mode)",
                options=list(application_mode_options.keys()),
                format_func=lambda x: application_mode_options[x],
                index=list(application_mode_options.keys()).index(active_data["Application_mode"])
            )
            app_order = st.number_input(
                "Urutan Pilihan Program Studi (Pilihan ke-)",
                min_value=0, max_value=9,
                value=int(active_data["Application_order"])
            )
        with col_reg2:
            attendance = st.selectbox(
                "Waktu Kuliah",
                options=[0, 1],
                format_func=lambda x: "Malam (Evening)" if x == 0 else "Siang (Daytime)",
                index=[0, 1].index(active_data["Daytime_evening_attendance"])
            )
            prev_qual = st.selectbox(
                "Kualifikasi Pendidikan Sebelumnya",
                options=list(qualification_options.keys()),
                format_func=lambda x: qualification_options[x],
                index=list(qualification_options.keys()).index(active_data["Previous_qualification"])
            )
            prev_grade = st.number_input(
                "Nilai Kualifikasi Sebelumnya (0.0 - 200.0)",
                min_value=0.0, max_value=200.0,
                value=float(active_data["Previous_qualification_grade"])
            )
            admission_grade = st.number_input(
                "Nilai Tes Masuk Seleksi (0.0 - 200.0)",
                min_value=0.0, max_value=200.0,
                value=float(active_data["Admission_grade"])
            )
            
        st.markdown("---")
        st.markdown("#### Performa Akademik Semester 1 & 2")
        col_sem1, col_sem2 = st.columns(2)
        with col_sem1:
            st.markdown("**Semester 1**")
            sem1_credited = st.number_input("Mata Kuliah Dikonversi/Kredit (Credit)", min_value=0, max_value=50, value=int(active_data["Curricular_units_1st_sem_credited"]))
            sem1_enrolled = st.number_input("Mata Kuliah Diambil (Enrolled)", min_value=0, max_value=50, value=int(active_data["Curricular_units_1st_sem_enrolled"]))
            sem1_eval = st.number_input("Mata Kuliah Dievaluasi (Evaluations)", min_value=0, max_value=50, value=int(active_data["Curricular_units_1st_sem_evaluations"]))
            sem1_app = st.number_input("Mata Kuliah Lulus (Approved)", min_value=0, max_value=50, value=int(active_data["Curricular_units_1st_sem_approved"]))
            sem1_grade = st.number_input("Nilai Rata-rata (Grade 0 - 20)", min_value=0.0, max_value=20.0, value=float(active_data["Curricular_units_1st_sem_grade"]))
            sem1_noneval = st.number_input("Mata Kuliah Tanpa Evaluasi", min_value=0, max_value=50, value=int(active_data["Curricular_units_1st_sem_without_evaluations"]))
        with col_sem2:
            st.markdown("**Semester 2**")
            sem2_credited = st.number_input("Mata Kuliah Dikonversi/Kredit (Credit) ", min_value=0, max_value=50, value=int(active_data["Curricular_units_2nd_sem_credited"]))
            sem2_enrolled = st.number_input("Mata Kuliah Diambil (Enrolled) ", min_value=0, max_value=50, value=int(active_data["Curricular_units_2nd_sem_enrolled"]))
            sem2_eval = st.number_input("Mata Kuliah Dievaluasi (Evaluations) ", min_value=0, max_value=50, value=int(active_data["Curricular_units_2nd_sem_evaluations"]))
            sem2_app = st.number_input("Mata Kuliah Lulus (Approved) ", min_value=0, max_value=50, value=int(active_data["Curricular_units_2nd_sem_approved"]))
            sem2_grade = st.number_input("Nilai Rata-rata (Grade 0 - 20) ", min_value=0.0, max_value=20.0, value=float(active_data["Curricular_units_2nd_sem_grade"]))
            sem2_noneval = st.number_input("Mata Kuliah Tanpa Evaluasi ", min_value=0, max_value=50, value=int(active_data["Curricular_units_2nd_sem_without_evaluations"]))

    with tab3:
        st.markdown("#### Faktor Pendorong Makro Ekonomi")
        c_ec1, c_ec2, c_ec3 = st.columns(3)
        with c_ec1:
            unemp_rate = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, max_value=100.0, value=float(active_data["Unemployment_rate"]))
        with c_ec2:
            inf_rate = st.number_input("Tingkat Inflasi (%)", min_value=-50.0, max_value=100.0, value=float(active_data["Inflation_rate"]))
        with c_ec3:
            gdp = st.number_input("Tingkat Pertumbuhan PDB / GDP", min_value=-100.0, max_value=100.0, value=float(active_data["GDP"]))

    submitted = st.form_submit_button("Mulai Analisis & Prediksi")

# Run Prediction Logic
if submitted or active_data:
    # Construct feature vector
    input_data = {
        'Marital_status': marital_status,
        'Application_mode': app_mode,
        'Application_order': app_order,
        'Course': course,
        'Daytime_evening_attendance': attendance,
        'Previous_qualification': prev_qual,
        'Previous_qualification_grade': prev_grade,
        'Nacionality': nationality,
        'Mothers_qualification': m_qual,
        'Fathers_qualification': f_qual,
        'Mothers_occupation': m_occ,
        'Fathers_occupation': f_occ,
        'Admission_grade': admission_grade,
        'Displaced': displaced,
        'Educational_special_needs': special_needs,
        'Debtor': debtor,
        'Tuition_fees_up_to_date': tuition_up_to_date,
        'Gender': gender,
        'Scholarship_holder': scholarship,
        'Age_at_enrollment': age,
        'International': international,
        'Curricular_units_1st_sem_credited': sem1_credited,
        'Curricular_units_1st_sem_enrolled': sem1_enrolled,
        'Curricular_units_1st_sem_evaluations': sem1_eval,
        'Curricular_units_1st_sem_approved': sem1_app,
        'Curricular_units_1st_sem_grade': sem1_grade,
        'Curricular_units_1st_sem_without_evaluations': sem1_noneval,
        'Curricular_units_2nd_sem_credited': sem2_credited,
        'Curricular_units_2nd_sem_enrolled': sem2_enrolled,
        'Curricular_units_2nd_sem_evaluations': sem2_eval,
        'Curricular_units_2nd_sem_approved': sem2_app,
        'Curricular_units_2nd_sem_grade': sem2_grade,
        'Curricular_units_2nd_sem_without_evaluations': sem2_noneval,
        'Unemployment_rate': unemp_rate,
        'Inflation_rate': inf_rate,
        'GDP': gdp
    }
    
    # Create pandas dataframe
    input_df = pd.DataFrame([input_data])
    
    # Scale numeric columns
    scaled_df = input_df.copy()
    scaled_df[preprocessor['num_cols_to_scale']] = preprocessor['scaler'].transform(input_df[preprocessor['num_cols_to_scale']])
    
    # Predict Probability
    prob = model.predict_proba(scaled_df)[0, 0]
    
    # Apply custom threshold
    prediction = 1 if prob >= preprocessor['threshold'] else 0
    
    st.markdown("<div class='section-title'>Hasil Analisis Prediksi</div>", unsafe_allow_html=True)
    
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box-dropout">
                <h2 class="result-title"> BERISIKO DROPOUT</h2>
                <p class="result-text">Model mendeteksi mahasiswa ini memiliki probabilitas dropout sebesar <b>{prob*100:.1f}%</b> (melebihi batas aman {preprocessor['threshold']*100:.1f}%).</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-graduate">
                <h2 class="result-title">AKTIF / AKAN LULUS</h2>
                <p class="result-text">Model memprediksi mahasiswa ini aman dengan probabilitas dropout hanya sebesar <b>{prob*100:.1f}%</b> (berada di bawah batas aman {preprocessor['threshold']*100:.1f}%).</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col_res2:
        st.write("#### Indikator Tingkat Risiko")
        # Visual progress bar with dynamic color
        if prob < 0.35:
            st.progress(prob, text=f"Risiko Rendah: {prob*100:.1f}%")
        elif prob < 0.65:
            st.progress(prob, text=f"Risiko Sedang: {prob*100:.1f}%")
        else:
            st.progress(prob, text=f"Risiko Sangat Tinggi: {prob*100:.1f}%")
            
        # Key driver highlights
        st.markdown("**Analisis Faktor Kunci:**")
        drivers = []
        if tuition_up_to_date == 0:
            drivers.append("Pembayaran SPP tidak lancar / menunggak.")
        if sem1_app < 3 or sem2_app < 3:
            drivers.append("Jumlah mata kuliah yang lulus di semester awal sangat sedikit.")
        if sem1_grade < 10 or sem2_grade < 10:
            drivers.append("Nilai akademik berada di bawah standar rata-rata (Grade < 10.0).")
        if age > 25:
            drivers.append("Usia pendaftaran dewasa (>25 tahun), cenderung memiliki tantangan ekstra.")
        if scholarship == 1:
            drivers.append("Penerima beasiswa (faktor positif untuk mengurangi risiko).")
            
        if drivers:
            for driver in drivers:
                st.write(driver)
        else:
            st.write("Tidak ada indikator risiko besar yang terdeteksi secara akademis maupun administratif.")

    st.markdown("---")
    st.markdown("<div class='section-title'>Rekomendasi Action Items Bagi Institusi (HR/Akademik)</div>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.warning("""
        **Rekomendasi Penanganan Segera:**
        1. **Bimbingan Konseling Akademis**: Jadwalkan pertemuan khusus dengan Dosen Wali untuk meninjau kegagalan mata kuliah di semester 1/2.
        2. **Relaksasi SPP / Cicilan Finansial**: Kirim pemberitahuan opsi keringanan finansial jika tunggakan SPP merupakan kendala utama.
        3. **Program Mentoring 'Buddy System'**: Hubungkan mahasiswa dengan mentor senior untuk membantu adaptasi dan pemulihan performa akademik.
        """)
    else:
        st.success("""
        **Rekomendasi Pemantauan Rutin:**
        1. **Pertahankan Kinerja**: Berikan apresiasi atau umpan balik positif secara berkala agar mahasiswa mempertahankan prestasi.
        2. **Early warning check**: Lakukan pemantauan berkala setiap akhir semester berikutnya untuk mendeteksi perubahan tren nilai.
        """)
