import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="توقع نجاة ركاب التيتانيك",
    page_icon="🚢",
    layout="centered"
)

# Custom CSS for high-quality dark theme, custom fonts, RTL layout, glassmorphism, and beautiful result cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
    
    /* Main body & container setup */
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        text-align: right;
        direction: rtl;
    }
    
    /* Sleek gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #020617 100%);
        color: #f8fafc;
    }
    
    /* Header typography and gradient */
    .title-text {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-family: 'Cairo', sans-serif;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 2.5rem;
        font-family: 'Cairo', sans-serif;
    }

    /* Glassmorphism containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(129, 140, 248, 0.3);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6);
    }
    
    /* Label typography */
    label, p, span, div {
        font-family: 'Cairo', sans-serif !important;
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 12px 30px !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6) !important;
        background: linear-gradient(90deg, #06b6d4 0%, #4f46e5 100%) !important;
    }

    /* Result cards */
    .result-survived {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .result-perished {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.25) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.2);
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .status-badge {
        font-size: 3rem;
        margin-bottom: 10px;
    }

    .status-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .status-desc {
        font-size: 1.2rem;
        color: #e2e8f0;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Load the saved best model pipeline
@st.cache_resource
def load_model():
    return joblib.load("best_titanic_model.joblib")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"خطأ في تحميل الموديل: {e}")
    model_loaded = False

# Layout - Title and Header
st.markdown('<div class="title-text">نظام التنبؤ بالنجاة لركاب التيتانيك 🚢</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">أدخل بيانات الراكب لمعرفة احتمالية نجاته من الغرق باستخدام نموذج الذكاء الاصطناعي</div>', unsafe_allow_html=True)

# Wrap inputs inside a card
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📋 بيانات الراكب")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("العمر (بالسنوات)", min_value=0, max_value=100, value=28, step=1)
    
    sex_arabic = st.selectbox("النوع", options=["ذكر", "أنثى"])
    sex = "male" if sex_arabic == "ذكر" else "female"
    
    pclass_arabic = st.selectbox("درجة السفر", options=["الدرجة الأولى", "الدرجة الثانية", "الدرجة الثالثة"])
    pclass = 1 if pclass_arabic == "الدرجة الأولى" else (2 if pclass_arabic == "الدرجة الثانية" else 3)

with col2:
    fare = st.number_input("سعر التذكرة ($)", min_value=0.0, max_value=600.0, value=32.2, step=1.0)
    
    embarked_arabic = st.selectbox("ميناء الصعود", options=["ساوثهامبتون (Southampton)", "شيربورغ (Cherbourg)", "كوينزتاون (Queenstown)"])
    embarked = "S" if embarked_arabic == "ساوثهامبتون (Southampton)" else ("C" if embarked_arabic == "شيربورغ (Cherbourg)" else "Q")
    
    st.markdown("<p style='font-size:0.95rem; font-weight:600; margin-bottom:5px;'>عدد الأقارب المرافقين</p>", unsafe_allow_html=True)
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        sibsp = st.number_input("الأشقاء/الزوج", min_value=0, max_value=10, value=0, step=1)
    with sub_col2:
        parch = st.number_input("الوالدين/الأطفال", min_value=0, max_value=10, value=0, step=1)

st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button("🔮 توقع حالة النجاة"):
    if model_loaded:
        # Create input DataFrame matching model features
        input_df = pd.DataFrame([{
            "Age": float(age),
            "SibSp": int(sibsp),
            "Parch": int(parch),
            "Fare": float(fare),
            "Pclass": int(pclass),
            "Sex": sex,
            "Embarked": embarked
        }])
        
        # Predict using model pipeline
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Display the result
        if prediction == 1:
            prob_percent = probabilities[1] * 100
            st.markdown(f"""
            <div class="result-survived">
                <div class="status-badge">🟢</div>
                <div class="status-title">الراكب سينجو بإذن الله!</div>
                <div class="status-desc">نسبة احتمال النجاة: <strong>{prob_percent:.2f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            prob_percent = probabilities[0] * 100
            st.markdown(f"""
            <div class="result-perished">
                <div class="status-badge">🔴</div>
                <div class="status-title">الراكب لن ينجو!</div>
                <div class="status-desc">نسبة احتمال الوفاة: <strong>{prob_percent:.2f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("لم يتم تحميل الموديل بنجاح. تأكد من وجود ملف best_titanic_model.joblib.")
