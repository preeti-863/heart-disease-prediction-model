# landing.py
import streamlit as st
import base64

st.set_page_config(page_title="Heart Disease Risk Assessment", layout="wide", page_icon="💓")

# ---------- FUNCTION TO ENCODE IMAGE ----------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---------- IMAGE PATHS ----------
banner_path = r"D:/training/Project/images/banner_image.jpg"
logo_path = r"D:/training/Project/images/logo_image.jpg"

# ---------- ENCODE IMAGES ----------
try:
    banner_b64 = get_base64(banner_path)
except Exception:
    banner_b64 = None

try:
    logo_b64 = get_base64(logo_path)
except Exception:
    logo_b64 = None

# ===============================
# REMOVE STREAMLIT DEFAULT UI
# ===============================
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@700&display=swap" rel="stylesheet">
    <style>
        header {visibility: hidden;}
        [data-testid="stToolbar"], [data-testid="stDecoration"], 
        [data-testid="stStatusWidget"], #MainMenu, footer,
        [data-testid="stSidebar"], [data-testid="stSidebarNav"],
        section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        .block-container { padding: 0 !important; }
    </style>
""", unsafe_allow_html=True)

# ===============================
# CUSTOM HEADER
# ===============================
st.markdown(f"""
<style>
    @keyframes colorPulse {{
        0% {{ color: white; }}
        50% {{ color: #C21807; }}
        100% {{ color: white; }}
    }}

    .header-container {{
        width: 100%;
        background: black;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 5px 40px;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 9999;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }}

    .header-logo img {{
        width: 140px;
        height: auto;
    }}

    .header-title {{
        flex: 2;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'Roboto', sans-serif;
        animation: colorPulse 2.5s infinite ease-in-out;
        color: white;
    }}

    .header-btn a {{
        font-size: 1.2rem;
        font-weight: 700;
        color: white !important;
        background: linear-gradient(90deg, #C21807, #FF5252);
        padding: 10px 18px;
        border-radius: 10px;
        text-decoration: none;
        transition: 0.3s;
    }}

    .header-btn a:hover {{
        background: linear-gradient(90deg, #FF5252, #C21807);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }}

    .header-space {{ height:120px; }}
</style>

<div class="header-container">
    <div class="header-logo">
        {'<img src="data:image/jpg;base64,' + logo_b64 + '">' if logo_b64 else ''}
    </div>
    <div class="header-title">Heart Disease Risk Predictor</div>
    <div class="header-btn">
        <a href="/streamlit_file" target="_self">
          Test
        </a>
    </div>
</div>
<div class="header-space"></div>
""", unsafe_allow_html=True)

# ===============================
# BANNER SECTION
# ===============================
if banner_b64:
    st.markdown(
        f"""
        <style>
            .banner-container {{
                background: url('data:image/jpg;base64,{banner_b64}') center/cover no-repeat;
                background-attachment: fixed;
                height: 100vh;
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .banner-text {{
                background: rgba(255,255,255,0.15);
                backdrop-filter: blur(12px);
                padding: 50px 40px;
                border-radius: 25px;
                max-width: 700px;
                text-align: center;
                color: #FFF8F0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            }}
            .banner-text h2 {{
                font-size: 3rem;
                color: #FFCDD2;
                margin-bottom: 20px;
                text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
            }}
            .banner-text p {{
                font-size: 1.3rem;
                line-height: 1.8;
            }}
            .banner-btn {{
                font-size: 1.5rem;
                font-weight: 700;
                color: #FFF;
                background: linear-gradient(90deg, #FF8A65, #FF5252);
                border: none;
                border-radius: 50px;
                padding: 15px 40px;
                margin-top: 25px;
                cursor: pointer;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                text-decoration: none;
            }}
            .banner-btn:hover {{
                transform: translateY(-3px);
                box-shadow: 0 12px 25px rgba(0,0,0,0.3);
            }}
        </style>

        <div class="banner-container">
            <div class="banner-text">
                <h2>Predict Your Heart Disease Risk</h2>
                <p>Enter a few health metrics & receive a personalised risk score and actionable suggestions.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================
# CENTERED BUTTON
# ===============================

st.markdown("""
<div style="display:flex; justify-content:center; margin-top:60px;">
    <a href="/streamlit_file">
        <button style="
            background: linear-gradient(90deg, #FF8A65, #FF5252);
            color: #FFF;
            border-radius: 50px;
            padding: 30px 80px;
            font-size: 2rem;    /* Change this to control font size */
            font-weight: 800;   /* Change this to control font weight */
            border: none;
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.3);
        ">
            Check My Heart Status
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

# ===============================
# UNIVERSAL CONTENT STYLING
# ===============================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #FF5252, #FF8A65, #FFCDD2);
    background-attachment: fixed;
    font-family: 'Roboto', sans-serif;
}
body::before {
    content: "";
    position: fixed;
    top:0; left:0;
    width:100%; height:100%;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.1), transparent 70%),
                radial-gradient(circle at 70% 70%, rgba(255,255,255,0.05), transparent 70%);
    z-index: 0;
    pointer-events: none;
}

.section-container {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 50px 30px;
    margin: 50px auto;
    max-width: 900px;
    box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    transition: all 0.3s ease-in-out;
}
.section-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 35px rgba(0,0,0,0.25);
}
.section-title {
    font-size: 2.5rem;
    font-weight: 900;
    text-align: center;
    color: black;
    margin-bottom: 25px;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
}
.section-text {
    font-size: 1.3rem;
    text-align: center;
    line-height: 1.8;
    color: black;
}
.features-grid {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    justify-content: center;
}
.feature-box {
    background: rgba(255, 255, 255, 0.2);
    padding: 30px 20px;
    width: 260px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    transition: all 0.3s ease-in-out;
    color: black;
    border: 1px solid rgba(255,255,255,0.2);
}
.feature-box:hover {
    transform: translateY(-10px) scale(1.02);
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    background: rgba(255, 255, 255, 0.25);
}
.step-box {
    background: rgba(255,255,255,0.15);
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    color: black;
    border: 1px solid rgba(255,255,255,0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.step-box:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
}
footer {
    position: fixed;
}
.main-footer{
    bottom: 0;
    width: 100%;
    height:100%;
    color: #FFF8F0 !important;
    text-align: center;
    padding: 15px 0;
    font-size: 1rem;
    background: rgba(0,0,0,0.4);
    backdrop-filter: blur(5px);
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# ABOUT MODEL
# ===============================
st.markdown("""
<div class="section-container">
    <div class="section-title">🔍 About the Heart Disease Prediction Model</div>
    <div class="section-text">
        Our AI-powered model analyzes key medical parameters such as cholesterol, blood pressure, glucose levels, 
        lifestyle patterns, and historical risk factors to estimate heart disease probability.
        <br><br>
        Trained using **Random Forest** and **Logistic Regression**, 
        the model delivers accurate and explainable predictions.
    </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# FEATURES GRID
# ===============================
st.markdown("""
<div class="section-container">
    <div class="section-title">⭐ Key Features</div>
    <div class="features-grid">
        <div class="feature-box"><b>📊 Accurate Predictions</b><br>Trained on medical datasets.</div>
        <div class="feature-box"><b>⏱ Instant Results</b><br>Get predictions within seconds.</div>
        <div class="feature-box"><b>💡 Recommendations</b><br>Receive personalized health tips.</div>
        <div class="feature-box"><b>🔐 Secure</b><br>Your data stays private.</div>
        <div class="feature-box"><b>📉 Easy Input</b><br>Simple user-friendly form.</div>
        <div class="feature-box"><b>🌍 Works Everywhere</b><br>Mobile, desktop, tablet.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# HOW IT WORKS
# ===============================
st.markdown("""
<div class="section-container">
    <div class="section-title">⚙️ How It Works</div>
    <div class="step-box"><b>1️⃣ Enter Your Data</b><br>Age, cholesterol, glucose, heart rate.</div>
    <div class="step-box"><b>2️⃣ AI Processes Input</b><br>ML model analyzes instantly.</div>
    <div class="step-box"><b>3️⃣ Get Your Result</b><br>See your predicted risk level.</div>
    <div class="step-box"><b>4️⃣ Improve Your Health</b><br>Follow our personalized tips.</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="main-footer">
    <div class="section-text" style="text-align:center; font-size:1rem; color:black;">
        © 2025 Preeti Verma • Not a substitute for medical advice. Always consult a doctor.
    </div>
</div>
""", unsafe_allow_html=True)
