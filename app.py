import streamlit as st
from PIL import Image
import subprocess
import os
import pandas as pd

from predict import predict_image
from dataset_utils import save_uploaded_images

# ---------------- AUTH CONFIGURATION ----------------
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

def login(username, password):
    user = users.get(username)
    if user and user["password"] == password:
        return user["role"]
    return None

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(page_title="Bird Classifier", layout="centered")
st.title("Bird Species Classification Enhancement via Adaptive Inertia Weight Particle Swarm Optimization-Based Image Augmentation Selection")

# ---------------- STATE MANAGEMENT ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = None

if "scroll_trigger" not in st.session_state:
    st.session_state.scroll_trigger = False

# ---------------- SIDEBAR MANAGEMENT (ONLY ACTIVE WHEN LOGGED IN) ----------------
if st.session_state.logged_in:
    st.sidebar.title("Navigation Menu")
    st.sidebar.write(f"👤 Account: {st.session_state.role.upper()}")
    
    if st.session_state.role == "admin":
        page = st.sidebar.selectbox("Workspace Control:", ["Admin Dashboard"])
    else:
        page = st.sidebar.selectbox("Workspace Control:", ["User Dashboard"])
        
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Secure Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.selected_preset = None
        st.session_state.scroll_trigger = False
        st.success("Logged out successfully!")
        st.rerun()

# ================= SITUATION A: PUBLIC VISITOR LANDING SCREEN =================
if not st.session_state.logged_in:
    
    # 1. Show the Public Evaluation Presets Row
    st.write("### 🧪 Quick Evaluation Presets")
    st.write("Recruiters can click any model target preset below to run real-time inference calculations instantly without logging in:")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🟢 Asian Green Bee-Eater"):
            st.session_state.selected_preset = "Asian-green-Bee-Eater-Sample.jpg"
            st.session_state.scroll_trigger = True
    with col_btn2:
        if st.button("🔵 Painted Bunting"):
            st.session_state.selected_preset = "Painted_Bunting_Sample.jpg"
            st.session_state.scroll_trigger = True
    with col_btn3:
        if st.button("⚪ White Wagtail"):
            st.session_state.selected_preset = "White-Wagtail_sample.jpg"
            st.session_state.scroll_trigger = True

    st.markdown("---")

    # 2. Show the Account Login Form (With Standard, Clean Labels)
    st.subheader("🔐 Secure Workspace Authentication")
    st.write("Authorized accounts can log in below to unlock custom file upload testing channels or administrative tools.")
    
    with st.form("login_form_container"):
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        submit_login = st.form_submit_button("Verify & Sign In")
        
        if submit_login:
            detected_role = login(username_input, password_input)
            if detected_role:
                st.session_state.logged_in = True
                st.session_state.role = detected_role
                st.session_state.selected_preset = None  # Clear presets upon active login
                st.session_state.scroll_trigger = False
                st.success("Access authorized successfully!")
                st.rerun()  # Forces immediate layout change on first single click
            else:
                st.error("Invalid credentials provided. Please try again.")

    # 3. Dynamic Assessment Rendering Layer (Displays at the bottom only if a preset is clicked)
    if st.session_state.selected_preset and os.path.exists(st.session_state.selected_preset):
        st.markdown("---")
        
        # --- AUTOMATIC SCROLL INJECTION POINT ---
        # This invisible anchor combined with JS components jumps the viewport directly to the results
        st.markdown('<div id="result-view"></div>', unsafe_allow_html=True)
        if st.session_state.scroll_trigger:
            st.components.v1.html(
                """
                <script>
                    window.parent.document.getElementById('result-view').scrollIntoView({behavior: 'smooth'});
                </script>
                """,
                height=0,
                width=0
            )
            st.session_state.scroll_trigger = False # Reset trigger so it doesn't loop scroll endlessly

        st.write(f"### 📊 Live Model Inference Output: `{st.session_state.selected_preset}`")
        
        image = Image.open(st.session_state.selected_preset)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Preset Target Image", use_container_width=True)

        with col2:
            with st.spinner("Processing deep learning weights... 🧠"):
                status, label, confidence, top3, prediction = predict_image(image)

            if status == "unknown":
                st.error("❌ Unknown Bird Detected")
            elif status == "uncertain":
                st.warning("⚠️ High uncertainty threshold reached")
                st.info(f"🔍 Nearest matching distribution: {label}")
            else:
                st.success("✅ Species Confirmed!")
                st.write(f"🐦 **Classification Result:** {label}")

            st.info(f"📊 Pipeline Confidence: {confidence:.2f}")

        st.markdown("---")
        st.subheader("🏆 Distribution Hierarchy")
        for cls, prob in top3:
            st.write(f"{cls} ({prob:.2f})")
            st.progress(prob)

        st.markdown("---")
        st.subheader("📊 Probability Metrics Chart")
        from predict import class_names
        df = pd.DataFrame({"Class": class_names, "Probability": prediction})
        df = df.sort_values(by="Probability", ascending=False).head(5)
        st.bar_chart(df.set_index("Class"))

        st.markdown("---")
        st.subheader("📈 Core Optimization Metrics")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if os.path.exists("accuracy.png"):
                st.image("accuracy.png", caption="Model Accuracy Plot")
        with col_g2:
            if os.path.exists("loss.png"):
                st.image("loss.png", caption="Model Convergence Loss Plot")

# ================= SITUATION B: SECURE LOCKED USER DASHBOARD =================
elif st.session_state.logged_in and st.session_state.role == "user":
    st.header("🔍 Custom Image Upload Channel")
    st.write("Account status verified. You now have secure permission access to upload your own media assets.")

    # Custom "Browse files" loader is completely safe here behind the login wall
    uploaded_file = st.file_uploader("Upload Bird Target Asset", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image File", use_container_width=True)

        with col2:
            with st.spinner("Analyzing custom metrics pipeline... 🧠"):
                status, label, confidence, top3, prediction = predict_image(image)

            if status == "unknown":
                st.error("❌ Unknown Bird Profile")
            elif status == "uncertain":
                st.warning("⚠️ Dynamic variance mismatch")
                st.info(f"🔍 Alternate suggestion: {label}")
            else:
                st.success("✅ Successful Inference Mapping!")
                st.write(f"🐦 **Species Detected:** {label}")

            st.info(f"📊 Accuracy Confidence: {confidence:.2f}")

        st.markdown("---")
        st.subheader("🏆 Top Metric Outputs")
        for cls, prob in top3:
            st.write(f"{cls} ({prob:.2f})")
            st.progress(prob)

        st.markdown("---")
        st.subheader("📊 Class Density Distribution")
        from predict import class_names
        df = pd.DataFrame({"Class": class_names, "Probability": prediction})
        df = df.sort_values(by="Probability", ascending=False).head(5)
        st.bar_chart(df.set_index("Class"))

# ================= SITUATION C: SECURE ADMIN CONTROL LEVEL =================
elif st.session_state.logged_in and st.session_state.role == "admin":
    st.header("🛠 Enterprise Admin Core Dashboard")

    tab1, tab2, tab3 = st.tabs([
        "📂 Data Pipeline Uploads",
        "⚙ Trigger Optimization Training",
        "📊 System Performance Logs"
    ])

    # -------- TAB 1: UPLOAD --------
    with tab1:
        st.subheader("Append Training Directory Data")
        if "upload_key" not in st.session_state:
            st.session_state.upload_key = 0
        if "upload_message" not in st.session_state:
            st.session_state.upload_message = ""

        uploaded_files = st.file_uploader(
            "Select Training Image Files", accept_multiple_files=True, key=st.session_state.upload_key
        )
        class_name = st.text_input("Enter Target Class Key Directory Label")

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            if st.button("Commit Images to Directory"):
                if uploaded_files and class_name:
                    save_uploaded_images(uploaded_files, class_name)
                    st.session_state.upload_message = "✅ Directory matrices updated!"
                    st.session_state.upload_key += 1
                    st.rerun()
                else:
