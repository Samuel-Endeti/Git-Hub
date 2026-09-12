import streamlit as st
from PIL import Image
import subprocess
import os
import pandas as pd

from predict import predict_image
from dataset_utils import save_uploaded_images

# ---------------- AUTH SYSTEM ----------------
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

def login(username, password):
    user = users.get(username)
    if user and user["password"] == password:
        return user["role"]
    return None

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Bird Classifier", layout="centered")
st.title("Bird Species Classification Enhancement via Adaptive Inertia Weight Particle Swarm Optimization-Based Image Augmentation Selection")

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        role = login(username, password)

        if role:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
st.sidebar.write(f"👤 Logged in as: {st.session_state.role}")

if st.session_state.role == "admin":
    page = st.sidebar.selectbox("Menu", ["Admin"])
else:
    page = st.sidebar.selectbox("Menu", ["User"])

# ---------------- LOGOUT ----------------
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.success("Logged out successfully!")
    st.rerun()

# ================= USER PAGE =================
if page == "User":
    st.header("🔍 Bird Species Prediction")

    # -------- QUICK TEST PRESETS FOR RECRUITERS --------
    st.write("### 🧪 Quick Evaluation Presets")
    st.write("Click any sample option below to instantly run evaluation metrics using preset dataset targets:")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    selected_preset_path = None

    with col_btn1:
        if st.button("🟢 Asian Green Bee-Eater"):
            selected_preset_path = "Asian-green-Bee-Eater-Sample.jpg"
    with col_btn2:
        if st.button("🔵 Painted Bunting"):
            selected_preset_path = "Painted_Bunting_Sample.jpg"
    with col_btn3:
        if st.button("⚪ White Wagtail"):
            selected_preset_path = "White-Wagtail_sample.jpg"

    st.markdown("---")

    # -------- FILE UPLOADER --------
    uploaded_file = st.file_uploader("Upload Bird Image", type=["jpg", "png"])

    image = None

    # Handle image prioritization (Manual Upload takes precedence over Button Click)
    if uploaded_file:
        image = Image.open(uploaded_file)
    elif selected_preset_path and os.path.exists(selected_preset_path):
        image = Image.open(selected_preset_path)
        st.info(f"Loaded evaluation preset target: `{selected_preset_path}`")

    # Run execution pipeline only if an image source is active
    if image:
        col1, col2 = st.columns(2)

        # -------- IMAGE VIEW --------
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # -------- PREDICTION OUTPUT --------
        with col2:
            with st.spinner("Analyzing image... 🧠"):
                status, label, confidence, top3, prediction = predict_image(image)

            # Status messages
            if status == "unknown":
                st.error("❌ Unknown Bird Detected")

            elif status == "uncertain":
                st.warning("⚠️ No exact match found")
                st.info(f"🔍 Closest match: {label}")

            else:
                st.success("✅ Bird Species Detected!")
                st.write(f"🐦 **Predicted:** {label}")

            st.info(f"📊 Confidence: {confidence:.2f}")

        st.markdown("---")

        # -------- TOP 3 --------
        st.subheader("🏆 Top Predictions")

        for cls, prob in top3:
            st.write(f"{cls} ({prob:.2f})")
            st.progress(prob)

        st.markdown("---")

        # -------- REAL-TIME CHART --------
        st.subheader("📊 Prediction Probabilities")

        # Create dataframe
        from predict import class_names  # import class names

        df = pd.DataFrame({
            "Class": class_names,
            "Probability": prediction
        })

        # Show top 5 only
        df = df.sort_values(by="Probability", ascending=False).head(5)

        st.bar_chart(df.set_index("Class"))

        st.markdown("---")

        # -------- MODEL PERFORMANCE METRICS --------
        st.subheader("📈 Model Performance")

        col3, col4 = st.columns(2)

        with col3:
            if os.path.exists("accuracy.png"):
                st.image("accuracy.png", caption="Accuracy Graph")
            else:
                st.warning("Train model to generate accuracy graph")

        with col4:
            if os.path.exists("loss.png"):
                st.image("loss.png", caption="Loss Graph")
            else:
                st.warning("Train model to generate loss graph")

# ================= ADMIN PAGE =================
if page == "Admin":
    st.header("🛠 Admin Dashboard")

    tab1, tab2, tab3 = st.tabs([
        "📂 Upload Data",
        "⚙ Train Model",
        "📊 Performance"
    ])

    # -------- TAB 1 --------
    with tab1:
        st.subheader("Upload New Bird Images")

        if "upload_key" not in st.session_state:
            st.session_state.upload_key = 0

        if "upload_message" not in st.session_state:
            st.session_state.upload_message = ""

        uploaded_files = st.file_uploader(
            "Upload Images",
            accept_multiple_files=True,
            key=st.session_state.upload_key
        )

        class_name = st.text_input("Enter Class Name")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save Images"):
                if uploaded_files and class_name:
                    save_uploaded_images(uploaded_files, class_name)
                    st.session_state.upload_message = "✅ Images saved!"
                    st.session_state.upload_key += 1
                    st.rerun()
                else:
                    st.warning("Provide class name and images")

        with col2:
            if st.button("Clear Upload"):
                st.session_state.upload_key += 1
                st.session_state.upload_message = "🧹 Upload cleared!"
                st.rerun()

        if st.session_state.upload_message:
            st.success(st.session_state.upload_message)

    # -------- TAB 2 --------
    with tab2:
        st.subheader("Train Model")

        if st.button("Start Training"):
            status = st.empty()
            status.warning("Training in progress... ⏳")

            subprocess.run(["python", "train_model.py"])

            status.success("Training Completed ✅")

    # -------- TAB 3 --------
    with tab3:
        st.subheader("Model Performance")

        if os.path.exists("accuracy.png"):
            st.image("accuracy.png", caption="Accuracy Graph")
        else:
            st.info("Train model first")

        if os.path.exists("loss.png"):
            st.image("loss.png", caption="Loss Graph")
        else:
            st.info("Train model first")
