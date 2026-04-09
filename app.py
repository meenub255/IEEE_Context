import streamlit as st
import os
import time
from video_pipeline import process_video

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Deep Hazard System", layout="wide", page_icon="🚘")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* Base dark theme */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }

    /* Title */
    h1 {
        color: #58a6ff !important;
        font-weight: 500 !important;
        font-size: 1.4rem !important;
        letter-spacing: -0.3px;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    h2, h3 {
        color: #8b949e !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #c9d1d9 !important;
        font-size: 0.85rem !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        width: 100%;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(56, 139, 253, 0.45);
    }

    /* Video player — fill the available width */
    video {
        width: 100% !important;
        max-height: 75vh !important;
        border-radius: 10px;
        background: #000;
    }

    /* Image frames from OpenCV — full width */
    img {
        border-radius: 8px;
    }

    /* Make the stImage container flex so it fills width */
    div[data-testid="stImage"] {
        width: 100% !important;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    div[data-baseweb="tab"] {
        background-color: transparent;
        color: #8b949e;
        font-size: 0.85rem;
        border-radius: 6px;
    }
    div[aria-selected="true"] {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
    }

    /* Remove Streamlit footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Override default block padding that makes video small */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    /* ── Kill all white backgrounds ─────────────────────────────────────── */
    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"],
    div[data-testid="stTabContent"],
    div[data-testid="stEmpty"],
    div[class*="uploadedFile"] {
        background-color: transparent !important;
    }

    /* File uploader dropzone */
    section[data-testid="stFileUploadDropzone"] {
        background-color: #161b22 !important;
        border: 1px dashed #30363d !important;
        border-radius: 8px !important;
    }

    /* Generic element containers */
    .element-container, .stMarkdown {
        background: transparent !important;
    }

    /* Video container — black behind the player, no white flash */
    div[data-testid="stVideo"] > div,
    div[data-testid="stVideo"] iframe {
        background: #000 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ Neural Architecture")
    auto_detect = st.toggle("🤖 Auto-Detect Environment (EfficientNet)", value=True)
    enable_potholes = st.toggle("Enable Pothole Scanning (Static Hazards)", value=False)
    st.divider()

    st.markdown("### 🌍 Environment Context")
    if auto_detect:
        st.success("Autonomous Scene Classification Active")
        weather   = "CLEAR"
        lighting  = "DAY"
        road_type = "HIGHWAY"
    else:
        st.markdown("Inject real-world conditions manually.")
        weather   = st.selectbox("Weather", ["CLEAR", "RAIN", "FOG", "SNOW"])
        lighting  = st.selectbox("Lighting", ["DAY", "DAWN/DUSK", "NIGHT"])
        road_type = st.selectbox("Road Type", ["HIGHWAY", "URBAN", "SCHOOL_ZONE"])
    
        w_map = {"CLEAR": 1.0, "RAIN": 0.75, "FOG": 0.85, "SNOW": 0.5}
        l_map = {"DAY": 1.0, "DAWN/DUSK": 0.75, "NIGHT": 0.5}
        
        total_mod = w_map[weather] * l_map[lighting]
        st.info(f"Manual Grip/Vis Modifier: **{total_mod:.2f}x**")
        
    st.divider()

    st.markdown("### 📂 Video Input")
    uploaded_file = st.file_uploader("Browse dashcam clip", type=["mp4", "avi"])
    
    st.divider()
    st.caption("YOLOv8n · ResNet-18 · Uni-GRU")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1>🛡️ Context-Aware Deep Hazard Prediction</h1>", unsafe_allow_html=True)
st.divider()

tab1, tab2 = st.tabs(["🎥 Live Inference", "🧠 Architecture"])

with tab1:
    if uploaded_file is None:
        st.info("👈 Upload a dashcam video from the sidebar to begin.")
    else:
        temp_input  = "temp_input.mp4"
        temp_output = "temp_output.mp4"

        with open(temp_input, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # IEEE Pre-Inference Scene Preview
        import cv2
        try:
            if auto_detect:
                from spatial_encoder import EnvironmentClassifier
                cap_preview = cv2.VideoCapture(temp_input)
                ret_p, frame_p = cap_preview.read()
                cap_preview.release()
                if ret_p:
                    p_w, p_l, p_r = EnvironmentClassifier.zero_shot_heuristic_override(frame_p)
                    st.caption(f"🤖 **EfficientNet Scene Preview:** {p_w} Weather | {p_l} | {p_r}")
        except Exception as e:
            pass

        # Engage button sits above the video
        run_btn = st.button("🚀 Engage Deep Learning Framework")

        # Full-width frame placeholder
        frame_placeholder = st.empty()

        if not run_btn:
            # Show the raw video, filling the entire content width
            frame_placeholder.video(temp_input, format="video/mp4")
        else:
            st.toast("Initializing PyTorch Tensor Bindings...", icon="⚙️")

            import cv2

            def render_frame(frame):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb)

            t0 = time.time()
            success = process_video(
                temp_input, temp_output,
                weather=weather, road_type=road_type, lighting=lighting,
                ui_callback=render_frame, enable_potholes=enable_potholes, auto_detect_env=auto_detect
            )
            elapsed = time.time() - t0

            if success:
                st.success(f"✅ Processing complete in {elapsed:.1f}s")
                if os.path.exists("telemetry_logs.csv"):
                    with open("telemetry_logs.csv", "rb") as csv_file:
                        st.download_button(
                            label="📊 Download Telemetry Logs (CSV)",
                            data=csv_file,
                            file_name="hazard_telemetry_logs.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        # Cleanup
        for f in [temp_input, temp_output]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

with tab2:
    st.markdown("### Spatial-Temporal Architecture")
    
    st.markdown("""
    ```mermaid
    graph TD
        A[Raw HD Video Frame] --> B{Lighting Condition}
        B -->|NIGHT| C[CLAHE Contrast Recovery]
        B -->|DAY| D[Standard Pass]
        C --> E
        D --> E
        
        E[YOLOv8 Object Tracking] --> G[Spatial Bounding Boxes]
        E[ResNet-18 Encoder] --> H[256-Dim Environment Vector]
        
        I[Context Matrix: Friction * Vis] --> J[Uni-GRU Temporal Engine]
        G --> J
        H --> J
        
        J --> K((Hazard Risk %))
        J --> L((Min TTC Threshold))
    ```
    """)
    
    st.markdown("""
**Stage 1 — Spatial Encoder (ResNet-18)**
- Extracts a 256-dim environment feature vector from each dashcam frame.
- Multi-task heads classify Weather type and Road topology.

**Stage 2 — Temporal Anticipation Engine (Uni-GRU)**
- Maintains a 15-frame sliding window of spatial features + telemetry.
- Predicts continuous **Hazard Probability** and **Time-To-Collision (TTC)**.
- Runs entirely on CPU — no GPU required.
    """)
    st.info("`spatial_encoder.pt` and `temporal_gru.pt` are loaded at inference start.")
