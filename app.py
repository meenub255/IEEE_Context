import streamlit as st
import os
import time
from video_pipeline import process_video

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Context-Aware Road Hazard Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── LIGHT THEME (ACADEMIC STYLE) ────────────────────────────────────── */
    :root {
        --bg-deep: #f0f2f6;
        --bg-surface: #ffffff;
        --accent-blue: #0969da;
        --accent-cyan: #0599a8;
        --accent-orange: #d46000;
        --accent-red: #cf222e;
        --text-main: #1f2328;
        --text-dim: #636c76;
        --glass-bg: rgba(255, 255, 255, 0.85);
        --glass-border: rgba(31, 35, 40, 0.12);
    }

    .stApp {
        background-color: var(--bg-deep);
        color: var(--text-main);
        font-family: 'Outfit', sans-serif;
    }

    /* ── TITLES & HEADERS ────────────────────────────────────────────────── */
    h1 {
        color: var(--accent-blue) !important;
        font-weight: 600 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.04em !important;
        margin-bottom: 0.5rem !important;
    }

    h2, h3 {
        color: var(--accent-blue) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.2rem !important;
        margin-bottom: 0rem !important;
    }

    /* ── CONTAINERS ──────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--glass-border) !important;
    }

    /* Tighten up sidebar element spacing */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.1rem !important;
    }

    /* Slashing divider margins */
    [data-testid="stSidebar"] hr {
        margin: 0.5rem 0 !important;
    }

    /* Force all sidebar text to be dark directly */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #1f2328 !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] h3 {
        color: var(--accent-blue) !important;
        font-weight: 600 !important;
    }

    div.element-container:has(div.stAlert), 
    div.element-container:has(section[data-testid="stFileUploadDropzone"]) {
        background: var(--bg-surface);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 5px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    /* ── INTERACTIVE ELEMENTS ───────────────────────────────────────────── */
    .stButton > button {
        background: var(--accent-blue) !important;
        color: white !important;
        border-radius: 50px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        transition: all 0.2s !important;
        width: 100%;
    }

    .stButton > button:hover {
        background: #085cc0 !important;
        box-shadow: 0 4px 12px rgba(9, 105, 218, 0.3) !important;
    }

    div[data-baseweb="tab-list"] {
        background-color: #e6edf3 !important;
        border-radius: 50px !important;
        padding: 6px !important;
        border: 1px solid var(--glass-border) !important;
        margin-bottom: 2rem;
    }

    div[data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 50px !important;
        padding: 8px 25px !important;
    }

    /* Force tab label text visibility */
    div[data-baseweb="tab"] p, 
    div[data-baseweb="tab"] span {
        color: #4b535d !important;
        font-weight: 600 !important;
    }

    div[aria-selected="true"] {
        background-color: var(--bg-surface) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }

    div[aria-selected="true"] p,
    div[aria-selected="true"] span {
        color: var(--accent-blue) !important;
    }

    /* ── MEDIA & VISUALS ───────────────────────────────────────────────── */
    video, img {
        border-radius: 12px !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
    }

    /* Exhaustive uploader nuke */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploaderFileData"],
    div[data-testid="stFileUploaderFile"],
    section[data-testid="stFileUploadDropzone"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        fill: #000000 !important;
    }

    /* Target the 'Browse files' button specifically */
    [data-testid="stFileUploader"] button {
        border: 1px solid #d0d7de !important;
        background-color: #f6f8fa !important;
        color: #000000 !important;
    }

    div.stAlert {
        background: #fff !important;
        border: 1px solid #d0d7de !important;
        border-radius: 12px !important;
        color: #1f2328 !important;
    }

    /* Force success text in sidebar to be dark and compact */
    [data-testid="stSidebar"] div.stAlert {
        padding: 0.5rem 0.75rem !important;
        margin-top: 0.25rem !important;
    }
    
    [data-testid="stSidebar"] div.stAlert p {
        color: #1f2328 !important;
        font-size: 0.85rem !important;
    }

    /* ── SCROLLBAR ─────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb { background: #d0d7de; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #afb8c1; }

    /* Kill default blocks */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1.5rem 5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Video Input")
    uploaded_file = st.file_uploader("Browse dashcam clip", type=["mp4", "avi"])
    
    st.divider()

    st.markdown("### 🛠️ Neural Architecture")
    auto_detect = st.toggle("🤖 Auto-Detect Environment (EfficientNet)", value=True)
    enable_potholes = st.toggle("Enable Pothole Scanning (Static Hazards)", value=False)
    
    st.divider()

    st.markdown("### 🌍 Environmental Context")
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
    st.caption("YOLOv8n · ResNet-18 · Uni-GRU")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1>🛡️ Context-Aware Road Hazard Prediction System</h1>", unsafe_allow_html=True)
st.divider()

if uploaded_file is not None:
    temp_input  = "temp_input.mp4"
    temp_output = "temp_output.mp4"

    with open(temp_input, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # IEEE Pre-Inference Scene Preview
    import cv2
    try:
        if auto_detect:
            from environment_classifier import EnvironmentClassifier
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

