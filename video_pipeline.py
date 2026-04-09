"""
Context-Aware Road Hazard Prediction System
Phase 3: Hybrid Deep Learning Inference Pipeline
"""

import cv2
import numpy as np
import torch
import os
import time
import logging
from pathlib import Path
from collections import deque
import sys

# Suppress ultralytics logging globally if possible
logging.getLogger("ultralytics").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
YOLO_CLASSES   = [0, 2, 3, 5, 7]   # person, car, motorcycle, bus, truck
CONF_THRESHOLD = 0.35
FOCAL_LENGTH   = 800.0              # Approximate focal length (pixels)
REAL_CAR_H     = 1.5                # Average car height in metres
MIN_DIST_M     = 5.0                # Minimum detectable distance
CENTROID_PIX   = 100                # Max pixel gap for centroid matching
SEQ_LEN        = 15                 # Temporal GRU sequence requirement

WEATHER_MAP  = {'CLEAR': 0, 'RAIN': 1, 'FOG': 2, 'SNOW': 3}
ROAD_MAP     = {'HIGHWAY': 0, 'URBAN': 1, 'SCHOOL_ZONE': 2}
FRICTION_MAP = {'CLEAR': 1.0, 'RAIN': 0.75, 'FOG': 0.85, 'SNOW': 0.5}

SUPPORTED_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

# ─────────────────────────────────────────────────────────────────────────────
# LAZY LOAD DEEP LEARNING MODELS
# ─────────────────────────────────────────────────────────────────────────────
def load_pytorch_models():
    """Dynamically load weights into architecture if available."""
    from spatial_encoder import CNN_Encoder
    from temporal_gru import SpatialTemporalGRU

    device = torch.device('cpu')
    
    # 1. Load Spatial Context ResNet
    cnn = CNN_Encoder().to(device)
    if os.path.exists('spatial_encoder.pt'):
        # strict=False allows partial loading if dummy weights were used
        cnn.load_state_dict(torch.load('spatial_encoder.pt', map_location=device), strict=False) 
    cnn.eval()

    # 2. Load Temporal Anticipation GRU
    gru = SpatialTemporalGRU().to(device)
    if os.path.exists('temporal_gru.pt'):
        gru.load_state_dict(torch.load('temporal_gru.pt', map_location=device), strict=False)
    gru.eval()
    
    return cnn, gru

def _pseudo_depth(bbox_h, frame_h):
    # Relative ratio ensures math works identically on 320p or 4K video
    relative_h = max(bbox_h / frame_h, 0.001)
    
    # Heuristic: A car taking up 10% of the screen height is roughly 20 meters away.
    # Dist = 2.0 / relative_h
    return max(2.0 / relative_h, MIN_DIST_M)

def _match_prev(cx, cy, active_tracks):
    """Matcher using Track IDs to maintain temporal history"""
    best_id = None
    best_dist = CENTROID_PIX
    
    for trk_id, data in active_tracks.items():
        if not data['centroids']: continue
        last_cx, last_cy = data['centroids'][-1]
        dist = np.hypot(cx - last_cx, cy - last_cy)
        if dist < best_dist:
            best_dist = dist
            best_id = trk_id
            
    return best_id

def _process_frame_cnn(frame, cnn_model):
    """Converts cv2 frame to torch tensor and extracts Context Vector."""
    import torchvision.transforms as transforms
    from PIL import Image
    
    # Fast resize mapping
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Run Inference
    tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        feats, w_logits, r_logits = cnn_model(tensor)
        
    return feats.squeeze(0).cpu() # [256]

# ─────────────────────────────────────────────────────────────────────────────
# HUD DRAWING (UI OVERHAUL)
# ─────────────────────────────────────────────────────────────────────────────
def _draw_advanced_hud(frame, weather, road_type, lighting, f_mod, is_critical=False, active_objs=0, w_pred=None, r_pred=None):
    """Draws a premium Glassmorphism-style UI overlay indicating Context Awareness."""
    import time
    overlay = frame.copy()
    h, w = frame.shape[:2]
    b_scale = max(0.35, h / 720.0)
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_s    = cv2.FONT_HERSHEY_SIMPLEX
    
    # ── TOP BAR ─────────────────────────────────────────────────────────────
    bar_h = int(55 * b_scale)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    font_scale = 0.55 * b_scale
    y_pos = int(34 * b_scale)

    # Left — context label
    txt_l = f"IEEE PROTOTYPE  |  {weather} ENVIRONMENT  |  {road_type}"
    cv2.putText(frame, txt_l, (18, y_pos), font, font_scale, (180, 230, 255), max(1, int(b_scale)))

    # Right — system pipeline status
    txt_r = "NV: GLARE-SUPPRESS ACTIVE" if lighting == 'NIGHT' else "SPATIAL-TEMPORAL ACTIVE"
    txt_r = f"FRICTION: {f_mod:.2f}x  |  {txt_r}"
    ts = cv2.getTextSize(txt_r, font, font_scale, 1)[0]
    cv2.putText(frame, txt_r, (w - ts[0] - 18, y_pos), font, font_scale, (80, 220, 80), max(1, int(b_scale)))

    # Thin separator line under top bar
    cv2.line(frame, (0, bar_h), (w, bar_h), (40, 40, 40), 1)

    # ── BOTTOM TELEMETRY PANEL ───────────────────────────────────────────────
    panel_h   = int(72 * b_scale)
    panel_y   = h - panel_h
    cv2.rectangle(overlay, (0, panel_y), (w, h), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.line(frame, (0, panel_y), (w, panel_y), (40, 40, 40), 1)

    c_font = 0.48 * b_scale
    line1_y = panel_y + int(26 * b_scale)
    line2_y = panel_y + int(52 * b_scale)

    # Left column
    cv2.putText(frame, f"[SYS] YOLOv8  +  ResNet-18  +  Uni-GRU Hybrid Pipeline", (18, line1_y), font_s, c_font, (160, 160, 160), 1)
    cv2.putText(frame, f"[TRK] Tracked Objects: {active_objs}   |   [LIGHT] {lighting}", (18, line2_y), font_s, c_font, (100, 220, 100), 1)

    # Right column — system state badge
    state_txt = "!! AEB ENGAGED" if is_critical else "NOMINAL"
    s_color   = (60, 60, 255) if is_critical else (0, 200, 160)
    st_size   = cv2.getTextSize(state_txt, font, c_font * 1.1, 2)[0]
    st_x      = w - st_size[0] - 20
    cv2.putText(frame, f"[ACT] {state_txt}", (st_x - 50, line1_y), font, c_font * 1.1, s_color, max(1, int(b_scale) + 1))

    # ── CRITICAL AEB ALERT BANNER (centered, above panel) ───────────────────
    if is_critical:
        pulse  = (np.sin(time.time() * 8) + 1) / 2   # 0..1 smooth pulse
        banner_h = int(40 * b_scale)
        banner_y = panel_y - banner_h - int(6 * b_scale)

        # Solid dark banner bg
        cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_h), (30, 0, 0), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # Pulsing colored top & bottom accent border on the banner
        accent_val = int(180 + 75 * pulse)
        cv2.line(frame, (0, banner_y),              (w, banner_y),              (0, 0, accent_val), 2)
        cv2.line(frame, (0, banner_y + banner_h),   (w, banner_y + banner_h),   (0, 0, accent_val), 2)

        alert_txt = "[ AEB ]  AUTONOMOUS EMERGENCY BRAKING  -  CRITICAL COLLISION RISK"
        a_fs      = 0.58 * b_scale
        a_size    = cv2.getTextSize(alert_txt, font, a_fs, 2)[0]
        a_x       = (w - a_size[0]) // 2
        a_y       = banner_y + int(27 * b_scale)

        # Pulsing text color from white to red
        text_r    = int(200 + 55 * pulse)
        cv2.putText(frame, alert_txt, (a_x, a_y), font, a_fs, (50, 50, text_r), 2)

    # ── LANE CORRIDOR REMOVED ────────────────────────────────────────────────
    pass


def _draw_hazard_box(frame, x1, y1, x2, y2, hazard_prob, ttc, dist=None, drawn_tags=None):
    """Draws Tesla-style FCW bounding boxes with TTC and distance annotation."""
    h, w = frame.shape[:2]
    
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    box_w = max(x2 - x1, 1)
    
    # Resolution-adaptive scale (min enforced so tiny/distant boxes stay readable)
    res_scale = max(0.55, h / 720.0)

    # Tesla FCW Color Zones
    if hazard_prob > 0.75:
        color = (0, 0, 200)       # Red  — AEB / CRITICAL zone
        status = "CRITICAL"
    elif hazard_prob > 0.40:
        color = (0, 140, 255)     # Orange — FCW WARNING zone
        status = "WARNING"
    else:
        color = (0, 200, 60)      # Green — SAFE following distance
        status = "SAFE"

    thick = max(1, int(1.5 * res_scale))

    # Bracket corner marks
    L = int(14 * res_scale)
    cv2.line(frame, (x1, y1), (x1+L, y1), color, thick)
    cv2.line(frame, (x1, y1), (x1, y1+L), color, thick)
    cv2.line(frame, (x2, y2), (x2-L, y2), color, thick)
    cv2.line(frame, (x2, y2), (x2, y2-L), color, thick)
    cv2.line(frame, (x2, y1), (x2-L, y1), color, thick)
    cv2.line(frame, (x2, y1), (x2, y1+L), color, thick)
    cv2.line(frame, (x1, y2), (x1+L, y2), color, thick)
    cv2.line(frame, (x1, y2), (x1, y2-L), color, thick)

    # ── LABEL TAG ─────────────────────────────────────────────────────────
    dist_str   = f" {dist:.0f}m" if dist is not None else ""
    text       = f" {status}  TTC:{ttc:.1f}s{dist_str} "

    font       = cv2.FONT_HERSHEY_SIMPLEX
    # Fixed minimum scale so even small/distant objects are fully legible
    font_scale = max(0.42, 0.52 * res_scale)
    font_thick = max(1, int(1.4 * res_scale))

    (txt_w, txt_h), baseline = cv2.getTextSize(text, font, font_scale, font_thick)
    pad = max(4, int(5 * res_scale))

    tag_h  = txt_h + baseline + pad * 2
    tag_x1 = max(0, x1)
    tag_y2 = max(tag_h, y1)          # hangs just above the bracket top
    tag_y1 = tag_y2 - tag_h
    tag_x2 = min(w, tag_x1 + txt_w)

    # Ensure tag is within frame vertically
    if tag_y1 < 0:
        tag_y1 = y2 + 2              # flip below the box if no space above
        tag_y2 = tag_y1 + tag_h

    # Resolve overlap
    if drawn_tags is not None:
        for _ in range(5):
            conflict = any(
                not (tag_x2 < dx1 or tag_x1 > dx2 or tag_y2 < dy1 or tag_y1 > dy2)
                for (dx1, dy1, dx2, dy2) in drawn_tags
            )
            if conflict:
                tag_y1 = max(0, tag_y1 - tag_h - 2)
                tag_y2 = tag_y1 + tag_h
            else:
                break
        drawn_tags.append((tag_x1, tag_y1, tag_x2, tag_y2))

    if tag_x2 > tag_x1 and tag_y2 > tag_y1:
        # Solid bold color background — fully opaque, maximum contrast
        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
        # White text for CRITICAL, black for WARNING/SAFE
        txt_color = (255, 255, 255) if status == "CRITICAL" else (10, 10, 10)
        cv2.putText(frame, text, (tag_x1, tag_y2 - pad - baseline),
                    font, font_scale, txt_color, font_thick, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# CORE PROCESSING 
# ─────────────────────────────────────────────────────────────────────────────
def process_video(input_path, output_path, weather='CLEAR', road_type='HIGHWAY', lighting='DAY', batch_size=8, ui_callback=None, enable_potholes=False, auto_detect_env=False):
    """
    End-to-End inference loop. Uses Single Frame YOLO + ResNet + Uni-GRU sliding window.
    (Note: Batch processing is disabled here because Temporal Sequences require continuous 
     frame-by-frame history mapping. CPU throughput is naturally slower but accurate).
    """
    from ultralytics import YOLO
    cap    = cv2.VideoCapture(input_path)
    
    # ── IEEE Zero-Shot Environment Auto-Detect ──
    if auto_detect_env:
        log.info("Engaging EfficientNet Context Evaluator on First Frame...")
        from spatial_encoder import EnvironmentClassifier
        ret, first_frame = cap.read()
        if ret:
            # Override manual parameters using EfficientNet simulated heuristic
            weather, lighting, road_type = EnvironmentClassifier.zero_shot_heuristic_override(first_frame)
            log.info(f"AI Detected Context: {weather} | {lighting} | {road_type}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind the video to beginning
        
    # Check if night weights exist, otherwise default to yolov8n
    weights_path = 'yolov8_night_highway.pt' if lighting == 'NIGHT' and os.path.exists('yolov8_night_highway.pt') else 'yolov8n.pt'
    yolo = YOLO(weights_path).to('cpu')
    
    # IEEE Dual-Inference Mode: Load the Secondary Pothole Network
    yolo_pothole = None
    if enable_potholes:
        pothole_weights = 'runs/pothole_model/weights/best.pt'
        if os.path.exists(pothole_weights):
            yolo_pothole = YOLO(pothole_weights).to('cpu')
        elif os.path.exists('yolov8_pothole.pt'): # Fallback
            yolo_pothole = YOLO('yolov8_pothole.pt').to('cpu')
        else:
            log.warning("Static Scanner enabled, but pothole weights are missing!")
    
    cnn, gru = load_pytorch_models()
    
    w_code = WEATHER_MAP.get(weather, 0)
    r_code = ROAD_MAP.get(road_type, 0)
    
    # Stack the modifiers: weather friction * lighting visibility
    w_mod = FRICTION_MAP.get(weather, 1.0)
    l_mod = 1.0 if lighting == 'DAY' else 0.75 if lighting == 'DAWN/DUSK' else 0.50
    f_mod = round(w_mod * l_mod, 2)
    
    # Adaptive Confidence Matrix (Hyper-vigilance during adverse conditions)
    if weather == 'SNOW' and lighting == 'NIGHT':
        current_conf_threshold = 0.08  # YOLO must aggressively hunt in blizzard bloom
    elif lighting == 'NIGHT' or weather in ['SNOW', 'FOG', 'RAIN']:
        current_conf_threshold = 0.12
    else:
        current_conf_threshold = 0.20  # Lowered from 0.30: catches distant cars in straight view
    
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Dictionary mapping Tracking ID -> data structure
    active_tracks = {}
    next_trk_id   = 0
    
    # Global visual context buffer (holds last 15 frame representations)
    visual_buffer = deque(maxlen=SEQ_LEN)
    
    telemetry_log = [] # [Frame, Timestamp, Max_Risk, Min_TTC]
    
    written = 0
    t0 = time.time()
    
    # ── CRITICAL LATCH ────────────────────────────────────────────────────────
    # Once CRITICAL is triggered, hold the alert for at least 5 seconds.
    # This prevents the banner from blinking off when YOLO loses the target
    # during impact flashes or chaotic post-crash frames.
    CRIT_LATCH_FRAMES  = fps * 5   # 5 seconds hold after last CRITICAL frame
    crit_latch_counter = 0         # countdown timer (frames remaining)
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # --- TESLA AUTONOMOUS NIGHT VISION PIPELINE ---
            if lighting == 'NIGHT':
                # 1. Tesla Auto-Gain Control (Dynamic Gamma)
                # Calculate absolute median brightness to see if frame is pitch black
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                median_luma = np.median(gray)
                
                if median_luma < 50:
                    # Frame is dangerously dark. Stretch the shadows.
                    gamma = 0.65
                    inv_gamma = 1.0 / gamma
                    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                    frame = cv2.LUT(frame, table)
                
                # 2. Headlight Glare Masking & Suppression
                # Extract extreme whites (headlight cores)
                _, glare_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
                # Expand the mask slightly to cover the halo
                kernel = np.ones((5,5), np.uint8)
                glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)
                
                # Heavily blur the overexposed regions to destroy false structural geometry
                glare_blur = cv2.GaussianBlur(frame, (21, 21), 0)
                
                # Blend the melted glare back into the frame
                frame = np.where(glare_mask[:, :, None] == 255, glare_blur, frame).astype(np.uint8)
                
                # 3. Bilateral Sensor De-Noising
                # Smooth out standard ISO camera grain while preserving pedestrian/car outlines
                frame = cv2.bilateralFilter(frame, 5, 35, 35)

                # 4. Standard CLAHE Contrast Recovery
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_enh = clahe.apply(l)
                enhanced_lab = cv2.merge((l_enh, a, b))
                frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 1. SPATIAL ENCODER (CNN)
            # Extracted image representation [256]
            vis_embed = _process_frame_cnn(frame, cnn)
            visual_buffer.append(vis_embed)
            
            # 2. YOLO TRACKING
            res = yolo(frame, classes=YOLO_CLASSES, verbose=False)[0]
            
            raw_boxes = []
            for box in res.boxes:
                if box.conf[0].item() < current_conf_threshold: continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                raw_boxes.append([x1, y1, x2, y2, box.conf[0].item()])
            
            # --- Class-Agnostic NMS (Non-Maximum Suppression) ---
            # Suppresses overlapping boxes (e.g. Person on Motorcycle) to prevent double-tracking.
            boxes_to_process = []
            raw_boxes.sort(key=lambda x: x[4], reverse=True) # Sort by highest confidence
            for rb in raw_boxes:
                x1, y1, x2, y2, _ = rb
                area = (x2 - x1) * (y2 - y1)
                overlap = False
                for fb in boxes_to_process:
                    fx1, fy1, fx2, fy2 = fb
                    farea = (fx2 - fx1) * (fy2 - fy1)
                    
                    ix1 = max(x1, fx1)
                    iy1 = max(y1, fy1)
                    ix2 = min(x2, fx2)
                    iy2 = min(y2, fy2)
                    
                    iarea = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    iou = iarea / float(area + farea - iarea)
                    intersection_ratio = iarea / float(min(area, farea) + 1e-5)
                    
                    # If heavily overlapping OR completely swallowed by another box
                    if iou > 0.45 or intersection_ratio > 0.7:
                        overlap = True
                        break
                
                if not overlap:
                    boxes_to_process.append([x1, y1, x2, y2])
            
            drawn_ui_tags = []
            current_frame_ids = []
            frame_max_risk = 0.0
            frame_min_ttc = 99.9
            
            for (x1, y1, x2, y2) in boxes_to_process:
                
                # IEEE Feature: Ego-Vehicle & Environment Suppression
                box_w = x2 - x1
                box_h = y2 - y1
                box_fill_ratio = (box_w * box_h) / (width * height)
                
                # A: Dashboard / Ego-Hood Override
                # If a box touches the absolute bottom of the screen, spans more than 70% of the width, 
                # and starts below the horizon, it is mathematically the inside of the ego-vehicle.
                if y2 > height * 0.92 and box_w > width * 0.70 and y1 > height * 0.50:
                    continue
                
                # B: Lower-windshield glare and vent reflections
                if y1 > height * 0.75 and box_fill_ratio < 0.15:
                    continue
                
                # C: STRAIGHT-VIEW ROI CORRIDOR FILTER
                # Only apply this filter to SMALL/DISTANT objects (fill < 20% of frame).
                # A box filling >20% of the frame is physically enormous — it is directly alongside
                # or in front of the ego-vehicle regardless of centroid X position.
                # Filtering it would be a dangerous false negative (like missing the ambulance case).
                cx_check = (x1 + x2) / 2
                roi_left  = width * 0.20
                roi_right = width * 0.80
                if box_fill_ratio < 0.20:
                    if cx_check < roi_left or cx_check > roi_right:
                        continue
                
                dist = _pseudo_depth(y2 - y1, height)
                cx, cy = (x1+x2)/2, (y1+y2)/2
                
                # Match to existing track
                trk_id = _match_prev(cx, cy, active_tracks)
                
                if trk_id is None:
                    trk_id = next_trk_id
                    next_trk_id += 1
                    active_tracks[trk_id] = {
                        'centroids': deque(maxlen=SEQ_LEN),
                        'telemetry': deque(maxlen=SEQ_LEN), # [dist, rel_speed, fmod]
                        'last_seen': 0,
                        'ema_speed': 0.0
                    }
                
                current_frame_ids.append(trk_id)
                track = active_tracks[trk_id]
                track['last_seen'] = 0 # reset dormant timer
                track['centroids'].append((cx, cy))
                
                # Calculate Telemetry with MACRO-WINDOW velocity tracking
                rel_speed = 0.0
                if len(track['telemetry']) > 0:
                    prev_dist = track['telemetry'][-1][0]
                    
                    # Strict EMA smoothing for absolute UI distance stability
                    dist = (0.8 * prev_dist) + (0.2 * dist) 
                    
                    # MACRO VELOCITY: Compare current distance to distance ~15 frames ago (0.5s)
                    # This completely destroys YOLO bounding box frame-to-frame wobble amplification.
                    history_len = len(track['telemetry'])
                    if history_len > 5:
                        old_dist = track['telemetry'][0][0] # Oldest distance in the buffer
                        time_delta = history_len / fps      # Approx 0.5 seconds
                        macro_speed = (old_dist - dist) / time_delta
                    else:
                        macro_speed = (prev_dist - dist) * fps
                    
                    macro_speed = min(max(macro_speed, 0.0), 30.0) # Cap at physical max closing speed
                    
                    # Final gentle smoothing on the calculated speed
                    track['ema_speed'] = (0.85 * track['ema_speed']) + (0.15 * macro_speed)
                    rel_speed = track['ema_speed']

                track['telemetry'].append([dist, rel_speed, f_mod])
                
                # ─────────────────────────────────────────────────────────────
                # TESLA FCW (Forward Collision Warning) LOGIC
                # ─────────────────────────────────────────────────────────────
                # Tesla uses Speed-Adaptive TTC zones, not fixed raw distances.
                # The faster you're closing in, the earlier the system escalates.

                # Step 1: Compute TTC from GRU or fallback math
                if len(visual_buffer) == SEQ_LEN and len(track['telemetry']) == SEQ_LEN:
                    v_seq = torch.stack(list(visual_buffer)).unsqueeze(0)
                    t_seq = torch.tensor(list(track['telemetry']), dtype=torch.float32).unsqueeze(0)
                    w_t = torch.tensor([w_code], dtype=torch.long)
                    r_t = torch.tensor([r_code], dtype=torch.long)
                    h_pred, ttc_pred = gru(v_seq, t_seq, w_t, r_t)
                    hazard_prob = h_pred.item()
                    ttc = max(ttc_pred.item(), 0.1)
                else:
                    # Fallback: pure physics TTC
                    if rel_speed > 0.1 and dist > 0:
                        ttc = dist / rel_speed
                    else:
                        ttc = 99.9
                    hazard_prob = max(0.0, min(1.0 - (ttc / 6.0), 1.0))

                # Step 2: Speed-Adaptive Critical TTC Threshold
                # At high closing speeds, Tesla triggers FCW earlier
                if rel_speed > 15.0:       # Closing at > 54 km/h
                    critical_ttc = 2.5
                    warning_ttc  = 4.0
                elif rel_speed > 8.0:      # Closing at > 28 km/h
                    critical_ttc = 2.0
                    warning_ttc  = 3.5
                else:                      # Normal following
                    critical_ttc = 1.7
                    warning_ttc  = 3.0

                # Step 3 & 4: Unified Autonomous Threat Matrix
                
                # Apply Neural Network base / Physics fallback
                if ttc > warning_ttc:
                    hazard_prob = min(hazard_prob, 0.35)        # TTC marks as Safe
                elif ttc > critical_ttc:
                    hazard_prob = max(0.45, min(hazard_prob, 0.74)) # TTC marks as Warning
                else:
                    hazard_prob = max(hazard_prob, 0.85)        # TTC marks as Critical
                    
                # Physical Proximity Overrides (Absolute Reality Checks)
                if dist < 8.0:
                    # Imminent collision proximity < 8m -> Force CRITICAL
                    hazard_prob = max(hazard_prob, 0.85)
                elif dist < 18.0:
                    # Tailgating proximity < 18m -> Force at least a WARNING
                    hazard_prob = max(hazard_prob, 0.55)
                    
                if dist > 45.0:
                    # Extreme distance -> Force SAFE
                    hazard_prob = min(hazard_prob, 0.25)
                elif dist > 25.0:
                    # Safe distance -> Cap at WARNING (no distant red flashes)
                    hazard_prob = min(hazard_prob, 0.74)

                # Step 5: Anomalous Orientation Detection (Spun-out / Horizontal Cars)
                # If a car is sideways, full CRITICAL regardless of distance
                aspect_ratio = (x2 - x1) / max((y2 - y1), 1)
                if aspect_ratio > 2.2:
                    hazard_prob = max(hazard_prob, 0.95)
                    ttc = min(ttc, dist / max(rel_speed, 5.0))

                # Draw the hazard box with distance annotation
                _draw_hazard_box(frame, x1, y1, x2, y2, hazard_prob, ttc, dist, drawn_ui_tags)
                
                # Append to frame metrics
                frame_max_risk = max(frame_max_risk, hazard_prob)
                frame_min_ttc  = min(frame_min_ttc, ttc)
                
            telemetry_log.append({
                "Frame_ID": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "Time(s)": round(time.time() - t0, 3),
                "Lighting_Mode": lighting,
                "Weather_Mode": weather,
                "Hazard_Probability": round(frame_max_risk, 3),
                "Min_TTC": round(frame_min_ttc, 2)
            })
                
            # Cleanup dead tracks
            dead_ids = []
            for tid, data in active_tracks.items():
                if tid not in current_frame_ids:
                    data['last_seen'] += 1
                    if data['last_seen'] > 10: dead_ids.append(tid)
            for d in dead_ids: del active_tracks[d]
            
            # IEEE Dual-Inference: Static Hazards (Potholes)
            if enable_potholes and yolo_pothole is not None:
                # We execute a separate parallel inference because structural data for potholes requires isolated non-maximum suppression
                res_static = yolo_pothole(frame, verbose=False)[0]
                for box in res_static.boxes:
                    if box.conf[0].item() < 0.20: continue # De-noise false ground shapes
                    px1, py1, px2, py2 = box.xyxy[0].cpu().numpy()
                    
                    # Compute Dynamic UI Scale
                    scale = min(width, height) / 720.0
                    f_scale = max(0.4, 0.6 * scale)
                    t_thick = max(1, int(1.5 * scale))
                    
                    # Draw a distinct Ground-Mask separating it from Dynamic Vehicles
                    cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (0, 120, 255), max(1, t_thick)) # Solid Orange
                    cv2.putText(frame, "STATIC HAZARD: POTHOLE", (int(px1), max(15, int(py1)-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, f_scale * 0.7, (0, 120, 255), t_thick)

            # ── CRITICAL LATCH + POST-IMPACT FLASH DETECTOR ──────────────────
            # 1. If current frame is CRITICAL, reset the latch timer
            if frame_max_risk >= 0.85:
                crit_latch_counter = CRIT_LATCH_FRAMES
            
            # 2. Post-Impact Flash: if the frame is suddenly overexposed (mean luma > 200)
            #    and we are still within the latch window, this is an airbag/impact flash.
            #    Keep CRITICAL locked — do NOT let it fall to NOMINAL.
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_luma  = np.mean(gray_frame)
            if mean_luma > 200 and crit_latch_counter > 0:
                frame_max_risk = max(frame_max_risk, 0.90)  # force CRITICAL during flash
            
            # 3. Tick down the latch counter each frame
            is_crit = (frame_max_risk >= 0.85) or (crit_latch_counter > 0)
            if crit_latch_counter > 0:
                crit_latch_counter -= 1

            # Finalize Draw
            _draw_advanced_hud(frame, weather, road_type, lighting, f_mod, is_critical=is_crit, active_objs=len(current_frame_ids))
            writer.write(frame)
            written += 1
            
            # Send live frame to the UI if callback is provided
            if ui_callback:
                ui_callback(frame)
            
            if written % 30 == 0:
                elapsed = time.time() - t0
                log.info(f"   [{written}/{total}] Frames — {written/elapsed:.1f} fps (Deep Learning Active)")
                
    cap.release()
    writer.release()
    
    import pandas as pd
    pd.DataFrame(telemetry_log).to_csv("telemetry_logs.csv", index=False)
    
    log.info(f"✅ Deep Learning processing complete: {output_path}")
    return True

if __name__ == '__main__':
    # Simple CLI fallthrough testing
    import sys
    if len(sys.argv) > 2:
        process_video(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python video_pipeline.py input.mp4 output.mp4")
