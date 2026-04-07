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
def _draw_advanced_hud(frame, weather, road_type, f_mod, w_pred=None, r_pred=None):
    """Draws a premium Glassmorphism-style UI overlay indicating Context Awareness."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    b_scale = max(0.35, h / 720.0)
    
    # Top Bar Dark gradient dynamically sized
    bar_h = int(60 * b_scale)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Dynamic text parameters
    font_scale = 0.6 * b_scale
    y_pos = int(35 * b_scale)
    
    # Left Context Data
    txt_l = f"IEEE PROTOTYPE | {weather} ENVIRONMENT"
    cv2.putText(frame, txt_l, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale, (200, 255, 255), max(1, int(b_scale)))
    
    # Right Neural Network Diagnostics
    txt_r = f"FRICTION: {f_mod:.2f}x | SPATIAL-TEMPORAL ACTIVE"
    text_size = cv2.getTextSize(txt_r, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)[0]
    cv2.putText(frame, txt_r, (w - text_size[0] - 20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), max(1, int(b_scale)))


def _draw_hazard_box(frame, x1, y1, x2, y2, hazard_prob, ttc):
    """Draws sleek UI bounding boxes representing neural risk predictions."""
    h, w = frame.shape[:2]
    b_scale = max(0.35, h / 720.0)  # Dynamic Resolution Auto-Scaling
    
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
    # Color mapping based on GRU Hazard Prob
    if hazard_prob > 0.75:
        color = (0, 0, 255)       # Red - High Risk
        status = "CRITICAL"
        thick = 3
    elif hazard_prob > 0.40:
        color = (0, 165, 255)     # Orange - Warning
        status = "WARNING"
        thick = 2
    else:
        color = (0, 255, 100)     # Green - Safe
        status = "SAFE"
        thick = 2

    # Larger bracket corners for visibility at distance
    L = int(28 * b_scale)
    cv2.line(frame, (x1, y1), (x1+L, y1), color, thick+1)
    cv2.line(frame, (x1, y1), (x1, y1+L), color, thick+1)
    cv2.line(frame, (x2, y2), (x2-L, y2), color, thick+1)
    cv2.line(frame, (x2, y2), (x2, y2-L), color, thick+1)
    cv2.line(frame, (x2, y1), (x2-L, y1), color, thick+1)
    cv2.line(frame, (x2, y1), (x2, y1+L), color, thick+1)
    cv2.line(frame, (x1, y2), (x1+L, y2), color, thick+1)
    cv2.line(frame, (x1, y2), (x1, y2-L), color, thick+1)

    # ── Two-line data tag ──
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65 * b_scale
    font_thick = max(1, int(2 * b_scale))

    line1 = status                                    # e.g. SAFE / WARNING / CRITICAL
    line2 = f"RISK: {hazard_prob*100:.0f}%  TTC: {ttc:.1f}s"

    (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, font_thick)
    (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, font_thick)

    pad_h = int(12 * b_scale)
    pad_w = int(8 * b_scale)
    
    tag_w   = max(w1, w2) + (pad_w * 2)
    tag_h   = h1 + h2 + pad_h       # padding between lines + top/bottom
    tag_x1  = x1
    tag_y1  = max(y1 - tag_h - 4, 0)
    tag_x2  = tag_x1 + tag_w
    tag_y2  = max(tag_y1 + tag_h, y1)

    # Solid filled background
    cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)

    # Line 1 — Status label  (black text on coloured bg)
    y_line1 = tag_y1 + h1 + int(pad_h * 0.3)
    cv2.putText(frame, line1, (tag_x1 + pad_w, y_line1),
                font, font_scale, (0, 0, 0), font_thick + 1)

    # Line 2 — Risk + TTC (dark text for legibility)
    cv2.putText(frame, line2, (tag_x1 + pad_w, y_line1 + h2 + int(pad_h * 0.5)),
                font, font_scale, (20, 20, 20), font_thick)

# ─────────────────────────────────────────────────────────────────────────────
# CORE PROCESSING 
# ─────────────────────────────────────────────────────────────────────────────
def process_video(input_path, output_path, weather='CLEAR', road_type='HIGHWAY', lighting='DAY', batch_size=8, ui_callback=None, enable_potholes=False):
    """
    End-to-End inference loop. Uses Single Frame YOLO + ResNet + Uni-GRU sliding window.
    (Note: Batch processing is disabled here because Temporal Sequences require continuous 
     frame-by-frame history mapping. CPU throughput is naturally slower but accurate).
    """
    from ultralytics import YOLO
    
    log.info(f"▶ Initializing Deep Learning Context Pipeline for: {Path(input_path).name}")
    
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
        current_conf_threshold = 0.15
    else:
        current_conf_threshold = 0.30
    
    cap    = cv2.VideoCapture(input_path)
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
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # --- CLAHE ENHANCEMENT FOR NIGHT VISIBILITY ---
            if lighting == 'NIGHT':
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
            
            boxes_to_process = []
            for box in res.boxes:
                if box.conf[0].item() < current_conf_threshold: continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes_to_process.append([x1, y1, x2, y2])
            
            current_frame_ids = []
            frame_max_risk = 0.0
            frame_min_ttc = 99.9
            
            for (x1, y1, x2, y2) in boxes_to_process:
                
                # IEEE Feature: Ego-Vehicle & Environment Suppression
                box_area = (x2 - x1) * (y2 - y1)
                screen_area = width * height
                
                # A: Strict dashboard/hood cutoff. Any boxes strictly isolated to the bottom 25% of the frame are physical car parts.
                if y1 > height * 0.75:
                    continue
                # B: Massive boxes glued to the bottom
                if box_area > screen_area * 0.25 and y2 > height * 0.90 and y1 > height * 0.50: 
                    continue
                
                # If a box is massive but sits in the middle of the screen (y1 < 0.5 * height), 
                # it's a massive vehicle directly in front of us (e.g. total collision). Do NOT suppress it!
                
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
                
                # Calculate Telemetry with HEAVY EMA (Exponential Moving Average) smoothing
                rel_speed = 0.0
                if len(track['telemetry']) > 0:
                    prev_dist = track['telemetry'][-1][0]
                    
                    # 90% strict EMA smoothing to entirely kill YOLO pixel jumping
                    dist = (0.9 * prev_dist) + (0.1 * dist) 
                    
                    raw_rel_speed = (prev_dist - dist) * fps
                    raw_rel_speed = min(max(raw_rel_speed, 0.0), 30.0) # hard cap internal physics
                    
                    # Smooth the velocity itself so it doesn't spike from 0 to 30 instantly
                    track['ema_speed'] = (0.8 * track['ema_speed']) + (0.2 * raw_rel_speed)
                    rel_speed = track['ema_speed']

                track['telemetry'].append([dist, rel_speed, f_mod])
                
                # 3. TEMPORAL INFERENCE (GRU)
                # Only execute if we have a full sequence to predict correctly
                hazard_prob = 0.0
                ttc = 99.9

                if len(visual_buffer) == SEQ_LEN and len(track['telemetry']) == SEQ_LEN:
                    # Construct Tensor Payload: [Batch=1, Seq=15, Dim]
                    v_seq = torch.stack(list(visual_buffer)).unsqueeze(0) # [1, 15, 256]
                    t_seq = torch.tensor(list(track['telemetry']), dtype=torch.float32).unsqueeze(0) # [1, 15, 3]

                    w_t = torch.tensor([w_code], dtype=torch.long)
                    r_t = torch.tensor([r_code], dtype=torch.long)
                    
                    # Forward Pass
                    h_pred, ttc_pred = gru(v_seq, t_seq, w_t, r_t)
                    
                    hazard_prob = h_pred.item()
                    ttc = ttc_pred.item()
                    if ttc < 0: ttc = 99.9 # bounds check
                    
                else:
                    # Fallback math if sequence isn't ready
                    if rel_speed > 0.5:
                        ttc = dist / rel_speed
                        hazard_prob = min(max((5.0 - ttc)/5.0, 0.0), 1.0) # heuristic fallback during 0.5s warmup
                
                # IEEE Presentation Fix: Spatial Distance Decay & Orientation Anomalies
                # 1. Distant cars should mathematically not be labeled CRITICAL. 
                if dist > 40.0:
                    hazard_prob = min(hazard_prob, 0.3)  # Forces GREEN (SAFE)
                elif dist > 20.0:
                    hazard_prob = min(hazard_prob, 0.6)  # Forces ORANGE (WARNING)
                    
                # 2. Anomalous Orientation Detection (Spun-out / Horizontal Cars)
                aspect_ratio = (x2 - x1) / max((y2 - y1), 1)
                
                # Only apply orientation logic to YOLO boxes
                if aspect_ratio > 2.2:
                    # If a car is horizontal, it is blocking the road (spun out or T-bone accident).
                    # Overrule the distance decay and instantly flag as a severe threat.
                    hazard_prob = max(hazard_prob, 0.95)
                    ttc = min(ttc, dist / 20.0) # Assume highway speed approach
                    
                # Draw
                _draw_hazard_box(frame, x1, y1, x2, y2, hazard_prob, ttc)
                
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

            # Finalize Draw
            _draw_advanced_hud(frame, weather, road_type, f_mod)
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
