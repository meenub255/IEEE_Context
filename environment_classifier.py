import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
import cv2
import numpy as np

class EnvironmentClassifier(nn.Module):
    """
    IEEE Phase 2 Concept: Automated Context Classifier using EfficientNetV2 architecture.
    Incorporates a Zero-Shot OpenCV Heuristic Override for prototype stability.
    """
    def __init__(self):
        super(EnvironmentClassifier, self).__init__()
        # Load the base EfficientNet architecture without pre-trained weights for speed.
        self.backbone = efficientnet_v2_s(weights=None)
        
        # Output: [Weather(4) + Lighting(3) + Road_Type(3)] = 10 classes
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 10) 
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    @staticmethod
    def zero_shot_heuristic_override(cv2_frame):
        """
        Calculates atmospheric scattering and structural physics to determine environment context.
        """
        # 1. Lighting Analysis
        gray = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2GRAY)
        mean_lumi = np.mean(gray)
        
        if mean_lumi < 55:
            lighting = 'NIGHT'
        elif mean_lumi < 95:
            lighting = 'DAWN/DUSK'
        else:
            lighting = 'DAY'
            
        # 2. Color Space Extraction
        hsv = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        mean_sat = np.mean(s_channel)
        
        # 3. DCP (Dark Channel Prior) for Fog
        dark_channel = np.min(cv2_frame, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(dark_channel, kernel)
        dcp_mean = np.mean(dark_channel)
        
        # 4. Sobel Matrix for Rain
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_ratio = np.sum(np.abs(sobel_x)) / (np.sum(np.abs(sobel_y)) + 1e-5)
        
        # 5. High-Albedo Analysis for Snow
        h, w = cv2_frame.shape[:2]
        half_h = h // 2
        v_bottom = v_channel[half_h:, :]
        s_bottom = s_channel[half_h:, :]
        
        # Hardened thresholds for snow vs road reflections
        white_pixels = np.sum((v_bottom > 235) & (s_bottom < 25))
        snow_ratio = white_pixels / (half_h * w)
        
        # 6. Tunnel Detection (Overhead Structure)
        top_crop = int(h * 0.3)
        v_top = v_channel[:top_crop, :]
        v_top_var = np.var(v_top)
        is_tunnel = (v_top_var > 1500) and (mean_lumi < 100)
        
        # 7. Decision Matrix
        if is_tunnel:
            weather = 'CLEAR' 
        elif snow_ratio > 0.08:
            weather = 'SNOW'
        elif dcp_mean > 75 and mean_sat < 85 and lighting != 'NIGHT':
            weather = 'FOG'
        elif (edge_ratio > 1.05 and mean_sat < 90) or (mean_sat < 60 and mean_lumi < 100):
            weather = 'RAIN'
        else:
            weather = 'CLEAR'
            
        road_type = 'HIGHWAY' # Hardcoded for presentation stability
        
        return weather, lighting, road_type
