import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class BDD100kSpatialDataset(Dataset):
    """
    Dataset loader for BDD100k Images used to train the CNN Spatial Encoder.
    Reads images and extracts Weather & Road Context labels.
    """
    def __init__(self, images_dir, labels_json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform or self.default_transforms()
        
        # We define integer mapping for contexts based on video_pipeline.py MAP
        self.weather_map = {'clear': 0, 'rainy': 1, 'foggy': 2, 'snowy': 3}
        self.road_map = {'highway': 0, 'city street': 1, 'residential': 2}
        
        self.data_index = []
        
        print(f"Loading BDD100k annotations from {labels_json_path}...")
        try:
            with open(labels_json_path, 'r') as f:
                labels = json.load(f)
                
            for item in labels:
                img_name = item.get('name', '')
                full_path = os.path.join(images_dir, img_name)
                
                # Check if image actually exists on disk
                if not os.path.exists(full_path):
                    continue
                    
                attr = item.get('attributes', {})
                w = attr.get('weather', 'clear')
                r = attr.get('scene', 'highway')
                
                # Map to our integers, default to 0 if an unknown string appears
                w_idx = self.weather_map.get(w, 0)
                r_idx = self.road_map.get(r, 0)
                
                self.data_index.append({
                    'img_path': full_path,
                    'weather_idx': w_idx,
                    'road_idx': r_idx
                })
        except Exception as e:
            print(f"Warning: Could not parse real labels ({e}). Using synthetic distribution.")
            # For CPU dry run tests, generate 100 synthetic pointers
            self.data_index = [{'img_path': 'dummy', 'weather_idx': 0, 'road_idx': 0} for _ in range(100)]
            
        print(f"Dataset initialized with {len(self.data_index)} valid samples.")

    def default_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        item = self.data_index[idx]
        
        if item['img_path'] == 'dummy':
            # Generate synthetic noise image for pipeline testing 
            img_tensor = torch.randn(3, 224, 224)
        else:
            try:
                img = Image.open(item['img_path']).convert('RGB')
                img_tensor = self.transform(img)
            except:
                img_tensor = torch.zeros(3, 224, 224)
                
        return img_tensor, item['weather_idx'], item['road_idx']


class CrashTemporalDataset(Dataset):
    """
    Simulates sequence extraction from the Crash-1500 video dataset.
    Because reading hundreds of videos requires heavy I/O, this pre-computes 
    or simulates sequences of visual features (representing output from the spatial encoder).
    """
    def __init__(self, videos_dir, seq_len=15, is_hazard=True):
        self.videos_dir = videos_dir
        self.seq_len = seq_len
        self.is_hazard = is_hazard
        
        # Let's count available videos
        self.video_files = []
        if os.path.exists(videos_dir):
            import glob
            self.video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
            
        if len(self.video_files) == 0:
            print("Warning: No MP4 files found. Supplying synthetic trajectories.")
            self.num_samples = 100
        else:
            self.num_samples = len(self.video_files)
            
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # In a real heavy implementation, you would cv2.VideoCapture() the mp4, 
        # extract frames, run them through spatial_encoder, and return the [seq_len, 256] tensor.
        # Since this is CPU-optimized training, we return synthetic features that represent
        # the closing trajectory of an object right before impact.
        
        # Spatial features [Seq, 256]
        visual_seq = torch.randn(self.seq_len, 256)
        
        # Telemetry: [distance_m, rel_speed, friction] over time
        # Synthesize a car rapidly decreasing distance (closing in)
        start_dist = 60.0 + torch.rand(1).item() * 20.0
        closing_speed = 15.0 + torch.rand(1).item() * 10.0 # 15-25 m/s
        
        telemetry_seq = []
        for t in range(self.seq_len):
            current_dist = max(start_dist - (t * closing_speed / 30.0), 1.0) # assuming 30fps
            ttc_local = current_dist / closing_speed
            telemetry_seq.append([current_dist, closing_speed, 1.0]) # 1.0 = clear weather friction
            
        telemetry_seq = torch.tensor(telemetry_seq, dtype=torch.float32)
        
        # Ground Truth Context 
        weather_id = 0 # CLEAR
        road_id = 0    # HIGHWAY
        
        # Final target metrics
        target_hazard = 1.0 if self.is_hazard else 0.0
        target_ttc = current_dist / closing_speed # TTC at the final frame
        
        return visual_seq, telemetry_seq, weather_id, road_id, target_hazard, target_ttc

if __name__ == "__main__":
    print("Testing CPU Dataset Loaders...")
    ds_spatial = BDD100kSpatialDataset("d:/Context/dataset/bdd100k/images", "d:/Context/dataset/bdd100k/labels.json")
    img, w, r = ds_spatial[0]
    print(f"Spatial item -> Shape: {img.shape}, Weather: {w}, Road: {r}")
    
    ds_temporal = CrashTemporalDataset("d:/Context/dataset/crash1500")
    v, t, wid, rid, haz, ttc = ds_temporal[0]
    print(f"Temporal item -> Visual: {v.shape}, Telemetry: {t.shape}, Hazard: {haz}, Final TTC: {ttc:.2f}s")
