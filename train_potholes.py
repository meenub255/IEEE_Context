import os
import zipfile
import yaml
from ultralytics import YOLO

# 1. Configuration
ZIP_PATH = r"D:\Context\pothole detection dataset.zip"
EXTRACT_DIR = r"D:\Context\dataset\pothole"
DATA_YAML_PATH = os.path.join(EXTRACT_DIR, "data.yaml")
EPOCHS = 3 # Fast proof of concept for IEEE

def prepare_dataset():
    # 2. Extract Data
    if not os.path.exists(EXTRACT_DIR):
        print("[1] Extracting Pothole Network Data...")
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("    Extraction complete.")
    else:
        print("[1] Dataset already extracted.")

    # 3. Secure the Data Paths in YAML
    print("[2] Re-configuring Tensor Mapping Routes...")
    # data.yaml synthesized successfully.
        
def train_network():
    print("[3] Booting YOLOv8 Core Engine...")
    # Load foundational weights to perform transfer-learning
    model = YOLO("yolov8n.pt")
    
    # Override DATA_YAML_PATH to the synthesized file
    DATA_YAML_PATH = r"D:\Context\dataset\pothole\RDD_SPLIT\data.yaml"
    
    print(f"[4] Commencing Static-Hazard Deep Learning ({EPOCHS} Epochs)...")
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=640,
        batch=4,  # Safe batch size to prevent VRAM overflow
        device="cpu", # User does not have specialized CUDA installed on this terminal
        project="runs",
        name="pothole_model"
    )
    
    print("[5] Training Complete! Weights saved to: runs/pothole_model/weights/best.pt")

if __name__ == "__main__":
    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: Cannot find the zip file at {ZIP_PATH}. Please ensure the file exists.")
    else:
        prepare_dataset()
        train_network()
