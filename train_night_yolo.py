from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Initialize the base highway model we have been using
    print("Loading base YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Path to the newly formatted ACDC dataset YAML
    data_path = r"d:\Context\dataset\acdc\data.yaml"
    
    print("\n[+] Initiating Zero-Lux 'Night-Time Highway' Fine-Tuning...")
    print(f"[>] Dataset Target: {data_path}")
    print("[!] Running natively on CPU. Multi-threading initialized.")
    
    # We run 3 robust epochs on CPU to inject the adverse conditions weights
    # without taking 15 hours. The batch size is reduced to 8 to avoid OOM.
    model.train(
        data=data_path,
        epochs=3,
        batch=8,
        imgsz=640,
        device='cpu',
        workers=4,
        optimizer='auto',
        project='runs/detect',
        name='yolov8_night_highway'
    )
    
    print("\n✅ Training Complete. The robust Night-Time weights have been saved.")
