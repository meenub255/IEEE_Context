import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from spatial_encoder import CNN_Encoder
from temporal_gru import SpatialTemporalGRU
from dataset_loader import BDD100kSpatialDataset, CrashTemporalDataset

def train_spatial(epochs, batch_size):
    print("="*50)
    print("STAGE 1: Training Spatial Encoder (BDD100k)")
    print("="*50)
    
    # 1. Setup Model (CPU explicitly)
    device = torch.device("cpu")
    model = CNN_Encoder().to(device)
    
    # 2. Setup Dataset
    dataset = BDD100kSpatialDataset("d:/Context/dataset/bdd100k", "d:/Context/dataset/bdd100k/meta.json")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Setup Optimizers
    criterion_weather = nn.CrossEntropyLoss()
    criterion_road = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = time.time()
        
        for i, (images, w_targets, r_targets) in enumerate(loader):
            images = images.to(device)
            w_targets = w_targets.to(device)
            r_targets = r_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            _, w_preds, r_preds = model(images)
            
            # Multi-Task Loss Calculate 
            loss_w = criterion_weather(w_preds, w_targets)
            loss_r = criterion_road(r_preds, r_targets)
            loss = loss_w + loss_r
            
            # Backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 50 == 0:
                print(f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")
                
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} Complete. Avg Loss: {running_loss/len(loader):.4f} | Time: {epoch_time:.2f}s")
        
    # Save Weights
    torch.save(model.state_dict(), "spatial_encoder.pt")
    print("Saved Spatial Encoder weights to spatial_encoder.pt")


def train_temporal(epochs, batch_size):
    print("="*50)
    print("STAGE 2: Training Temporal Anticipation GRU (Crash-1500)")
    print("="*50)
    
    device = torch.device("cpu")
    model = SpatialTemporalGRU().to(device)
    
    dataset = CrashTemporalDataset("d:/Context/dataset/crash1500", seq_len=15, is_hazard=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Hazard is Binary (0-1), TTC is Regression continuous output
    criterion_hazard = nn.BCELoss()
    criterion_ttc = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = time.time()
        
        for i, (v_seq, t_seq, w_id, r_id, h_target, ttc_target) in enumerate(loader):
            v_seq = v_seq.to(device)
            t_seq = t_seq.to(device)
            w_id = w_id.to(device)
            r_id = r_id.to(device)
            
            # Format targets
            h_target = h_target.to(device).unsqueeze(1).float() # [Batch, 1]
            ttc_target = ttc_target.to(device).unsqueeze(1).float() # [Batch, 1]
            
            optimizer.zero_grad()
            
            # Forward
            h_preds, ttc_preds = model(v_seq, t_seq, w_id, r_id)
            
            # Multi-Task Loss Calculate 
            loss_h = criterion_hazard(h_preds, h_target)
            loss_t = criterion_ttc(ttc_preds, ttc_target)
            
            # Heavy penalty for TTC mistakes closer to 0
            loss = loss_h + (0.5 * loss_t)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 50 == 0:
                print(f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f} (Haz: {loss_h.item():.4f}, TTC: {loss_t.item():.4f})")
                
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} Complete. Avg Loss: {running_loss/len(loader):.4f} | Time: {epoch_time:.2f}s")
        
    torch.save(model.state_dict(), "temporal_gru.pt")
    print("Saved Temporal GRU weights to temporal_gru.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["spatial", "temporal", "both"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    if args.stage in ["spatial", "both"]:
        train_spatial(args.epochs, args.batch_size)
        
    if args.stage in ["temporal", "both"]:
        train_temporal(args.epochs, args.batch_size)
