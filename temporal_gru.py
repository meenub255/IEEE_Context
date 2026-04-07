import torch
import torch.nn as nn

class SpatialTemporalGRU(nn.Module):
    """
    CPU-Optimized Temporal Uni-GRU Model
    Takes sequences of visual embeddings and telemetry (distance, relative_speed)
    and predicts continuous Probability of Hazard and estimated Time-to-Collision.
    """
    def __init__(self, cnn_embed_dim=256, telemetry_dim=3, context_dim=32, hidden_dim=128, num_layers=1):
        """
        telemetry_dim: [distance_m, relative_speed_ms, friction_modifier]
        context_dim: Size of the embedded representation for Weather/Road (e.g. 16+16=32)
        """
        super(SpatialTemporalGRU, self).__init__()
        
        self.weather_embed = nn.Embedding(num_embeddings=5, embedding_dim=16) # CLEAR/RAIN/FOG/SNOW/OTHER
        self.road_embed = nn.Embedding(num_embeddings=4, embedding_dim=16)    # HIGHWAY/URBAN/SCHOOL/OTHER
        
        # Combined size of the input sequence per frame
        combined_input_size = cnn_embed_dim + telemetry_dim + context_dim
        
        # Uni-directional GRU to maintain strictly causal logic suitable for real-time
        # batch_first=True makes inputs [Batch, Seq, Features]
        self.gru = nn.GRU(
            input_size=combined_input_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            batch_first=True
        )
        
        # Risk Classifier Head
        self.hazard_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Outputs 0.0 to 1.0 (Safe to Hazard)
        )
        
        # Time to Collision (TTC) Regression Head
        self.ttc_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs raw continuous seconds
        )

    def forward(self, visual_seq, telemetry_seq, weather_id, road_id):
        """
        visual_seq: [Batch, SeqLen, cnn_embed_dim]
        telemetry_seq: [Batch, SeqLen, telemetry_dim]
        weather_id: [Batch] - Integer IDs for weather
        road_id: [Batch] - Integer IDs for road
        """
        batch_size = visual_seq.size(0)
        seq_len = visual_seq.size(1)
        
        # 1. Embed discrete categorical contexts 
        w_emb = self.weather_embed(weather_id) # [Batch, 16]
        r_emb = self.road_embed(road_id)       # [Batch, 16]
        
        # 2. Expand embeddings temporarily to match sequence length (Context is constant per video clip)
        w_emb_seq = w_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        r_emb_seq = r_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # 3. Concatenate all modalities
        # [Batch, SeqLen, (256 + 3 + 16 + 16)]
        combined_seq = torch.cat((visual_seq, telemetry_seq, w_emb_seq, r_emb_seq), dim=-1)
        
        # 4. Pass through GRU
        gru_out, hidden_state = self.gru(combined_seq) 
        # gru_out is [Batch, SeqLen, hidden_dim]
        
        # 5. Extract the final timestep state (The "Anticipation" state)
        final_state = gru_out[:, -1, :] # [Batch, hidden_dim]
        
        # 6. Predict properties
        hazard_prob = self.hazard_classifier(final_state)
        ttc_estimate = self.ttc_regressor(final_state)
        
        return hazard_prob, ttc_estimate

if __name__ == "__main__":
    print("Testing CPU Temporal Uni-GRU...")
    model = SpatialTemporalGRU()
    
    # Dummy shapes
    b_size, seq_length = 4, 15
    dummy_visual = torch.randn(b_size, seq_length, 256)
    dummy_telemetry = torch.randn(b_size, seq_length, 3)
    dummy_w = torch.randint(0, 4, (b_size,))
    dummy_r = torch.randint(0, 3, (b_size,))
    
    prob, ttc = model(dummy_visual, dummy_telemetry, dummy_w, dummy_r)
    print(f"Hazard Prob shape: {prob.shape}")
    print(f"TTC shape: {ttc.shape}")
    print("Initialization passed.")
