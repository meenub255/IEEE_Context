import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CNN_Encoder(nn.Module):
    """
    CPU-Optimized Spatial Context Encoder for BDD100k
    Extracts deep visual features and classifies Weather & Road conditions.
    """
    def __init__(self, embed_size=256, num_weather=4, num_road=3):
        super(CNN_Encoder, self).__init__()
        
        # Load a pre-trained ResNet18 (small, fast on CPU)
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        
        # Remove the final fully-connected (ImageNet classification) layer
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        
        # Feature Projection Layer
        self.fc_embed = nn.Sequential(
            nn.Linear(num_ftrs, embed_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Context Classification Heads
        self.weather_head = nn.Linear(embed_size, num_weather)
        self.road_head = nn.Linear(embed_size, num_road)

    def forward(self, images):
        """
        Inputs: images [Batch, 3, 224, 224]
        Outputs: 
          features [Batch, embed_size]
          weather_logits [Batch, num_weather]
          road_logits [Batch, num_road]
        """
        # Extract visual representations
        features = self.backbone(images)                 # [Batch, num_ftrs, 1, 1]
        features = features.view(features.size(0), -1)   # Flatten to [Batch, num_ftrs]
        
        # Protect representations into desired dimension
        proj_features = self.fc_embed(features)          # [Batch, embed_size]
        
        # Classify the environmental context
        weather_out = self.weather_head(proj_features)
        road_out = self.road_head(proj_features)
        
        return proj_features, weather_out, road_out

if __name__ == "__main__":
    # CPU specific sanity check
    print("Testing CPU Spatial Encoder...")
    model = CNN_Encoder()
    dummy_input = torch.randn(2, 3, 224, 224) 
    feats, weather, road = model(dummy_input)
    print(f"Features: {feats.shape}")
    print(f"Weather Logits: {weather.shape}")
    print(f"Road Logits: {road.shape}")
    print("Initialization passed.")
