import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import TextModel, ImageCNN, AudioLSTM, HybridFusionNetwork

def train_fusion():
    print("=== Training Hybrid Multimodal Fusion Network ===")
    
    # 1. Configuration & Directories
    os.makedirs("../weights", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Branch Weights if available, otherwise initialize default models
    text_feat_extractor = TextModel(pretrained_dim=768, feature_dim=256).to(device)
    if os.path.exists("../weights/text_model.pt"):
        print("Loading pre-trained Text Model weights...")
        text_feat_extractor.load_state_dict(torch.load("../weights/text_model.pt", map_location=device))
    text_feat_extractor.eval()
    
    image_feat_extractor = ImageCNN(feature_dim=256).to(device)
    if os.path.exists("../weights/image_model.pt"):
        print("Loading pre-trained Image Model weights...")
        image_feat_extractor.load_state_dict(torch.load("../weights/image_model.pt", map_location=device))
    image_feat_extractor.eval()
    
    audio_feat_extractor = AudioLSTM(input_dim=40, hidden_dim=128, num_layers=2, feature_dim=256).to(device)
    if os.path.exists("../weights/audio_model.pt"):
        print("Loading pre-trained Audio Model weights...")
        audio_feat_extractor.load_state_dict(torch.load("../weights/audio_model.pt", map_location=device))
    audio_feat_extractor.eval()
    
    # 3. Create Dataset (Academic fallback to Synthetic multimodal dataset)
    print("Generating synthetic multimodal inputs (BERT text, CNN image, LSTM audio)...")
    num_samples = 200
    
    # Inputs corresponding to text (BERT 768), image (pixel 3x128x128), and audio (MFCC 50x40)
    text_in = torch.randn(num_samples, 768)
    image_in = torch.randn(num_samples, 3, 128, 128)
    audio_in = torch.randn(num_samples, 50, 40)
    labels = torch.randint(0, 3, (num_samples,))
    
    dataset = TensorDataset(text_in, image_in, audio_in, labels)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 4. Initialize Fusion Model, Loss, Optimizer
    fusion_model = HybridFusionNetwork(feature_dim=256, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)
    
    # 5. Training Loop
    epochs = 5
    for epoch in range(epochs):
        fusion_model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for t_in, i_in, a_in, batch_labels in train_loader:
            t_in, i_in, a_in, batch_labels = t_in.to(device), i_in.to(device), a_in.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through feature extractors (gradients disabled for backbones)
            with torch.no_grad():
                t_feat = text_feat_extractor(t_in)
                i_feat = image_feat_extractor(i_in)
                a_feat = audio_feat_extractor(a_in)
                
            # Forward pass through Fusion Model
            logits, weights = fusion_model(t_feat, i_feat, a_feat, fusion_type="hybrid")
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * t_in.size(0)
            _, predicted = logits.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
            
        epoch_loss = train_loss / total
        epoch_acc = correct / total
        
        # Validation
        fusion_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for t_in, i_in, a_in, batch_labels in val_loader:
                t_in, i_in, a_in, batch_labels = t_in.to(device), i_in.to(device), a_in.to(device), batch_labels.to(device)
                
                t_feat = text_feat_extractor(t_in)
                i_feat = image_feat_extractor(i_in)
                a_feat = audio_feat_extractor(a_in)
                
                logits, weights = fusion_model(t_feat, i_feat, a_feat, fusion_type="hybrid")
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item() * t_in.size(0)
                _, predicted = logits.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
    # 6. Save Model Weights
    weight_path = "../weights/fusion_model.pt"
    torch.save(fusion_model.state_dict(), weight_path)
    print(f"Successfully saved multimodal fusion model weights to {weight_path}")

if __name__ == "__main__":
    train_fusion()
