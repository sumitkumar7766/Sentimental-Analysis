import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import AudioLSTM

def train_audio():
    print("=== Training Audio Pitch & Prosody Sentiment Model ===")
    
    # 1. Configuration & Directories
    os.makedirs("../weights", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Dataset Setup (Academic fallback to Synthetic MFCC sequence vectors)
    print("Generating synthetic MFCC speech sequence features (seq_len=50, dim=40)...")
    num_samples = 400
    # Inputs: [batch, sequence_length, features]
    inputs = torch.randn(num_samples, 50, 40)
    # Labels: 0 (negative), 1 (neutral), 2 (positive)
    labels = torch.randint(0, 3, (num_samples,))
    
    dataset = TensorDataset(inputs, labels)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 3. Model, Loss, Optimizer
    # input_dim=40, hidden_dim=128, num_layers=2, feature_dim=256
    model = AudioLSTM(input_dim=40, hidden_dim=128, num_layers=2, feature_dim=256).to(device)
    # Classifier head for independent model training
    eval_classifier = nn.Linear(256, 3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(eval_classifier.parameters()), lr=0.001)
    
    # 4. Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        eval_classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_inputs, batch_labels in train_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            features = model(batch_inputs)
            logits = eval_classifier(features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_inputs.size(0)
            _, predicted = logits.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
            
        epoch_loss = train_loss / total
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        eval_classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                features = model(batch_inputs)
                logits = eval_classifier(features)
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item() * batch_inputs.size(0)
                _, predicted = logits.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
    # 5. Save Model Weights
    weight_path = "../weights/audio_model.pt"
    torch.save(model.state_dict(), weight_path)
    print(f"Successfully saved audio model weights to {weight_path}")

if __name__ == "__main__":
    train_audio()
