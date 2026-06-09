import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import TextModel

def train_text():
    print("=== Training Text Sentiment Analysis Model ===")
    
    # 1. Configuration & Directories
    os.makedirs("../weights", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    inputs = None
    labels = None
    
    # 2. Attempt to download the real dair-ai/emotion dataset from Hugging Face
    try:
        print("Attempting to load 'dair-ai/emotion' dataset from Hugging Face...")
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModel
        
        # Load the dataset train split
        raw_dataset = load_dataset("dair-ai/emotion", split="train[:1000]") # Limit to 1000 samples for fast compilation
        print(f"Successfully downloaded {len(raw_dataset)} real samples from Hugging Face.")
        
        # Initialize Tokenizer and Frozen DistilBERT model to extract text embeddings
        print("Initializing DistilBERT tokenizer and embeddings backbone...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        bert_backbone = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
        bert_backbone.eval() # Freeze layers
        
        # Tokenize and extract embeddings
        emb_list = []
        lbl_list = []
        
        # Map dair-ai/emotion labels:
        # 0: sadness -> 0 (Negative)
        # 1: joy -> 2 (Positive)
        # 2: love -> 2 (Positive)
        # 3: anger -> 0 (Negative)
        # 4: fear -> 0 (Negative)
        # 5: surprise -> 1 (Neutral)
        label_mapping = {0: 0, 1: 2, 2: 2, 3: 0, 4: 0, 5: 1}
        
        print("Extracting contextual embeddings from text...")
        with torch.no_grad():
            for item in raw_dataset:
                text = item["text"]
                raw_label = item["label"]
                mapped_label = label_mapping.get(raw_label, 1) # Fallback to Neutral
                
                # Tokenize text
                tokens = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                
                # Forward pass through DistilBERT
                outputs = bert_backbone(**tokens)
                # Use mean-pooled output representation (dim=768)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0).cpu()
                
                emb_list.append(embeddings)
                lbl_list.append(mapped_label)
                
        inputs = torch.stack(emb_list)
        labels = torch.tensor(lbl_list)
        print("Real Hugging Face dataset processing completed successfully!")
        
    except Exception as e:
        print(f"Hugging Face dataset download or tokenizer failed ({e}). Falling back to synthetic BERT-like vectors.")
        # Fallback to synthetic
        num_samples = 500
        inputs = torch.randn(num_samples, 768)
        labels = torch.randint(0, 3, (num_samples,))

    # 3. Create PyTorch DataLoader
    num_samples = len(inputs)
    dataset = TensorDataset(inputs, labels)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 4. Initialize Text Model & Optimizer
    # feature_dim = 256
    model = TextModel(pretrained_dim=768, feature_dim=256).to(device)
    eval_classifier = nn.Linear(256, 3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(eval_classifier.parameters()), lr=0.001)
    
    # 5. Training Loop
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
        
    # 6. Save Model Weights
    weight_path = "../weights/text_model.pt"
    torch.save(model.state_dict(), weight_path)
    print(f"Successfully saved text model weights to {weight_path}")

if __name__ == "__main__":
    train_text()
