#!/usr/bin/env python3
"""
Fix LSTM and GRU Models
Verify existing models and retrain if needed to match the expected architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Device config
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Model Classes (matching app.py) ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

# ---------- Helper Functions ----------
def load_word2vec_model():
    """Load Word2Vec model"""
    try:
        word2vec_model = Word2Vec.load('/Users/tusshar/sentiment-analysis-comparison/word2vec.model')
        print(f"✅ Word2Vec model loaded successfully")
        print(f"   Vector size: {word2vec_model.vector_size}")
        print(f"   Vocabulary size: {len(word2vec_model.wv)}")
        return word2vec_model
    except Exception as e:
        print(f"❌ Error loading Word2Vec model: {e}")
        return None

def preprocess_text(text, word2vec_model, seq_length=100):
    """Convert text to embedding sequence"""
    if word2vec_model is None:
        return torch.zeros(seq_length, word2vec_model.vector_size)
    
    tokens = text.lower().split()
    embeddings = []
    
    for word in tokens[:seq_length]:
        if word in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[word])
    
    # Pad or truncate to seq_length
    input_size = word2vec_model.vector_size
    if not embeddings:
        embeddings = [np.zeros(input_size)]
    
    while len(embeddings) < seq_length:
        embeddings.append(np.zeros(input_size))
    
    embeddings = embeddings[:seq_length]
    
    return torch.tensor(embeddings, dtype=torch.float32)

def prepare_data(reviews_df, word2vec_model, seq_length=100):
    """Prepare data for training"""
    print("Preparing training data...")
    
    # Convert text to embeddings
    X = []
    y = []
    
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    for idx, row in reviews_df.iterrows():
        text = row['review_text']
        sentiment = row['sentiment']
        
        if sentiment in sentiment_map:
            embedding = preprocess_text(text, word2vec_model, seq_length)
            X.append(embedding)
            y.append(sentiment_map[sentiment])
    
    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)
    
    print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} sequence length, {X.shape[2]} features")
    return X, y

def verify_model(model_path, model_class, input_size, hidden_size=64, output_size=3, num_layers=1, dropout_rate=0.4):
    """Verify if a model can be loaded correctly"""
    try:
        model = model_class(input_size, hidden_size, output_size, num_layers, dropout_rate).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Test with dummy input
        dummy_input = torch.randn(1, 100, input_size).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Model loaded successfully: {model_path}")
        print(f"   Output shape: {output.shape}")
        return True, model
        
    except Exception as e:
        print(f"❌ Error loading model {model_path}: {e}")
        return False, None

def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    """Train a model"""
    print(f"Training {model.__class__.__name__}...")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        train_acc = correct_predictions / total_samples
        val_acc = val_correct / val_total
        
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        model.train()
    
    return model

def main():
    print("🔧 Fixing LSTM and GRU Models")
    print("="*50)
    
    # Load Word2Vec model
    word2vec_model = load_word2vec_model()
    if word2vec_model is None:
        print("❌ Cannot proceed without Word2Vec model")
        return
    
    input_size = word2vec_model.vector_size
    hidden_size = 64
    output_size = 3
    num_layers = 1
    dropout_rate = 0.4
    seq_length = 100
    
    print(f"\nModel configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Output size: {output_size}")
    print(f"  Sequence length: {seq_length}")
    
    # Check existing models
    model_paths = {
        'LSTM': '/Users/tusshar/sentiment-analysis-comparison/lstm_model.pth',
        'GRU': '/Users/tusshar/sentiment-analysis-comparison/gru_model.pth',
        'RNN': '/Users/tusshar/sentiment-analysis-comparison/rnn_model.pth'
    }
    
    model_classes = {
        'LSTM': LSTMModel,
        'GRU': GRUModel,
        'RNN': RNNModel
    }
    
    print(f"\n🔍 Verifying existing models...")
    
    models_to_retrain = []
    working_models = {}
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            success, model = verify_model(path, model_classes[name], input_size, hidden_size, output_size, num_layers, dropout_rate)
            if success:
                working_models[name] = model
            else:
                models_to_retrain.append(name)
        else:
            print(f"❌ Model file not found: {path}")
            models_to_retrain.append(name)
    
    if not models_to_retrain:
        print(f"\n✅ All models are working correctly!")
        return
    
    print(f"\n🔧 Models to retrain: {models_to_retrain}")
    
    # Load training data
    print(f"\n📊 Loading training data...")
    try:
        reviews_df = pd.read_csv('/Users/tusshar/sentiment-analysis-comparison/sv_banquet_reviews_for_analysis.csv')
        print(f"   Loaded {len(reviews_df)} reviews")
        print(f"   Sentiment distribution:")
        print(reviews_df['sentiment'].value_counts())
    except Exception as e:
        print(f"❌ Error loading review data: {e}")
        return
    
    # Prepare data
    X, y = prepare_data(reviews_df, word2vec_model, seq_length)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nData split:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    
    # Retrain models
    for model_name in models_to_retrain:
        print(f"\n🚀 Retraining {model_name} model...")
        
        # Create model
        model = model_classes[model_name](input_size, hidden_size, output_size, num_layers, dropout_rate).to(device)
        
        # Train model
        trained_model = train_model(model, X_train, y_train, X_val, y_val, epochs=15)
        
        # Save model
        model_path = model_paths[model_name]
        torch.save(trained_model.state_dict(), model_path)
        print(f"✅ {model_name} model saved to: {model_path}")
        
        # Verify saved model
        success, _ = verify_model(model_path, model_classes[model_name], input_size, hidden_size, output_size, num_layers, dropout_rate)
        if success:
            print(f"✅ {model_name} model verification passed")
        else:
            print(f"❌ {model_name} model verification failed")
    
    print(f"\n🎉 Model fixing completed!")
    print(f"\n💡 You can now restart the Streamlit demo to see all models working.")

if __name__ == "__main__":
    main()
