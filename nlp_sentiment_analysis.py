# -*- coding: utf-8 -*-
"""NLP Sentiment Analysis for Local Execution"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from datasets import Dataset as HFDataset
import time
import contractions
import emoji
import html
from textblob import TextBlob
import ssl

# Bypass SSL verification for NLTK downloads
try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

# Download NLTK data with error handling
def download_nltk_data():
    datasets = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'vader_lexicon', 'averaged_perceptron_tagger']
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=True)
            print(f"Successfully downloaded NLTK dataset: {dataset}")
        except Exception as e:
            print(f"Error downloading NLTK dataset {dataset}: {e}")
            raise

# Run NLTK downloads at the start
download_nltk_data()

# Set device (prefer MPS for Apple Silicon, fallback to CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class AdvancedTextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_stopwords = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere',
            'hardly', 'scarcely', 'barely', 'doesn', 'aren', 'couldn', 'didn',
            'doesn', 'hadn', 'hasn', 'haven', 'isn', 'mightn', 'mustn', 'needn',
            'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
        }
        self.stop_words = self.stop_words - self.sentiment_stopwords
        self.slang_dict = {
            'u': 'you', 'ur': 'your', 'r': 'are', 'n': 'and',
            'luv': 'love', 'lol': 'laugh out loud', 'omg': 'oh my god',
            'wtf': 'what the hell', 'tbh': 'to be honest', 'imo': 'in my opinion',
            'imho': 'in my humble opinion', 'fyi': 'for your information',
            'btw': 'by the way', 'aka': 'also known as', 'asap': 'as soon as possible',
            'gr8': 'great', 'thx': 'thanks', 'pls': 'please', 'plz': 'please',
            'gud': 'good', 'kool': 'cool', 'awesome': 'excellent'
        }

    def expand_contractions(self, text):
        try:
            return contractions.fix(text)
        except:
            contractions_dict = {
                "won't": "will not", "can't": "cannot", "n't": " not",
                "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
                "'m": " am", "it's": "it is", "that's": "that is",
                "what's": "what is", "where's": "where is", "how's": "how is",
                "who's": "who is", "there's": "there is", "here's": "here is"
            }
            for contraction, expansion in contractions_dict.items():
                text = text.replace(contraction, expansion)
            return text

    def delete_emojis(self, text):
        text = emoji.demojize(text)
        text = re.sub(r'_', ' ', text)
        return text

    def Repetions(self, text):
        text = re.sub(r'(!)\1{2,}', '!', text)
        text = re.sub(r'(\?)\1{2,}', '?', text)
        text = re.sub(r'(\.)\1{2,}', '.', text)
        text = re.sub(r'(.)\1+', r'\1\1', text)
        return text

    def slang(self, text):
        words = text.split()
        expand_words = []
        for word in words:
            word_lower = word.lower()
            expand_words.append(self.slang_dict.get(word_lower, word))
        return ' '.join(expand_words)

    def clean_text_advanced(self, text):
        text = html.unescape(text)
        text = text.lower()
        text = self.delete_emojis(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = self.slang(text)
        text = self.Repetions(text)
        text = re.sub(r'[^\w\s!?.,]', '', text)
        text = text.lower()
        text = text.strip()
        return text

    def advanced_tokensize(self, text):
        if not text:
            return []
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words or token in self.sentiment_stopwords]
        tokens = [token for token in tokens if len(token) > 1 or token in ['!', '?']]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if token.strip()]
        return tokens

def preprocess_text(text, preprocessor):
    if not isinstance(text, str) or pd.isna(text):
        return "", ""
    cleaned_text = preprocessor.clean_text_advanced(text)
    tokens = preprocessor.advanced_tokensize(cleaned_text)
    return tokens, cleaned_text

def balance_dataset(df, label_column='label', method='oversample', max_samples=None):
    print("Original class distribution:")
    print(df[label_column].value_counts())
    
    if method == 'oversample':
        class_0 = df[df[label_column] == 0]
        class_1 = df[df[label_column] == 1]
        class_2 = df[df[label_column] == 2]
        max_size = min(max(len(class_0), len(class_1), len(class_2)), max_samples or float('inf'))
        class_0_upsampled = resample(class_0, replace=True, n_samples=max_size, random_state=42)
        class_1_upsampled = resample(class_1, replace=True, n_samples=max_size, random_state=42)
        class_2_upsampled = resample(class_2, replace=True, n_samples=max_size, random_state=42)
        df_balanced = pd.concat([class_0_upsampled, class_1_upsampled, class_2_upsampled])
    else:
        df_balanced = df
    
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print("\nClass distribution after balancing:")
    print(df_balanced[label_column].value_counts())
    return df_balanced

def load_and_preprocess_data(train_path, test_path, sample_size=5000):
    preprocessor = AdvancedTextPreprocessor()
    try:
        train_df = pd.read_csv(train_path, encoding='latin1')
        test_df = pd.read_csv(test_path, encoding='latin1')
    except UnicodeDecodeError:
        train_df = pd.read_csv(train_path, encoding='latin1', errors='replace')
        test_df = pd.read_csv(test_path, encoding='latin1', errors='replace')

    if 'text' not in train_df.columns or 'sentiment' not in train_df.columns or 'text' not in test_df.columns or 'sentiment' not in test_df.columns:
        raise ValueError("Datasets must contain 'text' and 'sentiment' columns")

    print(f"Original train dataset size: {len(train_df)}")
    print(f"Original test dataset size: {len(test_df)}")
    
    train_df = train_df.dropna(subset=['text', 'sentiment'])
    test_df = test_df.dropna(subset=['text', 'sentiment'])
    print(f"Train dataset size after dropping NaN: {len(train_df)}")
    print(f"Test dataset size after dropping NaN: {len(test_df)}")
    
    if sample_size and len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)
        print(f"Sampled train dataset to {sample_size} rows")
    
    train_df['sentiment'] = train_df['sentiment'].astype(str).str.strip().str.lower()
    test_df['sentiment'] = test_df['sentiment'].astype(str).str.strip().str.lower()

    print("Unique sentiment values in train.csv:", train_df['sentiment'].unique())
    print("Unique sentiment values in test.csv:", test_df['sentiment'].unique())

    sentiment_map = {
        'negative': 0, 'neutral': 1, 'positive': 2,
        '0': 0, '2': 1, '4': 2,
        0: 0, 2: 1, 4: 2
    }
    train_df['label'] = train_df['sentiment'].map(sentiment_map)
    test_df['label'] = test_df['sentiment'].map(sentiment_map)

    invalid_train = train_df[train_df['label'].isna()]
    invalid_test = test_df[test_df['label'].isna()]
    if not invalid_train.empty:
        print(f"Warning: Found {len(invalid_train)} invalid sentiment values in train.csv:")
        print(invalid_train[['text', 'sentiment']])
    if not invalid_test.empty:
        print(f"Warning: Found {len(invalid_test)} invalid sentiment values in test.csv:")
        print(invalid_test[['text', 'sentiment']])

    train_df = train_df.dropna(subset=['label'])
    test_df = test_df.dropna(subset=['label'])

    train_df = train_df[train_df['label'].isin([0, 1, 2])]
    test_df = test_df[test_df['label'].isin([0, 1, 2])]

    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)

    print("Preprocessing text...")
    start_time = time.time()
    train_df['processed_tokens'], train_df['cleaned_text'] = zip(*train_df['text'].apply(lambda x: preprocess_text(x, preprocessor)))
    test_df['processed_tokens'], test_df['cleaned_text'] = zip(*test_df['text'].apply(lambda x: preprocess_text(x, preprocessor)))
    print(f"Text preprocessing completed in {time.time() - start_time:.2f} seconds")

    train_df = train_df[train_df['cleaned_text'].str.len() > 0]
    test_df = test_df[test_df['cleaned_text'].str.len() > 0]

    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)
    val_df['processed_tokens'], val_df['cleaned_text'] = zip(*val_df['text'].apply(lambda x: preprocess_text(x, preprocessor)))

    train_hf = HFDataset.from_pandas(train_df[['cleaned_text', 'label']].rename(columns={'cleaned_text': 'text'}))
    val_hf = HFDataset.from_pandas(val_df[['cleaned_text', 'label']].rename(columns={'cleaned_text': 'text'}))
    test_hf = HFDataset.from_pandas(test_df[['cleaned_text', 'label']].rename(columns={'cleaned_text': 'text'}))

    return train_df, val_df, test_df, train_hf, val_hf, test_hf

def prepare_bert_inputs(hf_dataset):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', clean_up_tokenization_spaces=True)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    return hf_dataset.map(tokenize_function, batched=True)

def prepare_rnn_inputs(reviews, word2vec_model, seq_length=100):
    def text_to_embedding(tokens, word2vec_model, seq_length):
        if not tokens:
            return [np.zeros(word2vec_model.vector_size) for _ in range(seq_length)]
        embeddings = []
        for word in tokens[:seq_length]:
            if word in word2vec_model.wv:
                embeddings.append(word2vec_model.wv[word])
        if len(embeddings) > seq_length:
            embeddings = embeddings[:seq_length]
        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
            while len(embeddings) < seq_length:
                embeddings.append(mean_embedding)
        else:
            embeddings = [np.zeros(word2vec_model.vector_size) for _ in range(seq_length)]
        return np.array(embeddings)
    
    X = [text_to_embedding(review, word2vec_model, seq_length) for review in reviews]
    X = np.array(X, dtype=np.float32)
    return torch.tensor(X, dtype=torch.float32)

def train_word2vec(train_tokens):
    print("Training Word2Vec...")
    start_time = time.time()
    model = Word2Vec(
        sentences=train_tokens,
        vector_size=100,
        window=10,
        min_count=1,
        workers=2,
        epochs=20,
        sg=1,
        negative=10,
        alpha=0.025,
        min_alpha=0.0001
    )
    print(f"Word2Vec training completed in {time.time() - start_time:.2f} seconds")
    return model

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0, bidirectional=False)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        pooled_out = lstm_out[:, -1, :]
        pooled_out = self.batch_norm1(pooled_out)
        out = self.dropout1(pooled_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0, bidirectional=False)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size // 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        gru_out, hn = self.gru(x, h0)
        pooled_out = gru_out[:, -1, :]
        pooled_out = self.batch_norm1(pooled_out)
        out = self.dropout1(pooled_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.layer_norm(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out

def train_bert_model(train_hf, val_hf, num_epochs=2):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', clean_up_tokenization_spaces=True)
    train_dataset = prepare_bert_inputs(train_hf)
    val_dataset = prepare_bert_inputs(val_hf)
    train_dataset = train_dataset.remove_columns(['text'])
    val_dataset = val_dataset.remove_columns(['text'])
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    
    training_args = TrainingArguments(
        output_dir='./results',  # Required output directory
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        fp16=False,
        warmup_steps=50,
        evaluation_strategy='epoch',  # Compatible with transformers <4.45
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
        dataloader_num_workers=0,  # Avoid multiprocessing issues on macOS
        dataloader_pin_memory=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"DistilBERT training completed in {training_time:.2f} seconds")
    return trainer, training_time

def evaluate_bert_model(trainer, test_hf):
    test_dataset = prepare_bert_inputs(test_hf)
    test_dataset = test_dataset.remove_columns(['text'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = np.argmax(logits, axis=1)
    y_pred_proba = torch.softmax(torch.tensor(logits), dim=1).numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba, multi_class='ovr'),
        'training_time': 0
    }

def train_pytorch_model(model, train_loader, val_loader, num_epochs, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = 0
    max_patience = 7
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)

        print(f'{model_name} Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience >= max_patience:
            print(f'{model_name} Early stopped at epoch {epoch+1}')
            break

    training_time = time.time() - start_time
    print(f"{model_name} training completed in {training_time:.2f} seconds")
    return history, best_val_loss, training_time

def evaluate_pytorch_model(model, test_loader, device, model_name):
    model.eval()
    y_true, y_pred, y_pred_proba = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba, multi_class='ovr'),
        'training_time': 0
    }

def plot_metrics(models_metrics, metric_name, title):
    plt.figure(figsize=(10, 6))
    model_names = list(models_metrics.keys())
    values = [models_metrics[model][metric_name] for model in model_names]
    bars = plt.bar(model_names, values, alpha=0.8, color=['red', 'blue', 'green', 'orange'])
    plt.title(title)
    plt.ylabel(metric_name.capitalize())
    plt.xlabel('Models')
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'], cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    plt.close()

def plot_training_history(history, model_name):
    if not isinstance(history, dict) or 'train_loss' not in history:
        print(f"Invalid history for {model_name}")
        return
    required_keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    missing_keys = [key for key in required_keys if key not in history]
    if missing_keys:
        print(f"Missing history keys for {model_name}: {missing_keys}")
        return
    lengths = [len(history[key]) for key in required_keys]
    min_length = min(lengths)
    if min_length == 0:
        print(f"No training history data for {model_name}")
        return
    
    train_loss = history['train_loss'][:min_length]
    val_loss = history['val_loss'][:min_length]
    train_acc = history['train_acc'][:min_length]
    val_acc = history['val_acc'][:min_length]
    
    print(f"Plotting {model_name} history with {min_length} epochs")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    epochs = range(1, min_length + 1)
    ax1.plot(epochs, train_loss, 'bo-', label='Training Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_loss, 'ro-', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_title(f'{model_name} - Training/Validation Loss', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, train_acc, 'bo-', label='Training Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, val_acc, 'ro-', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_title(f'{model_name} - Training/Validation Accuracy', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

def simple_deployment_analysis(models_metrics):
    data = []
    for model_name, metrics in models_metrics.items():
        data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1_Score': metrics['f1'],
            'ROC_AUC': metrics['roc_auc'],
            'Training_Time': metrics['training_time']
        })
    
    df = pd.DataFrame(data)
    df['Efficiency_Score'] = df['F1_Score'] / (df['Training_Time'] / 100)
    df['Deployment_Score'] = (
        df['Accuracy'] * 0.3 +
        df['F1_Score'] * 0.3 +
        df['ROC_AUC'] * 0.2 +
        (df['Efficiency_Score'] / df['Efficiency_Score'].max()) * 0.2
    )
    df = df.sort_values('Deployment_Score', ascending=False)
    
    print("DEPLOYMENT RECOMMENDATION ANALYSIS")
    print("=" * 90)
    print(df.round(3))
    
    best_model = df.iloc[0]['Model']
    best_score = df.iloc[0]['Deployment_Score']
    
    print(f"\nRECOMMENDED MODEL: {best_model}")
    print(f"Deployment Score: {best_score:.3f}")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    colors = ['gold' if model == best_model else 'lightblue' for model in df['Model']]
    bars = plt.bar(df['Model'], df['Deployment_Score'], color=colors, edgecolor='black')
    plt.title('Overall Deployment Score', fontweight='bold')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    for i, (bar, score) in enumerate(zip(bars, df['Deployment_Score'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.subplot(2, 2, 2)
    plt.scatter(df['Efficiency_Score'], df['F1_Score'],
                c=['red' if model == best_model else 'blue' for model in df['Model']],
                s=200, alpha=0.7, edgecolors='black')
    for i, model in enumerate(df['Model']):
        plt.annotate(model, (df['Efficiency_Score'].iloc[i], df['F1_Score'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontweight='bold')
    plt.xlabel('Efficiency Score')
    plt.ylabel('F1 Score')
    plt.title('Performance vs Efficiency', fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    colors = ['gold' if model == best_model else 'lightcoral' for model in df['Model']]
    bars = plt.bar(df['Model'], df['Training_Time'], color=colors, edgecolor='black')
    plt.title('Training Time (Lower is Better)', fontweight='bold')
    plt.ylabel('Seconds')
    plt.xticks(rotation=45)
    for bar, time in zip(bars, df['Training_Time']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')

    plt.subplot(2, 2, 4)
    colors = ['gold' if model == best_model else 'lightgreen' for model in df['Model']]
    bars = plt.bar(df['Model'], df['Accuracy'], color=colors, edgecolor='black')
    plt.title('Accuracy Comparison', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    for bar, acc in zip(bars, df['Accuracy']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"\nWHY {best_model} IS RECOMMENDED:")
    print("-" * 30)
    best_row = df[df['Model'] == best_model].iloc[0]
    print(f"Accuracy: {best_row['Accuracy']:.1%}")
    print(f"F1-Score: {best_row['F1_Score']:.3f}")
    print(f"Training Time: {best_row['Training_Time']:.1f} seconds")
    print(f"Efficiency Score: {best_row['Efficiency_Score']:.3f}")
    
    return best_model

def main():
    train_file = "/Users/tusshar/sentiment-analysis-comparison/train.csv"
    test_file = "/Users/tusshar/sentiment-analysis-comparison/test.csv"

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Ensure 'train.csv' and 'test.csv' exist at {train_file} and {test_file}")

    print(f"Using train file: {train_file}")
    print(f"Using test file: {test_file}")

    train_df, val_df, test_df, train_hf, val_hf, test_hf = load_and_preprocess_data(train_file, test_file, sample_size=5000)

    print("Word2Vec Training")
    word2vec_model = train_word2vec(train_df['processed_tokens'].values.tolist())

    train_padded = prepare_rnn_inputs(train_df['processed_tokens'], word2vec_model)
    val_padded = prepare_rnn_inputs(val_df['processed_tokens'], word2vec_model)
    test_padded = prepare_rnn_inputs(test_df['processed_tokens'], word2vec_model)

    train_dataset = TensorDataset(train_padded, torch.tensor(train_df['label'].values, dtype=torch.long))
    val_dataset = TensorDataset(val_padded, torch.tensor(val_df['label'].values, dtype=torch.long))
    test_dataset = TensorDataset(test_padded, torch.tensor(test_df['label'].values, dtype=torch.long))

    train_loader_rnn = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader_rnn = DataLoader(val_dataset, batch_size=16)
    test_loader_rnn = DataLoader(test_dataset, batch_size=16)

    input_size = word2vec_model.vector_size
    hidden_size = 64
    output_size = 3
    num_layers = 1
    dropout_rate = 0.4
    num_epochs_bert = 2
    num_epochs_rnn = 10

    models_metrics = {}

    print("DistilBERT Training")
    trainer_bert, bert_time = train_bert_model(train_hf, val_hf, num_epochs_bert)
    models_metrics['DistilBERT'] = evaluate_bert_model(trainer_bert, test_hf)
    models_metrics['DistilBERT']['training_time'] = bert_time
    plot_confusion_matrix(models_metrics['DistilBERT']['confusion_matrix'], 'DistilBERT')

    print("Training RNN...")
    rnn_model = RNNModel(input_size, hidden_size, output_size, num_layers, dropout_rate).to(device)
    history_rnn, best_val_loss_rnn, rnn_time = train_pytorch_model(rnn_model, train_loader_rnn, val_loader_rnn, num_epochs_rnn, device, 'RNN')
    models_metrics['RNN'] = evaluate_pytorch_model(rnn_model, test_loader_rnn, device, 'RNN')
    models_metrics['RNN']['training_time'] = rnn_time
    plot_training_history(history_rnn, 'RNN')
    plot_confusion_matrix(models_metrics['RNN']['confusion_matrix'], 'RNN')

    print("Training LSTM...")
    lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout_rate).to(device)
    history_lstm, best_val_loss_lstm, lstm_time = train_pytorch_model(lstm_model, train_loader_rnn, val_loader_rnn, num_epochs_rnn, device, 'LSTM')
    models_metrics['LSTM'] = evaluate_pytorch_model(lstm_model, test_loader_rnn, device, 'LSTM')
    models_metrics['LSTM']['training_time'] = lstm_time
    plot_training_history(history_lstm, 'LSTM')
    plot_confusion_matrix(models_metrics['LSTM']['confusion_matrix'], 'LSTM')

    print("Training GRU...")
    gru_model = GRUModel(input_size, hidden_size, output_size, num_layers, dropout_rate).to(device)
    history_gru, best_val_loss_gru, gru_time = train_pytorch_model(gru_model, train_loader_rnn, val_loader_rnn, num_epochs_rnn, device, 'GRU')
    models_metrics['GRU'] = evaluate_pytorch_model(gru_model, test_loader_rnn, device, 'GRU')
    models_metrics['GRU']['training_time'] = gru_time
    plot_training_history(history_gru, 'GRU')
    plot_confusion_matrix(models_metrics['GRU']['confusion_matrix'], 'GRU')

    plot_metrics(models_metrics, 'accuracy', 'Accuracy Comparison')
    plot_metrics(models_metrics, 'f1', 'F1-Score Comparison (Macro)')
    plot_metrics(models_metrics, 'roc_auc', 'ROC-AUC Comparison (OVR)')
    plot_metrics(models_metrics, 'training_time', 'Training Time Comparison (seconds)')

    recommended_model = simple_deployment_analysis(models_metrics)
    print(f"Recommended model for deployment: {recommended_model}")

if __name__ == "__main__":
    main()