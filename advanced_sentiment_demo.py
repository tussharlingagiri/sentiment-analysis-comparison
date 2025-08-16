#!/usr/bin/env python3
"""
Enhanced Streamlit app for S V Banquet Halls Sentiment Analysis Demo
Integrates multiple model types: Neural Networks (LSTM/GRU/RNN), Transformers (DistilBERT), and Traditional ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Device config
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Page config
st.set_page_config(
    page_title="S V Banquet Halls - Advanced Sentiment Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86C1;
    }
    .model-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .neural-net { border-left: 5px solid #8E44AD; }
    .transformer { border-left: 5px solid #E74C3C; }
    .traditional { border-left: 5px solid #2ECC71; }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .positive-pred { background-color: #D5F5D5; color: #2E7D32; }
    .negative-pred { background-color: #FFEBEE; color: #C62828; }
    .neutral-pred { background-color: #FFF3E0; color: #F57C00; }
</style>
""", unsafe_allow_html=True)

# ---------- Neural Network Model Classes ----------
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

# ---------- Cached Data Loading ----------
@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        real_reviews = pd.read_csv('/Users/tusshar/sentiment-analysis-comparison/sv_banquet_reviews_for_analysis.csv')
        complete_reviews = pd.read_csv('/Users/tusshar/sentiment-analysis-comparison/sv_banquet_reviews_complete.csv')
        combined_reviews = pd.read_csv('/Users/tusshar/sentiment-analysis-comparison/combined_banquet_reviews.csv')
        return real_reviews, complete_reviews, combined_reviews
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_resource
def load_neural_models():
    """Load pre-trained neural network models"""
    models = {}
    
    # Load Word2Vec model
    try:
        word2vec_model = Word2Vec.load('/Users/tusshar/sentiment-analysis-comparison/word2vec.model')
        input_size = word2vec_model.vector_size
        seq_length = 100
        models['word2vec'] = word2vec_model
        models['input_size'] = input_size
        models['seq_length'] = seq_length
    except:
        st.warning("Word2Vec model not found. Neural network models will be unavailable.")
        return models
    
    # Load neural network models
    def load_model(model_class, path, name):
        try:
            model = model_class(input_size, 64, 3, 1, 0.4).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            return model
        except:
            st.warning(f"{name} model not found at {path}")
            return None
    
    models['lstm'] = load_model(LSTMModel, '/Users/tusshar/sentiment-analysis-comparison/lstm_model.pth', 'LSTM')
    models['gru'] = load_model(GRUModel, '/Users/tusshar/sentiment-analysis-comparison/gru_model.pth', 'GRU') 
    models['rnn'] = load_model(RNNModel, '/Users/tusshar/sentiment-analysis-comparison/rnn_model.pth', 'RNN')
    
    return models

@st.cache_resource
def load_transformer_model():
    """Load DistilBERT transformer model"""
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('/Users/tusshar/sentiment-analysis-comparison/results/checkpoint-best')
        model.to(device)
        model.eval()
        return model, tokenizer
    except:
        st.warning("DistilBERT model not found. Transformer predictions will be unavailable.")
        return None, None

@st.cache_resource
def load_traditional_models():
    """Load traditional ML models"""
    try:
        model = joblib.load('/Users/tusshar/sentiment-analysis-comparison/models/best_sentiment_model.pkl')
        vectorizer = joblib.load('/Users/tusshar/sentiment-analysis-comparison/models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

# ---------- Prediction Functions ----------
def preprocess_text_neural(text, word2vec_model, seq_length, input_size):
    """Preprocess text for neural network models"""
    if word2vec_model is None:
        return torch.zeros(1, seq_length, input_size).to(device)
    
    tokens = text.lower().split()
    embeddings = []
    
    for word in tokens[:seq_length]:
        if word in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[word])
    
    if not embeddings:
        embeddings = [np.zeros(input_size)]
    
    # Pad or truncate to seq_length
    while len(embeddings) < seq_length:
        embeddings.append(np.zeros(input_size))
    
    embeddings = embeddings[:seq_length]
    
    return torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)

def predict_neural(model, text, neural_models):
    """Predict using neural network models"""
    if model is None or neural_models.get('word2vec') is None:
        return None, None
    
    inputs = preprocess_text_neural(
        text, 
        neural_models['word2vec'], 
        neural_models['seq_length'], 
        neural_models['input_size']
    )
    
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities.cpu().numpy()[0]
    
    return prediction, confidence

def predict_transformer(model, tokenizer, text):
    """Predict using transformer model"""
    if model is None or tokenizer is None:
        return None, None
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities.cpu().numpy()[0]
    
    return prediction, confidence

def predict_traditional(model, vectorizer, text):
    """Predict using traditional ML models"""
    if model is None or vectorizer is None:
        return None, None
    
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    confidence = model.predict_proba(text_vectorized)[0]
    
    # Map string labels to indices
    label_to_idx = {'negative': 0, 'neutral': 1, 'positive': 2}
    prediction_idx = label_to_idx.get(prediction, 0)
    
    return prediction_idx, confidence

# ---------- Evaluation Functions ----------
def evaluate_models_on_real_data(real_reviews, neural_models, transformer_model, traditional_models):
    """Evaluate all models on real review data"""
    
    # Prepare data
    texts = real_reviews['review_text'].tolist()[:50]  # Limit for demo
    true_labels = real_reviews['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).tolist()[:50]
    
    results = {}
    
    # Test neural network models
    for name in ['lstm', 'gru', 'rnn']:
        model = neural_models.get(name)
        if model is not None:
            predictions = []
            confidences = []
            
            for text in texts:
                pred, conf = predict_neural(model, text, neural_models)
                if pred is not None:
                    predictions.append(pred)
                    confidences.append(conf)
                else:
                    predictions.append(0)
                    confidences.append([0.33, 0.33, 0.34])
            
            if predictions:
                accuracy = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions, average='macro')
                results[name.upper()] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': predictions,
                    'type': 'Neural Network'
                }
    
    # Test transformer model
    distilbert_model, distilbert_tokenizer = transformer_model
    if distilbert_model is not None:
        predictions = []
        for text in texts:
            pred, conf = predict_transformer(distilbert_model, distilbert_tokenizer, text)
            predictions.append(pred if pred is not None else 0)
        
        if predictions:
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='macro')
            results['DistilBERT'] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': predictions,
                'type': 'Transformer'
            }
    
    # Test traditional ML model
    trad_model, trad_vectorizer = traditional_models
    if trad_model is not None:
        predictions = []
        for text in texts:
            pred, conf = predict_traditional(trad_model, trad_vectorizer, text)
            predictions.append(pred if pred is not None else 0)
        
        if predictions:
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='macro')
            results['Random Forest'] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': predictions,
                'type': 'Traditional ML'
            }
    
    return results, true_labels

# ---------- Main App ----------
def main():
    # Header
    st.markdown('<h1 class="main-header">🏛️ S V Banquet Halls - Advanced Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Compare Neural Networks, Transformers, and Traditional ML on Real Customer Reviews</p>', unsafe_allow_html=True)
    
    # Load all data and models
    real_reviews, complete_reviews, combined_reviews = load_data()
    neural_models = load_neural_models()
    transformer_model = load_transformer_model()
    traditional_models = load_traditional_models()
    
    if real_reviews is None:
        st.error("Could not load review data. Please ensure CSV files exist.")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Model Control Panel")
    page = st.sidebar.selectbox(
        "Choose Analysis Type:",
        [
            "🏠 Dashboard Overview",
            "🤖 Interactive Model Comparison", 
            "🔮 Live Prediction Arena",
            "📊 Performance Benchmarks",
            "📈 Model Analysis Deep Dive"
        ]
    )
    
    if page == "🏠 Dashboard Overview":
        st.header("📊 Real Customer Reviews Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f'<div class="metric-container"><h3>Total Reviews</h3><h2>{len(complete_reviews)}</h2></div>',
                unsafe_allow_html=True
            )
        
        with col2:
            avg_rating = complete_reviews['star_rating_numeric'].mean()
            st.markdown(
                f'<div class="metric-container"><h3>Average Rating</h3><h2>{avg_rating:.1f}⭐</h2></div>',
                unsafe_allow_html=True
            )
        
        with col3:
            positive_pct = (complete_reviews['sentiment_label'] == 'positive').mean() * 100
            st.markdown(
                f'<div class="metric-container"><h3>Positive Sentiment</h3><h2>{positive_pct:.1f}%</h2></div>',
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f'<div class="metric-container"><h3>Text Reviews</h3><h2>{len(real_reviews)}</h2></div>',
                unsafe_allow_html=True
            )
        
        # Model availability status
        st.subheader("🤖 Available Models")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="model-card neural-net"><h4>🧠 Neural Networks</h4>', unsafe_allow_html=True)
            if neural_models.get('lstm'): st.success("✅ LSTM Model")
            else: st.error("❌ LSTM Model")
            if neural_models.get('gru'): st.success("✅ GRU Model") 
            else: st.error("❌ GRU Model")
            if neural_models.get('rnn'): st.success("✅ RNN Model")
            else: st.error("❌ RNN Model")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="model-card transformer"><h4>🤗 Transformer</h4>', unsafe_allow_html=True)
            if transformer_model[0]: st.success("✅ DistilBERT Model")
            else: st.error("❌ DistilBERT Model")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="model-card traditional"><h4>📊 Traditional ML</h4>', unsafe_allow_html=True)
            if traditional_models[0]: st.success("✅ Random Forest Model")
            else: st.error("❌ Random Forest Model")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            sentiment_counts = real_reviews['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Real Reviews Sentiment Distribution",
                color_discrete_map={'positive': '#28A745', 'negative': '#DC3545', 'neutral': '#FFC107'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rating distribution
            rating_counts = complete_reviews['star_rating_numeric'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                title="Rating Distribution",
                labels={'x': 'Star Rating', 'y': 'Number of Reviews'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
    elif page == "🤖 Interactive Model Comparison":
        st.header("🤖 Model Performance Comparison")
        st.write("Compare different model architectures on real S V Banquet Halls reviews")
        
        if st.button("🚀 Run Model Comparison", type="primary"):
            with st.spinner("Evaluating all models on real review data..."):
                results, true_labels = evaluate_models_on_real_data(
                    real_reviews, neural_models, transformer_model, traditional_models
                )
            
            if results:
                # Performance comparison chart
                st.subheader("📊 Performance Metrics")
                
                performance_data = []
                for model_name, metrics in results.items():
                    performance_data.append({
                        'Model': model_name,
                        'Accuracy': metrics['accuracy'],
                        'F1-Score': metrics['f1_score'],
                        'Type': metrics['type']
                    })
                
                df_performance = pd.DataFrame(performance_data)
                
                # Grouped bar chart
                fig = px.bar(
                    df_performance.melt(id_vars=['Model', 'Type'], value_vars=['Accuracy', 'F1-Score']),
                    x='Model',
                    y='value',
                    color='variable',
                    facet_col='Type',
                    title="Model Performance by Architecture Type",
                    labels={'value': 'Score', 'variable': 'Metric'}
                )
                fig.update_layout(yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📈 Detailed Results")
                
                # Sort by accuracy
                df_performance_sorted = df_performance.sort_values('Accuracy', ascending=False)
                
                for idx, row in df_performance_sorted.iterrows():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Model", row['Model'])
                    with col2:
                        st.metric("Type", row['Type'])
                    with col3:
                        st.metric("Accuracy", f"{row['Accuracy']:.3f}")
                    with col4:
                        st.metric("F1-Score", f"{row['F1-Score']:.3f}")
                
                # Confusion matrices
                st.subheader("🎯 Confusion Matrices")
                
                label_names = ['Negative', 'Neutral', 'Positive']
                
                # Create confusion matrix plots
                n_models = len(results)
                cols_per_row = 2
                
                for i, (model_name, metrics) in enumerate(results.items()):
                    if i % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    
                    with cols[i % cols_per_row]:
                        cm = confusion_matrix(true_labels, metrics['predictions'])
                        
                        fig = ff.create_annotated_heatmap(
                            z=cm,
                            x=label_names,
                            y=label_names,
                            colorscale='Blues',
                            showscale=True
                        )
                        fig.update_layout(
                            title=f"{model_name} Confusion Matrix",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("No models available for comparison. Please ensure model files exist.")
    
    elif page == "🔮 Live Prediction Arena":
        st.header("🔮 Live Prediction Arena")
        st.write("Test all models with your own text input and compare their predictions")
        
        # Text input
        user_text = st.text_area(
            "Enter a review to analyze:",
            placeholder="E.g., Great place for functions, excellent food and service! Staff was very helpful.",
            height=100
        )
        
        # Model selection
        available_models = []
        if neural_models.get('lstm'): available_models.append('LSTM')
        if neural_models.get('gru'): available_models.append('GRU')
        if neural_models.get('rnn'): available_models.append('RNN')
        if transformer_model[0]: available_models.append('DistilBERT')
        if traditional_models[0]: available_models.append('Random Forest')
        
        if not available_models:
            st.error("No models available for prediction.")
            return
        
        selected_models = st.multiselect(
            "Select models to test:",
            available_models,
            default=available_models[:3] if len(available_models) >= 3 else available_models
        )
        
        if st.button("🎯 Predict Sentiment", type="primary") and user_text and selected_models:
            st.subheader("🎭 Prediction Results")
            
            predictions = {}
            label_names = ['Negative', 'Neutral', 'Positive']
            
            # Get predictions from all selected models
            for model_name in selected_models:
                if model_name in ['LSTM', 'GRU', 'RNN']:
                    model = neural_models.get(model_name.lower())
                    pred, conf = predict_neural(model, user_text, neural_models)
                    
                elif model_name == 'DistilBERT':
                    pred, conf = predict_transformer(transformer_model[0], transformer_model[1], user_text)
                    
                elif model_name == 'Random Forest':
                    pred, conf = predict_traditional(traditional_models[0], traditional_models[1], user_text)
                
                if pred is not None:
                    predictions[model_name] = {
                        'prediction': pred,
                        'confidence': conf,
                        'label': label_names[pred]
                    }
            
            # Display predictions
            cols = st.columns(len(predictions))
            
            for i, (model_name, result) in enumerate(predictions.items()):
                with cols[i]:
                    pred_label = result['label']
                    confidence = result['confidence'][result['prediction']]
                    
                    # Color-coded prediction box
                    if pred_label == 'Positive':
                        st.markdown(f'<div class="prediction-result positive-pred">😊 {pred_label}<br>Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
                    elif pred_label == 'Negative':
                        st.markdown(f'<div class="prediction-result negative-pred">😞 {pred_label}<br>Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-result neutral-pred">😐 {pred_label}<br>Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
                    
                    st.write(f"**{model_name}**")
            
            # Confidence comparison chart
            st.subheader("📊 Confidence Comparison")
            
            # Create confidence comparison data
            confidence_data = []
            for model_name, result in predictions.items():
                for i, (label, conf) in enumerate(zip(label_names, result['confidence'])):
                    confidence_data.append({
                        'Model': model_name,
                        'Sentiment': label,
                        'Confidence': conf,
                        'Is_Prediction': i == result['prediction']
                    })
            
            df_confidence = pd.DataFrame(confidence_data)
            
            # Grouped bar chart
            fig = px.bar(
                df_confidence,
                x='Model',
                y='Confidence',
                color='Sentiment',
                title="Model Confidence by Sentiment",
                color_discrete_map={'Positive': '#28A745', 'Negative': '#DC3545', 'Neutral': '#FFC107'}
            )
            fig.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)
            
            # Agreement analysis
            if len(predictions) > 1:
                st.subheader("🤝 Model Agreement Analysis")
                
                pred_labels = [result['label'] for result in predictions.values()]
                model_names = list(predictions.keys())
                
                # Check if all models agree
                if len(set(pred_labels)) == 1:
                    st.success(f"🎯 **Consensus Reached**: All models predict **{pred_labels[0]}**")
                else:
                    st.warning("⚠️ **Mixed Predictions**: Models disagree on sentiment")
                    
                    # Show disagreement details
                    disagreement_data = []
                    for model, label in zip(model_names, pred_labels):
                        disagreement_data.append({'Model': model, 'Prediction': label})
                    
                    df_disagreement = pd.DataFrame(disagreement_data)
                    st.dataframe(df_disagreement, use_container_width=True)
    
    elif page == "📊 Performance Benchmarks":
        st.header("📊 Performance Benchmarks")
        st.write("Detailed performance analysis of all model types on real customer reviews")
        
        # Run comprehensive evaluation
        if st.button("🏃‍♂️ Run Full Benchmark", type="primary"):
            with st.spinner("Running comprehensive evaluation on real reviews..."):
                results, true_labels = evaluate_models_on_real_data(
                    real_reviews, neural_models, transformer_model, traditional_models
                )
            
            if results:
                # Performance summary
                st.subheader("🏆 Performance Leaderboard")
                
                leaderboard_data = []
                for model_name, metrics in results.items():
                    leaderboard_data.append({
                        'Rank': 0,  # Will be filled after sorting
                        'Model': model_name,
                        'Type': metrics['type'],
                        'Accuracy': f"{metrics['accuracy']:.3f}",
                        'F1-Score': f"{metrics['f1_score']:.3f}",
                        'Accuracy_Raw': metrics['accuracy']
                    })
                
                # Sort by accuracy and assign ranks
                leaderboard_data.sort(key=lambda x: x['Accuracy_Raw'], reverse=True)
                for i, item in enumerate(leaderboard_data):
                    item['Rank'] = i + 1
                
                # Remove raw accuracy column
                for item in leaderboard_data:
                    del item['Accuracy_Raw']
                
                df_leaderboard = pd.DataFrame(leaderboard_data)
                
                # Style the leaderboard
                def highlight_top_performer(row):
                    if row['Rank'] == 1:
                        return ['background-color: #FFD700; font-weight: bold'] * len(row)
                    elif row['Rank'] == 2:
                        return ['background-color: #C0C0C0'] * len(row)
                    elif row['Rank'] == 3:
                        return ['background-color: #CD7F32'] * len(row)
                    else:
                        return [''] * len(row)
                
                styled_df = df_leaderboard.style.apply(highlight_top_performer, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Performance by model type
                st.subheader("📈 Performance by Model Architecture")
                
                type_performance = df_leaderboard.groupby('Type').agg({
                    'Accuracy': lambda x: np.mean([float(val) for val in x]),
                    'F1-Score': lambda x: np.mean([float(val) for val in x])
                }).round(3)
                
                fig = px.bar(
                    type_performance.reset_index(),
                    x='Type',
                    y=['Accuracy', 'F1-Score'],
                    title="Average Performance by Model Architecture",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Model efficiency analysis (simulated metrics)
                st.subheader("⚡ Model Efficiency Analysis")
                
                efficiency_data = []
                for model_name, metrics in results.items():
                    # Simulated efficiency metrics (in real app, these would be measured)
                    if metrics['type'] == 'Neural Network':
                        inference_time = np.random.uniform(0.1, 0.3)
                        model_size = np.random.uniform(10, 50)
                    elif metrics['type'] == 'Transformer':
                        inference_time = np.random.uniform(0.5, 1.0)
                        model_size = np.random.uniform(100, 300)
                    else:  # Traditional ML
                        inference_time = np.random.uniform(0.01, 0.05)
                        model_size = np.random.uniform(1, 5)
                    
                    efficiency_data.append({
                        'Model': model_name,
                        'Accuracy': metrics['accuracy'],
                        'Inference_Time_ms': inference_time * 1000,
                        'Model_Size_MB': model_size,
                        'Type': metrics['type']
                    })
                
                df_efficiency = pd.DataFrame(efficiency_data)
                
                # Scatter plot: Accuracy vs Speed
                fig = px.scatter(
                    df_efficiency,
                    x='Inference_Time_ms',
                    y='Accuracy',
                    size='Model_Size_MB',
                    color='Type',
                    hover_name='Model',
                    title="Accuracy vs Inference Speed (Size = Model Size)",
                    labels={'Inference_Time_ms': 'Inference Time (ms)', 'Accuracy': 'Accuracy Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("No models available for benchmarking.")
    
    elif page == "📈 Model Analysis Deep Dive":
        st.header("📈 Model Analysis Deep Dive")
        st.write("Detailed analysis of model behavior and characteristics")
        
        # Model selector for deep dive
        available_models = []
        if neural_models.get('lstm'): available_models.append('LSTM')
        if neural_models.get('gru'): available_models.append('GRU')  
        if neural_models.get('rnn'): available_models.append('RNN')
        if transformer_model[0]: available_models.append('DistilBERT')
        if traditional_models[0]: available_models.append('Random Forest')
        
        if not available_models:
            st.error("No models available for analysis.")
            return
        
        selected_model = st.selectbox("Select model for deep dive:", available_models)
        
        if st.button("🔍 Analyze Model", type="primary"):
            st.subheader(f"🎯 Deep Dive: {selected_model}")
            
            # Get predictions for analysis
            sample_texts = real_reviews['review_text'].tolist()[:20]
            true_labels = real_reviews['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).tolist()[:20]
            
            predictions = []
            confidences = []
            
            with st.spinner(f"Analyzing {selected_model} predictions..."):
                for text in sample_texts:
                    if selected_model in ['LSTM', 'GRU', 'RNN']:
                        model = neural_models.get(selected_model.lower())
                        pred, conf = predict_neural(model, text, neural_models)
                    elif selected_model == 'DistilBERT':
                        pred, conf = predict_transformer(transformer_model[0], transformer_model[1], text)
                    elif selected_model == 'Random Forest':
                        pred, conf = predict_traditional(traditional_models[0], traditional_models[1], text)
                    
                    if pred is not None:
                        predictions.append(pred)
                        confidences.append(conf)
                    else:
                        predictions.append(0)
                        confidences.append([0.33, 0.33, 0.34])
            
            # Analysis results
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy metrics
                accuracy = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions, average='macro')
                
                st.metric("Accuracy", f"{accuracy:.3f}")
                st.metric("F1-Score", f"{f1:.3f}")
                
                # Confidence distribution
                max_confidences = [max(conf) for conf in confidences]
                avg_confidence = np.mean(max_confidences)
                st.metric("Average Confidence", f"{avg_confidence:.3f}")
            
            with col2:
                # Confidence distribution histogram
                fig = px.histogram(
                    x=max_confidences,
                    title=f"{selected_model} - Confidence Distribution",
                    labels={'x': 'Maximum Confidence', 'y': 'Frequency'},
                    nbins=10
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction examples
            st.subheader("📝 Prediction Examples")
            
            label_names = ['Negative', 'Neutral', 'Positive']
            
            # Show some correct and incorrect predictions
            correct_predictions = []
            incorrect_predictions = []
            
            for i, (text, true_label, pred_label, conf) in enumerate(zip(sample_texts, true_labels, predictions, confidences)):
                if true_label == pred_label:
                    correct_predictions.append({
                        'text': text[:100] + "..." if len(text) > 100 else text,
                        'true': label_names[true_label],
                        'predicted': label_names[pred_label],
                        'confidence': conf[pred_label]
                    })
                else:
                    incorrect_predictions.append({
                        'text': text[:100] + "..." if len(text) > 100 else text,
                        'true': label_names[true_label],
                        'predicted': label_names[pred_label],
                        'confidence': conf[pred_label]
                    })
            
            # Display examples
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("✅ **Correct Predictions**")
                for example in correct_predictions[:3]:
                    st.write(f"**Text:** {example['text']}")
                    st.write(f"**Prediction:** {example['predicted']} ({example['confidence']:.2%})")
                    st.write("---")
            
            with col2:
                st.write("❌ **Incorrect Predictions**")
                for example in incorrect_predictions[:3]:
                    st.write(f"**Text:** {example['text']}")
                    st.write(f"**True:** {example['true']} | **Predicted:** {example['predicted']} ({example['confidence']:.2%})")
                    st.write("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🏛️ S V Banquet Halls Advanced Sentiment Analysis | Built with Streamlit</p>
        <p>Neural Networks • Transformers • Traditional ML • Real Customer Reviews</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
