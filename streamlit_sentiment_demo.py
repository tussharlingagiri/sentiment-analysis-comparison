#!/usr/bin/env python3
"""
Streamlit app for S V Banquet Halls Sentiment Analysis Demo
Showcases real reviews data and different ML models for audience demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import os
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="S V Banquet Halls - Sentiment Analysis Demo",
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
    .review-card {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        margin: 0.5rem 0;
    }
    .positive { border-left: 5px solid #28A745; }
    .negative { border-left: 5px solid #DC3545; }
    .neutral { border-left: 5px solid #FFC107; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        # Load real reviews
        real_reviews = pd.read_csv('/Users/tusshar/sentiment-analysis-comparison/sv_banquet_reviews_for_analysis.csv')
        complete_reviews = pd.read_csv('/Users/tusshar/sentiment-analysis-comparison/sv_banquet_reviews_complete.csv')
        combined_reviews = pd.read_csv('/Users/tusshar/sentiment-analysis-comparison/combined_banquet_reviews.csv')
        
        return real_reviews, complete_reviews, combined_reviews
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    try:
        model = joblib.load('/Users/tusshar/sentiment-analysis-comparison/models/best_sentiment_model.pkl')
        vectorizer = joblib.load('/Users/tusshar/sentiment-analysis-comparison/models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

def train_models_live(X_train, X_test, y_train, y_test):
    """Train models live for demonstration"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(kernel='linear', random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'actual': y_test
            }
    
    return results

def create_sentiment_distribution(df):
    """Create sentiment distribution chart"""
    sentiment_counts = df['sentiment'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={
            'positive': '#28A745',
            'negative': '#DC3545',
            'neutral': '#FFC107'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_rating_distribution(df):
    """Create rating distribution chart"""
    if 'star_rating_numeric' in df.columns:
        rating_counts = df['star_rating_numeric'].value_counts().sort_index()
        
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Rating Distribution",
            labels={'x': 'Star Rating', 'y': 'Number of Reviews'},
            color=rating_counts.values,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(showlegend=False)
        return fig
    return None

def create_temporal_analysis(df):
    """Create temporal analysis chart"""
    if 'create_date' in df.columns:
        df['create_date'] = pd.to_datetime(df['create_date'])
        df['year_month'] = df['create_date'].dt.to_period('M')
        
        temporal_data = df.groupby(['year_month', 'sentiment_label']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        colors = {'positive': '#28A745', 'negative': '#DC3545', 'neutral': '#FFC107'}
        
        for sentiment in temporal_data.columns:
            fig.add_trace(go.Scatter(
                x=temporal_data.index.astype(str),
                y=temporal_data[sentiment],
                mode='lines+markers',
                name=sentiment.capitalize(),
                line=dict(color=colors.get(sentiment, '#2E86C1'), width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Reviews",
            hovermode='x unified'
        )
        
        return fig
    return None

def create_wordcloud(texts, sentiment_type):
    """Create word cloud"""
    text = ' '.join(texts)
    
    # Clean text
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    if text.strip():
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis' if sentiment_type == 'positive' else 'Reds' if sentiment_type == 'negative' else 'YlOrBr'
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.title(f'{sentiment_type.capitalize()} Reviews Word Cloud', fontsize=16)
        return fig
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏛️ S V Banquet Halls - Sentiment Analysis Demo</h1>', unsafe_allow_html=True)
    
    # Load data
    real_reviews, complete_reviews, combined_reviews = load_data()
    
    if real_reviews is None:
        st.error("Could not load data. Please ensure the CSV files exist.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        [
            "📊 Data Overview", 
            "📈 Visual Analytics", 
            "🤖 Model Comparison", 
            "🔮 Live Prediction", 
            "💼 Business Insights",
            "📝 Sample Reviews"
        ]
    )
    
    if page == "📊 Data Overview":
        st.header("Data Overview")
        
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
            date_range = f"{complete_reviews['create_date'].min()} to {complete_reviews['create_date'].max()}"
            st.markdown(
                f'<div class="metric-container"><h3>Date Range</h3><h2>2019-2025</h2></div>',
                unsafe_allow_html=True
            )
        
        st.subheader("Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Real Reviews Dataset:**")
            st.write(f"- Reviews with text: {len(real_reviews)}")
            st.write(f"- Average text length: {real_reviews['review_text'].str.len().mean():.1f} characters")
            st.write(f"- Languages: Mixed (English, Telugu)")
            
            st.write("**Sentiment Distribution:**")
            sentiment_dist = real_reviews['sentiment'].value_counts()
            for sentiment, count in sentiment_dist.items():
                percentage = count / len(real_reviews) * 100
                st.write(f"- {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        with col2:
            st.write("**Complete Dataset:**")
            st.write(f"- Total reviews: {len(complete_reviews)}")
            st.write(f"- Reviews with business replies: {complete_reviews['has_reply'].sum()}")
            st.write(f"- Response rate: {complete_reviews['has_reply'].mean()*100:.1f}%")
            
            st.write("**Rating Distribution:**")
            rating_dist = complete_reviews['star_rating_numeric'].value_counts().sort_index()
            for rating, count in rating_dist.items():
                percentage = count / len(complete_reviews) * 100
                st.write(f"- {rating} stars: {count} ({percentage:.1f}%)")
        
        # Sample data
        st.subheader("Sample Reviews")
        st.dataframe(
            real_reviews[['reviewer_name', 'rating', 'review_text', 'sentiment', 'create_date']].head(10),
            use_container_width=True
        )
    
    elif page == "📈 Visual Analytics":
        st.header("Visual Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            fig_sentiment = create_sentiment_distribution(real_reviews)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Rating distribution
            fig_rating = create_rating_distribution(complete_reviews)
            if fig_rating:
                st.plotly_chart(fig_rating, use_container_width=True)
        
        # Temporal analysis
        st.subheader("Sentiment Trends Over Time")
        fig_temporal = create_temporal_analysis(complete_reviews)
        if fig_temporal:
            st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Word clouds
        st.subheader("Word Clouds by Sentiment")
        
        sentiment_tabs = st.tabs(["Positive", "Negative", "Neutral"])
        
        for i, sentiment in enumerate(['positive', 'negative', 'neutral']):
            with sentiment_tabs[i]:
                sentiment_reviews = real_reviews[real_reviews['sentiment'] == sentiment]['review_text'].tolist()
                if sentiment_reviews:
                    fig_wc = create_wordcloud(sentiment_reviews, sentiment)
                    if fig_wc:
                        st.pyplot(fig_wc)
                else:
                    st.write(f"No {sentiment} reviews with sufficient text.")
    
    elif page == "🤖 Model Comparison":
        st.header("Machine Learning Model Comparison")
        
        st.write("Compare different sentiment analysis models on the real reviews dataset.")
        
        if st.button("Train Models Live", type="primary"):
            # Prepare data
            X = real_reviews['review_text'].values
            y = real_reviews['sentiment'].values
            
            # Vectorize
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', lowercase=True)
            X_vectorized = vectorizer.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train models
            results = train_models_live(X_train, X_test, y_train, y_test)
            
            # Display results
            st.subheader("Model Performance Comparison")
            
            model_names = list(results.keys())
            accuracies = [results[name]['accuracy'] for name in model_names]
            
            # Bar chart
            fig = px.bar(
                x=model_names,
                y=accuracies,
                title="Model Accuracy Comparison",
                labels={'x': 'Models', 'y': 'Accuracy'},
                color=accuracies,
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy Scores")
                for name, result in results.items():
                    st.metric(name, f"{result['accuracy']:.3f}")
            
            with col2:
                st.subheader("Best Model Details")
                best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
                best_accuracy = results[best_model]['accuracy']
                st.success(f"**Best Model:** {best_model}")
                st.write(f"**Accuracy:** {best_accuracy:.3f}")
                
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(
                    results[best_model]['actual'], 
                    results[best_model]['predictions']
                )
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['negative', 'neutral', 'positive'],
                           yticklabels=['negative', 'neutral', 'positive'])
                plt.title(f'Confusion Matrix - {best_model}')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
    
    elif page == "🔮 Live Prediction":
        st.header("Live Sentiment Prediction")
        
        # Load pre-trained model
        model, vectorizer = load_models()
        
        if model and vectorizer:
            st.success("✅ Pre-trained model loaded successfully!")
            
            # Text input
            user_input = st.text_area(
                "Enter a review to analyze:",
                placeholder="E.g., Great place for functions, excellent food and service!",
                height=100
            )
            
            if st.button("Analyze Sentiment", type="primary") and user_input:
                # Predict
                input_vectorized = vectorizer.transform([user_input])
                prediction = model.predict(input_vectorized)[0]
                confidence = model.predict_proba(input_vectorized)[0].max()
                
                # Display result
                col1, col2, col3 = st.columns(3)
                
                with col2:
                    if prediction == 'positive':
                        st.success(f"😊 **Positive**")
                        st.success(f"Confidence: {confidence:.2%}")
                    elif prediction == 'negative':
                        st.error(f"😞 **Negative**")
                        st.error(f"Confidence: {confidence:.2%}")
                    else:
                        st.warning(f"😐 **Neutral**")
                        st.warning(f"Confidence: {confidence:.2%}")
                
                # Show probability distribution
                st.subheader("Probability Distribution")
                proba = model.predict_proba(input_vectorized)[0]
                classes = model.classes_
                
                prob_df = pd.DataFrame({
                    'Sentiment': classes,
                    'Probability': proba
                })
                
                fig = px.bar(
                    prob_df,
                    x='Sentiment',
                    y='Probability',
                    title="Prediction Confidence",
                    color='Probability',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Could not load pre-trained model. Please train a model first.")
            
            # Option to train a simple model
            if st.button("Train Simple Model"):
                with st.spinner("Training a simple model..."):
                    X = real_reviews['review_text'].values
                    y = real_reviews['sentiment'].values
                    
                    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                    X_vectorized = vectorizer.fit_transform(X)
                    
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    model.fit(X_vectorized, y)
                    
                    st.success("✅ Simple model trained! Try entering a review above.")
                    st.session_state['temp_model'] = model
                    st.session_state['temp_vectorizer'] = vectorizer
    
    elif page == "💼 Business Insights":
        st.header("Business Intelligence Dashboard")
        
        st.subheader("Key Performance Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Customer satisfaction trend
            yearly_avg = complete_reviews.groupby('create_year')['star_rating_numeric'].mean()
            fig = px.line(
                x=yearly_avg.index,
                y=yearly_avg.values,
                title="Average Rating Trend",
                markers=True
            )
            fig.update_layout(
                yaxis=dict(range=[1, 5]),
                xaxis_title="Year",
                yaxis_title="Average Rating"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Response rate analysis
            response_rate = complete_reviews.groupby('create_year')['has_reply'].mean() * 100
            fig = px.bar(
                x=response_rate.index,
                y=response_rate.values,
                title="Business Response Rate (%)",
                color=response_rate.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Review volume trend
            review_volume = complete_reviews.groupby('create_year').size()
            fig = px.area(
                x=review_volume.index,
                y=review_volume.values,
                title="Review Volume by Year"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Issues analysis
        st.subheader("Common Issues Analysis")
        
        negative_reviews = real_reviews[real_reviews['sentiment'] == 'negative']['review_text'].str.lower()
        
        issues = {
            'Parking': negative_reviews.str.contains('parking|park').sum(),
            'AC/Cooling': negative_reviews.str.contains('ac|air|cool|hot').sum(),
            'Space/Capacity': negative_reviews.str.contains('space|small|capacity|cramped').sum(),
            'Staff': negative_reviews.str.contains('staff|service|rude').sum(),
            'Lift/Elevator': negative_reviews.str.contains('lift|elevator').sum()
        }
        
        issues_df = pd.DataFrame(list(issues.items()), columns=['Issue', 'Mentions'])
        issues_df = issues_df.sort_values('Mentions', ascending=True)
        
        fig = px.bar(
            issues_df,
            x='Mentions',
            y='Issue',
            orientation='h',
            title="Most Common Issues in Negative Reviews",
            color='Mentions',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Business Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🚨 Priority Issues to Address:**
            1. **Parking Solutions** - Most frequently mentioned issue
            2. **AC Maintenance** - Critical for customer comfort
            3. **Space Optimization** - Better layout and capacity planning
            4. **Staff Training** - Improve service quality
            5. **Infrastructure** - Fix elevator/lift issues
            """)
        
        with col2:
            st.markdown("""
            **✅ Strengths to Leverage:**
            1. **Food Quality** - Consistently praised in positive reviews
            2. **Ambience** - Customers appreciate the venue atmosphere
            3. **Small Events** - Perfect size for intimate gatherings
            4. **Location** - Good accessibility mentioned
            5. **Responsive Management** - 20.7% response rate shows engagement
            """)
    
    elif page == "📝 Sample Reviews":
        st.header("Sample Reviews by Sentiment")
        
        sentiment_filter = st.selectbox("Filter by sentiment:", ["All", "Positive", "Negative", "Neutral"])
        
        if sentiment_filter == "All":
            filtered_reviews = real_reviews
        else:
            filtered_reviews = real_reviews[real_reviews['sentiment'] == sentiment_filter.lower()]
        
        st.write(f"Showing {len(filtered_reviews)} reviews")
        
        # Display reviews as cards
        for idx, review in filtered_reviews.iterrows():
            sentiment_class = review['sentiment']
            
            st.markdown(f"""
            <div class="review-card {sentiment_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong>{review['reviewer_name']}</strong>
                    <div>
                        <span style="color: #666;">{'⭐' * review['rating']}</span>
                        <span style="margin-left: 10px; padding: 3px 8px; border-radius: 12px; font-size: 12px; 
                              background-color: {'#28A745' if sentiment_class == 'positive' else '#DC3545' if sentiment_class == 'negative' else '#FFC107'}; 
                              color: white;">
                            {sentiment_class.capitalize()}
                        </span>
                    </div>
                </div>
                <p style="margin: 10px 0; font-style: italic;">"{review['review_text']}"</p>
                <small style="color: #666;">{review['create_date']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🏛️ S V Banquet Halls Sentiment Analysis Demo | Built with Streamlit</p>
        <p>Data: 440 real customer reviews from Google Business Profile (2019-2025)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
