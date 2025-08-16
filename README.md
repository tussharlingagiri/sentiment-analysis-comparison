# S V Banquet Halls Sentiment Analysis Project

A comprehensive sentiment analysis project that extracts customer reviews from Google Business Profile Takeout data and compares different machine learning models to understand customer sentiment.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Data Extraction Process](#data-extraction-process)
- [Running the Demo Applications](#running-the-demo-applications)
- [Understanding the Results](#understanding-the-results)
- [Model Comparison Guide](#model-comparison-guide)
- [Business Insights](#business-insights)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

## 🎯 Project Overview

This project analyzes customer reviews for **S V Banquet Halls** (a banquet hall in Hyderabad, India) using Google Business Profile data. It demonstrates how different artificial intelligence models can automatically understand whether customer reviews are positive, negative, or neutral.

### What This Project Does:
- ✅ Extracts 440+ real customer reviews from Google Takeout data (2019-2025)
- ✅ Processes and cleans the review data for analysis
- ✅ Trains and compares 8 different AI models for sentiment analysis
- ✅ Provides interactive web demos to test different models
- ✅ Generates business insights and performance metrics

## 🚀 Getting Started

### Prerequisites

Before starting, you'll need:
- Python 3.11 or higher
- Access to Google Business Profile Takeout data
- Basic familiarity with running terminal commands

### Installation

1. **Clone or download this project**
   ```bash
   git clone https://github.com/tussharlingagiri/sentiment-analysis-comparison.git
   cd sentiment-analysis-comparison
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Extraction Process

### Step 1: Getting Your Google Takeout Data

1. **Request Google Takeout**
   - Go to [Google Takeout](https://takeout.google.com)
   - Select "Google Business Profile" data
   - Download the ZIP file when ready

2. **Extract the Data**
   - Unzip the downloaded file
   - Look for the folder: `Takeout/Google Business Profile/account-[ID]/location-[ID]/`
   - This folder contains your review files (reviews.json, reviews_001.json, etc.)

### Step 2: Organize Your Data

1. **Update the path in extract_reviews.py**
   ```python
   # Change this line to match your Takeout folder location
   location_path = "/path/to/your/Takeout/Google Business Profile/account-[ID]/location-[ID]"
   ```

2. **Run the extraction script**
   ```bash
   python extract_reviews.py
   ```

### What Gets Created:
- `sv_banquet_reviews_complete.csv` - All review data with metadata
- `sv_banquet_reviews_for_analysis.csv` - Reviews with text comments for AI analysis
- Console output showing statistics and insights

## 🖥️ Running the Demo Applications

### Basic Demo (Traditional ML Models)
```bash
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or .venv\Scripts\activate on Windows

# Run the basic demo
streamlit run streamlit_sentiment_demo.py --server.port=8501
```
- **Access at:** http://localhost:8501
- **Features:** Traditional machine learning models, business analytics, live prediction testing

### Advanced Demo (All Model Types)
```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # On macOS/Linux
# or .venv\Scripts\activate on Windows

# Run the advanced demo
streamlit run advanced_sentiment_demo.py --server.port=8502
```
- **Access at:** http://localhost:8502
- **Features:** Neural networks, transformers, traditional ML, comprehensive model comparison

### Running Both Demos Simultaneously
```bash
# Terminal 1 - Basic Demo
source .venv/bin/activate
streamlit run streamlit_sentiment_demo.py --server.port=8501

# Terminal 2 (new terminal window) - Advanced Demo
source .venv/bin/activate
streamlit run advanced_sentiment_demo.py --server.port=8502
```

## 📈 Understanding the Results

### For Non-Technical Users

#### What is Sentiment Analysis?
Sentiment analysis is like having a computer automatically read customer reviews and determine if they're:
- **😊 Positive** - Customer is happy (4-5 stars)
- **😐 Neutral** - Customer is okay (3 stars)  
- **😞 Negative** - Customer is unhappy (1-2 stars)

**Why We Use Simple 3-Category Classification:**
- ✅ **Matches Google's 1-5 star system** - Uses the actual rating scale customers understand
- ✅ **Business-friendly** - Clear, actionable categories that business owners can act upon
- ✅ **Industry standard** - Consistent with how all major review platforms work
- ✅ **High AI accuracy** - Well-defined boundaries make it easier for models to classify correctly
- ✅ **Real customer behavior** - Based on how customers actually rate businesses

*Note: We don't use extreme negative scales (like -5 to -12) because they're artificial, confusing for business owners, and don't match how customers actually think about their experiences.*

#### Key Metrics Explained

**1. Accuracy** 
- **What it means:** How often the AI gets the right answer
- **Example:** 85% accuracy = AI correctly identifies sentiment 85 out of 100 times
- **Good range:** 80%+ is excellent for this type of task

**2. Precision**
- **What it means:** When AI says "positive", how often is it actually positive?
- **Example:** 90% precision = When AI says "positive", it's right 9 out of 10 times
- **Why it matters:** Helps avoid false alarms

**3. Recall**
- **What it means:** Of all the actual positive reviews, how many did AI find?
- **Example:** 85% recall = AI found 85 out of 100 positive reviews
- **Why it matters:** Helps ensure we don't miss important feedback

**4. F1-Score**
- **What it means:** A balanced measure combining precision and recall
- **Range:** 0 to 1 (higher is better)
- **Why it matters:** Good overall measure of model performance

### Business Metrics Dashboard

The demos provide several business-relevant insights:

#### Review Analytics
- **Total Reviews:** How many customers left feedback
- **Average Rating:** Overall satisfaction score (1-5 stars)
- **Sentiment Distribution:** Percentage of positive/neutral/negative feedback
- **Response Rate:** How often you reply to customer reviews

#### Trend Analysis
- **Monthly/Yearly Trends:** How satisfaction changes over time
- **Peak Periods:** When you get the most reviews
- **Improvement Areas:** Categories with lower ratings

#### Customer Insights
- **Common Praise:** What customers love most
- **Common Complaints:** Areas needing improvement
- **Review Length:** How detailed customer feedback is

## 🤖 Model Comparison Guide

### Traditional Machine Learning Models

**1. Random Forest** 🌳
- **How it works:** Like asking 100 experts and taking the majority vote
- **Strengths:** Very reliable, handles different types of text well
- **Best for:** Consistent, interpretable results

**2. Support Vector Machine (SVM)** ⚡
- **How it works:** Finds the best way to separate positive from negative reviews
- **Strengths:** Great at finding patterns in text
- **Best for:** High accuracy on clear sentiment differences

**3. Logistic Regression** 📈
- **How it works:** Calculates probability of positive/negative sentiment
- **Strengths:** Fast, simple, easy to understand
- **Best for:** Quick analysis and baseline comparisons

**4. Naive Bayes** 🎯
- **How it works:** Uses word frequency to predict sentiment
- **Strengths:** Works well with limited data
- **Best for:** Fast processing of new reviews

### Neural Network Models

**5. LSTM (Long Short-Term Memory)** 🧠
- **How it works:** Remembers important parts of text while reading
- **Strengths:** Understands context and word relationships
- **Best for:** Complex sentiment patterns

**6. GRU (Gated Recurrent Unit)** ⚡
- **How it works:** Simplified version of LSTM, faster processing
- **Strengths:** Good balance of accuracy and speed
- **Best for:** Real-time sentiment analysis

**7. RNN (Recurrent Neural Network)** 🔄
- **How it works:** Processes text word by word, building understanding
- **Strengths:** Captures sequential patterns in text
- **Best for:** Understanding sentence structure

### Transformer Models

**8. DistilBERT** 🤖
- **How it works:** Advanced AI that understands context like humans
- **Strengths:** State-of-the-art accuracy, understands nuance
- **Best for:** Most accurate sentiment analysis

## 💼 Business Insights

### Sample Results for S V Banquet Halls

**Overall Performance:**
- 440 customer reviews analyzed (2019-2025)
- Average rating: 4.2/5 stars
- 83% positive sentiment
- 12% neutral sentiment  
- 5% negative sentiment

**Key Strengths (from positive reviews):**
- Excellent food quality and variety
- Beautiful venue decoration
- Professional staff service
- Good value for money
- Spacious halls for large events

**Areas for Improvement (from negative reviews):**
- AC/ventilation issues in some halls
- Parking space limitations
- Occasional delays in service timing

**Business Recommendations:**
1. **Maintain Strengths:** Continue focus on food quality and service
2. **Address Issues:** Improve ventilation and parking arrangements
3. **Monitor Trends:** Track monthly sentiment to catch issues early
4. **Respond to Reviews:** Increase response rate to show customer care

## 🔧 Technical Details

### Project Structure
```
sentiment-analysis-comparison/
├── extract_reviews.py                      # Data extraction from Takeout
├── streamlit_sentiment_demo.py             # Basic demo app
├── advanced_sentiment_demo.py              # Advanced demo app
├── nlp_sentiment_analysis.py               # Model training script
├── fix_models.py                           # Model verification and retraining
├── requirements.txt                        # Python dependencies
├── README.md                               # This documentation
├── LICENSE                                 # MIT License
├── sv_banquet_reviews_complete.csv         # Complete dataset (440 reviews)
├── sv_banquet_reviews_for_analysis.csv     # Text reviews for analysis (132 reviews)
├── train.csv / test.csv                    # Training/testing splits
├── lstm_model.pth                          # LSTM neural network
├── gru_model.pth                           # GRU neural network
├── rnn_model.pth                           # RNN neural network
├── word2vec.model                          # Word embedding model
├── models/                                 # Traditional ML models
│   ├── best_sentiment_model.pkl            # Random Forest model
│   └── tfidf_vectorizer.pkl                # TF-IDF vectorizer
└── results/                                # Transformer model checkpoints
    ├── checkpoint-best/                    # Best DistilBERT model
    └── checkpoint-1124/                    # Latest DistilBERT model
```

### Data Files Generated
- `sv_banquet_reviews_complete.csv` - Complete dataset (440 reviews)
- `sv_banquet_reviews_for_analysis.csv` - Text reviews only (132 reviews)
- `train.csv` / `test.csv` - Training/testing splits

### Model Files
- `lstm_model.pth` - LSTM neural network
- `gru_model.pth` - GRU neural network  
- `rnn_model.pth` - RNN neural network
- `word2vec.model` - Word embeddings
- `results/checkpoint-best/` - DistilBERT transformer model

## 🐛 Troubleshooting

### Common Issues

**1. "No reviews found" error**
```bash
# Check if Takeout path is correct
ls "/path/to/your/Takeout/Google Business Profile/"
# Update path in extract_reviews.py
```

**2. "Model not loading" error**
```bash
# Run model fix script
python fix_models.py
```

**3. Streamlit won't start**
```bash
# Check if port is available
lsof -i :8501
# Kill existing process if needed
pkill -f streamlit
```

**4. Missing dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt
```

### Performance Tips

**For Better Model Performance:**
- Ensure you have at least 100+ reviews with text
- Balance positive/negative examples if possible
- Remove duplicate reviews before training

**For Faster Processing:**
- Use traditional ML models for quick analysis
- Use neural networks for better accuracy
- Use transformer models for best results

## 📞 Support

### Getting Help

1. **Check the error message** - Most issues have clear error descriptions
2. **Verify file paths** - Ensure all paths point to correct locations
3. **Check Python environment** - Make sure all packages are installed
4. **Review model status** - Use the model checker in demos

### Contact Information

- **GitHub Issues:** [Create an issue](https://github.com/tussharlingagiri/sentiment-analysis-comparison/issues)
- **Email:** tussharlingagiri@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Business Profile for providing review data export
- Streamlit for the interactive web framework
- Hugging Face for transformer models
- PyTorch for neural network implementation
- scikit-learn for traditional ML algorithms

---

**📊 Ready to analyze your customer feedback? Start with the data extraction process and explore the demos to see how AI can help understand your customers better!**
