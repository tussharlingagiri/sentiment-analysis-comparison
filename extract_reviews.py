#!/usr/bin/env python3
"""
Extract and consolidate Google Business Profile reviews from Takeout data
for S V Banquet Halls sentiment analysis
"""

import json
import pandas as pd
import os
from datetime import datetime
import glob

def extract_reviews_from_takeout():
    """Extract all reviews from Google Takeout data and create a consolidated dataset"""
    
    # Define the path to the location folder
    location_path = "/Users/tusshar/Documents/Takeout/Google Business Profile/account-109784038955748369603/location-3692820271974439325"
    
    # Find all review files
    review_files = glob.glob(os.path.join(location_path, "reviews*.json"))
    
    print(f"Found {len(review_files)} review files to process")
    
    all_reviews = []
    
    # Process each review file
    for file_path in review_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'reviews' in data:
                for review in data['reviews']:
                    # Extract review data
                    review_data = {
                        'reviewer_name': review.get('reviewer', {}).get('displayName', 'Anonymous'),
                        'star_rating': review.get('starRating', ''),
                        'comment': review.get('comment', ''),
                        'create_time': review.get('createTime', ''),
                        'update_time': review.get('updateTime', ''),
                        'review_id': review.get('name', '').split('/')[-1] if review.get('name') else '',
                        'has_reply': 'reviewReply' in review,
                        'reply_comment': review.get('reviewReply', {}).get('comment', '') if 'reviewReply' in review else '',
                        'reply_time': review.get('reviewReply', {}).get('updateTime', '') if 'reviewReply' in review else '',
                        'source_file': os.path.basename(file_path)
                    }
                    
                    # Convert star ratings to numeric
                    rating_map = {
                        'ONE': 1,
                        'TWO': 2,
                        'THREE': 3,
                        'FOUR': 4,
                        'FIVE': 5
                    }
                    review_data['star_rating_numeric'] = rating_map.get(review_data['star_rating'], 0)
                    
                    # Parse dates
                    try:
                        if review_data['create_time']:
                            create_dt = datetime.fromisoformat(review_data['create_time'].replace('Z', '+00:00'))
                            review_data['create_date'] = create_dt.strftime('%Y-%m-%d')
                            review_data['create_year'] = create_dt.year
                            review_data['create_month'] = create_dt.month
                    except:
                        review_data['create_date'] = ''
                        review_data['create_year'] = ''
                        review_data['create_month'] = ''
                    
                    # Determine sentiment label based on rating
                    if review_data['star_rating_numeric'] >= 4:
                        review_data['sentiment_label'] = 'positive'
                    elif review_data['star_rating_numeric'] == 3:
                        review_data['sentiment_label'] = 'neutral'
                    elif review_data['star_rating_numeric'] <= 2:
                        review_data['sentiment_label'] = 'negative'
                    else:
                        review_data['sentiment_label'] = 'unknown'
                    
                    all_reviews.append(review_data)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return all_reviews

def create_datasets():
    """Create different datasets for analysis"""
    
    # Extract all reviews
    reviews = extract_reviews_from_takeout()
    
    if not reviews:
        print("No reviews found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(reviews)
    
    # Remove duplicates based on review_id
    df = df.drop_duplicates(subset=['review_id'], keep='first')
    
    # Sort by create_time
    df = df.sort_values('create_time')
    
    print(f"\nExtracted {len(df)} unique reviews")
    
    # Basic statistics
    print("\nReview Statistics:")
    print(f"Date range: {df['create_date'].min()} to {df['create_date'].max()}")
    print(f"Star rating distribution:")
    print(df['star_rating_numeric'].value_counts().sort_index())
    print(f"\nSentiment distribution:")
    print(df['sentiment_label'].value_counts())
    print(f"\nReviews with comments: {len(df[df['comment'] != ''])}")
    print(f"Reviews with business replies: {len(df[df['has_reply'] == True])}")
    
    # Save complete dataset
    df.to_csv('/Users/tusshar/sentiment-analysis-comparison/sv_banquet_reviews_complete.csv', index=False)
    print(f"\nSaved complete dataset to: sv_banquet_reviews_complete.csv")
    
    # Create dataset with only reviews that have comments (for sentiment analysis)
    df_with_comments = df[df['comment'] != ''].copy()
    
    if len(df_with_comments) > 0:
        # Select relevant columns for sentiment analysis
        sentiment_df = df_with_comments[['reviewer_name', 'star_rating_numeric', 'comment', 
                                       'create_date', 'sentiment_label']].copy()
        sentiment_df.rename(columns={
            'star_rating_numeric': 'rating',
            'comment': 'review_text',
            'sentiment_label': 'sentiment'
        }, inplace=True)
        
        sentiment_df.to_csv('/Users/tusshar/sentiment-analysis-comparison/sv_banquet_reviews_for_analysis.csv', index=False)
        print(f"Saved sentiment analysis dataset to: sv_banquet_reviews_for_analysis.csv")
        print(f"Reviews with text comments: {len(sentiment_df)}")
    
    # Create yearly breakdown
    if 'create_year' in df.columns:
        yearly_stats = df.groupby('create_year').agg({
            'star_rating_numeric': ['count', 'mean'],
            'sentiment_label': lambda x: (x == 'positive').sum()
        }).round(2)
        print(f"\nYearly breakdown:")
        print(yearly_stats)
    
    # Create business insights
    print(f"\nBusiness Insights for S V Banquet Halls:")
    avg_rating = df['star_rating_numeric'].mean()
    print(f"Average rating: {avg_rating:.2f}/5")
    
    positive_percentage = (df['sentiment_label'] == 'positive').sum() / len(df) * 100
    print(f"Positive sentiment: {positive_percentage:.1f}%")
    
    # Most common complaints/compliments (from comments)
    if len(df_with_comments) > 0:
        print(f"\nSample positive reviews:")
        positive_reviews = df_with_comments[df_with_comments['sentiment_label'] == 'positive']['comment'].head(3)
        for i, review in enumerate(positive_reviews, 1):
            print(f"{i}. {review[:100]}...")
        
        if len(df_with_comments[df_with_comments['sentiment_label'] == 'negative']) > 0:
            print(f"\nSample negative reviews:")
            negative_reviews = df_with_comments[df_with_comments['sentiment_label'] == 'negative']['comment'].head(3)
            for i, review in enumerate(negative_reviews, 1):
                print(f"{i}. {review[:100]}...")

if __name__ == "__main__":
    create_datasets()
