

import pandas as pd
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_output_directory():
    """Create directory for processed sentiment data"""
    base_path = Path(__file__).parent.parent / 'data_processed' / 'sentiment'
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def initialize_sentiment_analyzers():
    """Initialize VADER and FinBERT sentiment analyzers"""
    print("Initializing sentiment analyzers...")
    
    # VADER for social media text
    vader = SentimentIntensityAnalyzer()
    print("✓ VADER loaded")
    
    # FinBERT for financial text (may take time to download first time)
    try:
        print("Loading FinBERT model (this may take a moment)...")
        finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
        print("✓ FinBERT loaded")
    except Exception as e:
        print(f"⚠️  FinBERT loading failed: {str(e)}")
        print("Continuing with VADER only...")
        finbert = None
    
    return vader, finbert

def analyze_vader_sentiment(text, vader):
    """
    Analyze sentiment using VADER
    
    Args:
        text (str): Text to analyze
        vader: VADER analyzer instance
    
    Returns:
        dict: Sentiment scores
    """
    if pd.isna(text) or not text:
        return {'compound': 0, 'positive': 0, 'neutral': 1, 'negative': 0}
    
    scores = vader.polarity_scores(str(text))
    return scores

def analyze_finbert_sentiment(text, finbert):
    """
    Analyze sentiment using FinBERT
    
    Args:
        text (str): Text to analyze
        finbert: FinBERT pipeline
    
    Returns:
        dict: Sentiment label and score
    """
    if finbert is None or pd.isna(text) or not text:
        return {'label': 'neutral', 'score': 0.5}
    
    try:
        # Truncate text to max length
        text = str(text)[:512]
        result = finbert(text)[0]
        return result
    except:
        return {'label': 'neutral', 'score': 0.5}

def process_reddit_data(input_filepath, vader, finbert):
    """
    Process Reddit data and calculate sentiment scores
    
    Args:
        input_filepath (str): Path to Reddit data CSV
        vader: VADER analyzer
        finbert: FinBERT analyzer
    
    Returns:
        pd.DataFrame: Data with sentiment scores
    """
    print(f"\nProcessing Reddit data: {input_filepath}")
    
    # Load data
    df = pd.read_csv(input_filepath)
    print(f"  Loaded {len(df)} posts")
    
    # Combine title and text for sentiment analysis
    df['full_text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
    
    print("  Analyzing sentiment with VADER...")
    # VADER sentiment (fast)
    vader_scores = df['full_text'].apply(lambda x: analyze_vader_sentiment(x, vader))
    df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
    df['vader_positive'] = vader_scores.apply(lambda x: x['pos'])
    df['vader_neutral'] = vader_scores.apply(lambda x: x['neu'])
    df['vader_negative'] = vader_scores.apply(lambda x: x['neg'])
    
    # Classify sentiment based on compound score
    df['vader_label'] = df['vader_compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    
    if finbert is not None:
        print("  Analyzing sentiment with FinBERT...")
        # FinBERT sentiment (slower but more accurate for financial text)
        # Sample a subset for efficiency
        sample_size = min(1000, len(df))
        sample_indices = np.random.choice(df.index, sample_size, replace=False)
        
        df['finbert_label'] = 'neutral'
        df['finbert_score'] = 0.5
        
        for idx in sample_indices:
            result = analyze_finbert_sentiment(df.loc[idx, 'full_text'], finbert)
            df.loc[idx, 'finbert_label'] = result['label']
            df.loc[idx, 'finbert_score'] = result['score']
    
    print(f"✓ Sentiment analysis completed")
    
    return df

def process_news_data(input_filepath, vader, finbert):
    """
    Process news data and calculate sentiment scores
    
    Args:
        input_filepath (str): Path to news data CSV
        vader: VADER analyzer
        finbert: FinBERT analyzer
    
    Returns:
        pd.DataFrame: Data with sentiment scores
    """
    print(f"\nProcessing News data: {input_filepath}")
    
    # Load data
    df = pd.read_csv(input_filepath)
    print(f"  Loaded {len(df)} articles")
    
    # Combine title, description, and content
    df['full_text'] = (
        df['title'].fillna('') + ' ' + 
        df['description'].fillna('') + ' ' + 
        df['content'].fillna('')
    )
    
    print("  Analyzing sentiment with VADER...")
    # VADER sentiment
    vader_scores = df['full_text'].apply(lambda x: analyze_vader_sentiment(x, vader))
    df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
    df['vader_positive'] = vader_scores.apply(lambda x: x['pos'])
    df['vader_neutral'] = vader_scores.apply(lambda x: x['neu'])
    df['vader_negative'] = vader_scores.apply(lambda x: x['neg'])
    
    df['vader_label'] = df['vader_compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    
    if finbert is not None:
        print("  Analyzing sentiment with FinBERT (financial news)...")
        # FinBERT is particularly good for financial news
        # Process all articles (or sample if too many)
        sample_size = min(500, len(df))
        sample_indices = np.random.choice(df.index, sample_size, replace=False)
        
        df['finbert_label'] = 'neutral'
        df['finbert_score'] = 0.5
        
        for idx in sample_indices:
            result = analyze_finbert_sentiment(df.loc[idx, 'full_text'], finbert)
            df.loc[idx, 'finbert_label'] = result['label']
            df.loc[idx, 'finbert_score'] = result['score']
    
    print(f"✓ Sentiment analysis completed")
    
    return df

def aggregate_daily_sentiment(df, date_column='created_utc', ticker_column='keyword'):
    """
    Aggregate sentiment scores by date and ticker
    
    Args:
        df (pd.DataFrame): Data with sentiment scores
        date_column (str): Name of date column
        ticker_column (str): Name of ticker column
    
    Returns:
        pd.DataFrame: Aggregated daily sentiment
    """
    print("\n  Aggregating daily sentiment...")
    
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    df['date'] = df[date_column].dt.date
    
    # Aggregate by date and ticker
    daily_sentiment = df.groupby(['date', ticker_column]).agg({
        'vader_compound': ['mean', 'std', 'min', 'max'],
        'vader_positive': 'mean',
        'vader_negative': 'mean',
        'vader_neutral': 'mean'
    }).reset_index()
    
    # Flatten column names
    daily_sentiment.columns = ['_'.join(col).strip('_') for col in daily_sentiment.columns]
    
    # Rename for clarity
    daily_sentiment = daily_sentiment.rename(columns={
        'date_': 'date',
        f'{ticker_column}_': 'ticker'
    })
    
    return daily_sentiment

def display_sentiment_summary(df, data_type='Reddit'):
    """Display sentiment analysis summary"""
    print("\n" + "="*60)
    print(f"{data_type.upper()} SENTIMENT SUMMARY")
    print("="*60)
    
    print(f"\nTotal items analyzed: {len(df)}")
    
    print("\nVADER Sentiment Distribution:")
    print(df['vader_label'].value_counts())
    
    print("\nVADER Score Statistics:")
    print(df['vader_compound'].describe())
    
    if 'finbert_label' in df.columns:
        print("\nFinBERT Sentiment Distribution:")
        print(df['finbert_label'].value_counts())
    
    print("="*60 + "\n")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Sentiment Analysis - VADER & FinBERT                   ║
    ║  Research: Hybrid Sentiment-Technical Transformers      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize analyzers
    vader, finbert = initialize_sentiment_analyzers()
    
    output_path = create_output_directory()
    data_raw_path = Path(__file__).parent.parent / 'data_raw'
    
    # Process Reddit data
    reddit_path = data_raw_path / 'reddit'
    reddit_files = list(reddit_path.glob('reddit_all_posts_*.csv'))
    
    if reddit_files:
        latest_reddit = max(reddit_files, key=lambda p: p.stat().st_mtime)
        reddit_sentiment = process_reddit_data(latest_reddit, vader, finbert)
        
        # Save
        output_file = output_path / f'reddit_sentiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        reddit_sentiment.to_csv(output_file, index=False)
        print(f"✓ Saved to: {output_file}")
        
        display_sentiment_summary(reddit_sentiment, 'Reddit')
        
        # Aggregate daily
        daily_reddit = aggregate_daily_sentiment(reddit_sentiment)
        daily_file = output_path / f'reddit_daily_sentiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        daily_reddit.to_csv(daily_file, index=False)
        print(f"✓ Daily aggregation saved to: {daily_file}")
    else:
        print("⚠️  No Reddit data found. Run collect_reddit_data.py first.")
    
    # Process News data
    news_path = data_raw_path / 'news'
    news_files = list(news_path.glob('news_all_articles_*.csv'))
    
    if news_files:
        latest_news = max(news_files, key=lambda p: p.stat().st_mtime)
        news_sentiment = process_news_data(latest_news, vader, finbert)
        
        # Save
        output_file = output_path / f'news_sentiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        news_sentiment.to_csv(output_file, index=False)
        print(f"✓ Saved to: {output_file}")
        
        display_sentiment_summary(news_sentiment, 'News')
        
        # Aggregate daily
        daily_news = aggregate_daily_sentiment(
            news_sentiment, 
            date_column='published_at',
            ticker_column='ticker'
        )
        daily_file = output_path / f'news_daily_sentiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        daily_news.to_csv(daily_file, index=False)
        print(f"✓ Daily aggregation saved to: {daily_file}")
    else:
        print("⚠️  No news data found. Run collect_news_data.py first.")
    
    print("\n✓ Sentiment analysis completed!")
    print("\nNext steps:")
    print("1. Review sentiment scores in data_processed/sentiment/")
    print("2. Merge with technical indicators for hybrid dataset")
    print("3. Begin model training")
