

from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# NewsAPI credentials
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Stock-related keywords to search
NEWS_QUERIES = {
    'AAPL': 'Apple stock OR AAPL',
    'GOOGL': 'Google stock OR Alphabet OR GOOGL',
    'TSLA': 'Tesla stock OR TSLA OR Elon Musk Tesla',
    'AMZN': 'Amazon stock OR AMZN',
    'MSFT': 'Microsoft stock OR MSFT',
    'RELIANCE': 'Reliance Industries stock',
    'TCS': 'Tata Consultancy Services OR TCS stock',
    'INFY': 'Infosys stock',
    'JKH': 'John Keells Holdings',
    'COMB': 'Commercial Bank Sri Lanka',
    'DIAL': 'Dialog Axiata'
}

# News sources for financial news
FINANCIAL_SOURCES = [
    'bloomberg',
    'business-insider',
    'financial-post',
    'fortune',
    'the-wall-street-journal',
    'cnbc',
    'reuters'
]

def create_news_client():
    """Initialize NewsAPI client"""
    if not NEWS_API_KEY:
        print("⚠️  NewsAPI key not found!")
        print("\nPlease set up your .env file with:")
        print("NEWS_API_KEY=your_api_key_here")
        print("\nGet your free API key from: https://newsapi.org/")
        print("(Free tier: 100 requests/day)")
        return None
    
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        print("✓ NewsAPI connection successful")
        return newsapi
    except Exception as e:
        print(f"✗ NewsAPI connection failed: {str(e)}")
        return None

def create_data_directory():
    """Create directory for news data"""
    base_path = Path(__file__).parent.parent / 'data_raw' / 'news'
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def collect_news_for_query(newsapi, query, ticker, from_date, to_date, language='en'):
    """
    Collect news articles for a specific query
    
    Args:
        newsapi: NewsAPI client
        query (str): Search query
        ticker (str): Stock ticker symbol
        from_date (str): Start date (YYYY-MM-DD)
        to_date (str): End date (YYYY-MM-DD)
        language (str): Language code
    
    Returns:
        list: List of article dictionaries
    """
    articles_data = []
    
    try:
        print(f"  Searching for: {ticker} ({query[:50]}...)")
        
        # Get all articles
        response = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language=language,
            sort_by='relevancy',
            page_size=100  # Max allowed by free tier
        )
        
        if response['status'] == 'ok':
            articles = response['articles']
            
            for article in articles:
                article_data = {
                    'ticker': ticker,
                    'query': query,
                    'source': article['source']['name'],
                    'author': article.get('author', 'Unknown'),
                    'title': article['title'],
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article['url'],
                    'published_at': article['publishedAt'],
                    'collected_at': datetime.now().isoformat()
                }
                articles_data.append(article_data)
            
            print(f"    ✓ Found {len(articles)} articles")
        else:
            print(f"    ✗ API returned status: {response['status']}")
        
        # Rate limiting for free tier
        time.sleep(1)
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
    
    return articles_data

def collect_all_news_data(months_back=12):
    """
    Collect news data for all stocks
    
    Args:
        months_back (int): Number of months to look back
    """
    newsapi = create_news_client()
    if newsapi is None:
        return None
    
    base_path = create_data_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate date range (NewsAPI free tier allows max 1 month back)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=30)  # Free tier limitation
    
    from_date_str = from_date.strftime("%Y-%m-%d")
    to_date_str = to_date.strftime("%Y-%m-%d")
    
    print("\n" + "="*60)
    print("NEWS DATA COLLECTION STARTED")
    print("="*60)
    print(f"Date range: {from_date_str} to {to_date_str}")
    print(f"Note: Free tier limited to 100 articles/day\n")
    
    all_articles = []
    
    for ticker, query in NEWS_QUERIES.items():
        try:
            articles = collect_news_for_query(
                newsapi,
                query,
                ticker,
                from_date_str,
                to_date_str
            )
            
            if articles:
                all_articles.extend(articles)
                
                # Save individual ticker file
                ticker_df = pd.DataFrame(articles)
                filename = base_path / f'{ticker}_news_{timestamp}.csv'
                ticker_df.to_csv(filename, index=False)
                print(f"    Saved to {filename}\n")
        
        except Exception as e:
            print(f"✗ Error collecting news for {ticker}: {str(e)}\n")
            continue
    
    # Save combined file
    if all_articles:
        combined_df = pd.DataFrame(all_articles)
        
        # Remove duplicates based on URL
        combined_df = combined_df.drop_duplicates(subset=['url'])
        
        combined_filename = base_path / f'news_all_articles_{timestamp}.csv'
        combined_df.to_csv(combined_filename, index=False)
        
        print("="*60)
        print("COLLECTION SUMMARY")
        print("="*60)
        print(f"Total articles collected: {len(combined_df)}")
        print(f"Unique sources: {combined_df['source'].nunique()}")
        print(f"Tickers covered: {combined_df['ticker'].nunique()}")
        print(f"Combined data saved to: {combined_filename}")
        print("="*60 + "\n")
        
        # Display statistics
        print("\nArticles per Ticker:")
        print(combined_df['ticker'].value_counts())
        
        print("\nTop News Sources:")
        print(combined_df['source'].value_counts().head(10))
        
        return combined_df
    else:
        print("\n✗ No articles collected!")
        return None

def collect_daily_news():
    """Collect news from the last 24 hours (for daily updates)"""
    newsapi = create_news_client()
    if newsapi is None:
        return None
    
    base_path = create_data_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    to_date = datetime.now()
    from_date = to_date - timedelta(days=1)
    
    from_date_str = from_date.strftime("%Y-%m-%d")
    to_date_str = to_date.strftime("%Y-%m-%d")
    
    print(f"\nCollecting daily news for {to_date_str}...")
    
    daily_articles = []
    
    for ticker, query in NEWS_QUERIES.items():
        articles = collect_news_for_query(
            newsapi,
            query,
            ticker,
            from_date_str,
            to_date_str
        )
        
        if articles:
            daily_articles.extend(articles)
    
    if daily_articles:
        df = pd.DataFrame(daily_articles)
        df = df.drop_duplicates(subset=['url'])
        
        filename = base_path / f'news_daily_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"✓ Collected {len(df)} articles")
        return df
    
    return None

def collect_historical_news_batches():
    """
    Helper function to collect historical news in batches
    (To work around NewsAPI free tier limitations)
    """
    print("\n⚠️  Note: NewsAPI Free Tier Limitations")
    print("- Only articles from last 30 days")
    print("- 100 requests per day")
    print("\nFor historical data (2021-2024), consider:")
    print("1. Upgrade to NewsAPI paid plan")
    print("2. Use alternative sources (Google News RSS)")
    print("3. Use pre-collected datasets from Kaggle")
    print("\nFor research purposes, you can:")
    print("- Collect current data daily for live dataset")
    print("- Use archived news datasets from academic sources")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  News Data Collection - Financial News Sentiment        ║
    ║  Research: Hybrid Sentiment-Technical Transformers      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if API key is set
    if not NEWS_API_KEY:
        print("\n⚠️  SETUP REQUIRED ⚠️")
        print("\nBefore running this script, you need to:")
        print("\n1. Sign up for NewsAPI at: https://newsapi.org/")
        print("2. Get your free API key (100 requests/day)")
        print("3. Create a .env file in the project root with:")
        print("   NEWS_API_KEY=your_api_key_here")
        print("\nSee README.md for detailed instructions.")
    else:
        # Show limitations
        collect_historical_news_batches()
        
        print("\n" + "="*60)
        choice = input("\nCollect available news data? (y/n): ")
        
        if choice.lower() == 'y':
            # Collect news data
            data = collect_all_news_data()
            
            if data is not None:
                print("\n✓ News data collection completed successfully!")
                print("\nNext steps:")
                print("1. Review the collected data in data_raw/news/")
                print("2. Run sentiment_analyzer.py to analyze sentiment")
                print("3. Consider supplementing with Kaggle datasets for historical data")
            else:
                print("\n✗ Data collection failed.")
