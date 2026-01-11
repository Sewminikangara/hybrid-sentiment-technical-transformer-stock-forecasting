"""
FREE Sentiment Data Collection - No API costs!
Collects data from Reddit (FREE API) + Yahoo Finance (FREE)
"""

import praw
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Stock symbols
STOCKS = {
    'AAPL': 'Apple',
    'GOOGL': 'Google', 
    'TSLA': 'Tesla',
    'AMZN': 'Amazon',
    'MSFT': 'Microsoft',
    'RELIANCE.NS': 'Reliance',
    'TCS.NS': 'TCS',
    'INFY.NS': 'Infosys',
    'CSEALL': 'CSE'
}

def setup_reddit():
    """Initialize Reddit API connection"""
    try:
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=f"StockResearch/1.0 by {os.getenv('REDDIT_USERNAME')}",
            username=os.getenv('REDDIT_USERNAME'),
            password=os.getenv('REDDIT_PASSWORD')
        )
        print(f"✓ Reddit API connected as: {reddit.user.me()}")
        return reddit
    except Exception as e:
        print(f"✗ Reddit connection failed: {e}")
        return None

def collect_reddit_data(reddit, stock_symbol, stock_name, start_date, end_date):
    """Collect Reddit posts about a stock"""
    if not reddit:
        return pd.DataFrame()
    
    print(f"\n  Collecting Reddit data for {stock_symbol}...")
    
    # Subreddits to search
    subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
    
    # Search terms
    search_terms = [stock_symbol.replace('.NS', ''), stock_name]
    
    posts_data = []
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            for term in search_terms:
                print(f"    Searching r/{subreddit_name} for '{term}'...")
                
                # Search posts
                for post in subreddit.search(term, time_filter='all', limit=None):
                    post_time = post.created_utc
                    
                    # Filter by date range
                    if start_ts <= post_time <= end_ts:
                        posts_data.append({
                            'date': datetime.fromtimestamp(post_time).strftime('%Y-%m-%d'),
                            'created_utc': post_time,
                            'title': post.title,
                            'text': post.selftext,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'subreddit': subreddit_name,
                            'keyword': stock_symbol,
                            'url': f"https://reddit.com{post.permalink}"
                        })
                
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            print(f"    Error in r/{subreddit_name}: {e}")
            continue
    
    df = pd.DataFrame(posts_data)
    print(f"    Collected {len(df)} Reddit posts")
    return df

def collect_yahoo_news(stock_symbol, stock_name):
    """Collect news from Yahoo Finance (FREE)"""
    print(f"\n  Collecting Yahoo Finance news for {stock_symbol}...")
    
    try:
        ticker = yf.Ticker(stock_symbol)
        news = ticker.news
        
        news_data = []
        for article in news:
            news_data.append({
                'date': datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%d'),
                'created_utc': article.get('providerPublishTime', 0),
                'title': article.get('title', ''),
                'text': article.get('summary', ''),
                'publisher': article.get('publisher', ''),
                'link': article.get('link', ''),
                'keyword': stock_symbol
            })
        
        df = pd.DataFrame(news_data)
        print(f"    Collected {len(df)} Yahoo Finance news articles")
        return df
        
    except Exception as e:
        print(f"    Yahoo Finance error: {e}")
        return pd.DataFrame()

def main():
    print("=" * 80)
    print("FREE SENTIMENT DATA COLLECTION")
    print("=" * 80)
    print("\nSources:")
    print("  1. Reddit API (FREE - Historical access)")
    print("  2. Yahoo Finance (FREE - Recent news)")
    print()
    
    # Check credentials
    if not os.getenv('REDDIT_PASSWORD') or os.getenv('REDDIT_PASSWORD') == 'YOUR_PASSWORD_HERE':
        print("ERROR: Please update REDDIT_PASSWORD in .env file!")
        print("\nOpen .env file and replace 'YOUR_PASSWORD_HERE' with your Reddit password")
        return
    
    # Setup
    reddit = setup_reddit()
    if not reddit:
        print("\n⚠️  Reddit API not available. Will collect Yahoo Finance news only.")
        response = input("Continue with Yahoo Finance only? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Date range
    start_date = '2021-01-01'
    end_date = '2024-12-31'
    
    # Create output directory
    output_dir = Path('data_raw/sentiment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect for each stock
    all_reddit_data = []
    all_news_data = []
    
    for i, (symbol, name) in enumerate(STOCKS.items(), 1):
        print(f"\n[{i}/{len(STOCKS)}] Processing {symbol} ({name})")
        print("-" * 60)
        
        # Reddit data
        if reddit:
            reddit_df = collect_reddit_data(reddit, symbol, name, start_date, end_date)
            if not reddit_df.empty:
                all_reddit_data.append(reddit_df)
                
                # Save individual stock
                reddit_file = output_dir / f'reddit_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                reddit_df.to_csv(reddit_file, index=False)
                print(f"    ✓ Saved: {reddit_file}")
        
        # Yahoo Finance news
        news_df = collect_yahoo_news(symbol, name)
        if not news_df.empty:
            all_news_data.append(news_df)
            
            # Save individual stock
            news_file = output_dir / f'yahoo_news_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            news_df.to_csv(news_file, index=False)
            print(f"    ✓ Saved: {news_file}")
        
        time.sleep(2)  # Be nice to APIs
    
    # Combine all data
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if all_reddit_data:
        combined_reddit = pd.concat(all_reddit_data, ignore_index=True)
        reddit_combined_file = output_dir / f'reddit_all_stocks_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        combined_reddit.to_csv(reddit_combined_file, index=False)
        print(f"\n✓ Reddit Data: {len(combined_reddit)} total posts")
        print(f"  Saved to: {reddit_combined_file}")
    
    if all_news_data:
        combined_news = pd.concat(all_news_data, ignore_index=True)
        news_combined_file = output_dir / f'yahoo_news_all_stocks_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        combined_news.to_csv(news_combined_file, index=False)
        print(f"\n✓ Yahoo Finance News: {len(combined_news)} total articles")
        print(f"  Saved to: {news_combined_file}")
    
    print("\n" + "=" * 80)
    print("NEXT STEP: Run sentiment analysis on collected data")
    print("Command: python scripts/sentiment_analyzer.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
