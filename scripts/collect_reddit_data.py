

import praw
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Reddit API credentials (from .env file)
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockResearchBot/1.0')

# Subreddits to monitor
SUBREDDITS = [
    'stocks',
    'wallstreetbets',
    'investing',
    'StockMarket',
    'IndianStockMarket',
]

# Stock tickers to search for
STOCK_KEYWORDS = {
    'US': ['AAPL', 'Apple', 'GOOGL', 'Google', 'TSLA', 'Tesla', 'AMZN', 'Amazon', 'MSFT', 'Microsoft'],
    'India': ['Reliance', 'TCS', 'Infosys', 'INFY'],
    'Sri_Lanka': ['JKH', 'Commercial Bank', 'Dialog']
}

def create_reddit_client():
    """Initialize Reddit API client"""
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET]):
        print("⚠️  Reddit API credentials not found!")
        print("\nPlease set up your .env file with:")
        print("REDDIT_CLIENT_ID=your_client_id")
        print("REDDIT_CLIENT_SECRET=your_client_secret")
        print("REDDIT_USER_AGENT=your_user_agent")
        print("\nGet credentials from: https://www.reddit.com/prefs/apps")
        return None
    
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        # Test the connection
        reddit.user.me()
        print("✓ Reddit API connection successful")
        return reddit
    except Exception as e:
        print(f"✗ Reddit API connection failed: {str(e)}")
        return None

def create_data_directory():
    """Create directory for reddit data"""
    base_path = Path(__file__).parent.parent / 'data_raw' / 'reddit'
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

def collect_subreddit_posts(reddit, subreddit_name, keywords, limit=1000, time_filter='year'):
    """
    Collect posts from a subreddit containing specific keywords
    
    Args:
        reddit: Reddit API client
        subreddit_name (str): Name of subreddit
        keywords (list): List of keywords to search for
        limit (int): Maximum number of posts to collect
        time_filter (str): Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
    
    Returns:
        pd.DataFrame: Collected posts data
    """
    posts_data = []
    subreddit = reddit.subreddit(subreddit_name)
    
    print(f"\n--- Collecting from r/{subreddit_name} ---")
    
    for keyword in keywords:
        try:
            print(f"  Searching for: {keyword}...")
            
            # Search for posts containing the keyword
            for submission in subreddit.search(keyword, time_filter=time_filter, limit=limit):
                post_data = {
                    'subreddit': subreddit_name,
                    'keyword': keyword,
                    'post_id': submission.id,
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'score': submission.score,
                    'upvote_ratio': submission.upvote_ratio,
                    'num_comments': submission.num_comments,
                    'created_utc': datetime.fromtimestamp(submission.created_utc),
                    'author': str(submission.author),
                    'url': submission.url,
                    'permalink': f"https://reddit.com{submission.permalink}"
                }
                posts_data.append(post_data)
            
            print(f"    ✓ Found {len([p for p in posts_data if p['keyword'] == keyword])} posts")
            
            # Rate limiting - be nice to Reddit's servers
            time.sleep(2)
            
        except Exception as e:
            print(f"    ✗ Error searching for {keyword}: {str(e)}")
            continue
    
    return pd.DataFrame(posts_data)

def collect_all_reddit_data():
    """Collect data from all subreddits for all stock keywords"""
    
    reddit = create_reddit_client()
    if reddit is None:
        return None
    
    base_path = create_data_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_posts = []
    
    print("\n" + "="*60)
    print("REDDIT DATA COLLECTION STARTED")
    print("="*60)
    print(f"Subreddits: {', '.join(SUBREDDITS)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Flatten all keywords
    all_keywords = []
    for market_keywords in STOCK_KEYWORDS.values():
        all_keywords.extend(market_keywords)
    
    for subreddit_name in SUBREDDITS:
        try:
            subreddit_posts = collect_subreddit_posts(
                reddit, 
                subreddit_name, 
                all_keywords,
                limit=200,  # Adjust based on your needs
                time_filter='year'
            )
            
            if not subreddit_posts.empty:
                all_posts.append(subreddit_posts)
                
                # Save individual subreddit file
                filename = base_path / f'{subreddit_name}_{timestamp}.csv'
                subreddit_posts.to_csv(filename, index=False)
                print(f"✓ Saved {len(subreddit_posts)} posts from r/{subreddit_name}")
        
        except Exception as e:
            print(f"✗ Error collecting from r/{subreddit_name}: {str(e)}")
            continue
    
    # Combine all posts
    if all_posts:
        combined_df = pd.concat(all_posts, ignore_index=True)
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['post_id'])
        
        # Save combined file
        combined_filename = base_path / f'reddit_all_posts_{timestamp}.csv'
        combined_df.to_csv(combined_filename, index=False)
        
        print("\n" + "="*60)
        print("COLLECTION SUMMARY")
        print("="*60)
        print(f"Total posts collected: {len(combined_df)}")
        print(f"Subreddits covered: {combined_df['subreddit'].nunique()}")
        print(f"Date range: {combined_df['created_utc'].min()} to {combined_df['created_utc'].max()}")
        print(f"Combined data saved to: {combined_filename}")
        print("="*60 + "\n")
        
        # Display statistics
        print("\nPosts per Subreddit:")
        print(combined_df['subreddit'].value_counts())
        
        print("\nPosts per Keyword (Top 10):")
        print(combined_df['keyword'].value_counts().head(10))
        
        return combined_df
    else:
        print("\n✗ No data collected!")
        return None

def collect_recent_posts(hours=24):
    """
    Collect recent posts from the last N hours (for daily updates)
    
    Args:
        hours (int): Number of hours to look back
    """
    reddit = create_reddit_client()
    if reddit is None:
        return None
    
    base_path = create_data_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nCollecting posts from the last {hours} hours...")
    
    all_keywords = []
    for market_keywords in STOCK_KEYWORDS.values():
        all_keywords.extend(market_keywords)
    
    recent_posts = []
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    for subreddit_name in SUBREDDITS:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Get recent posts
        for submission in subreddit.new(limit=100):
            post_time = datetime.fromtimestamp(submission.created_utc)
            
            if post_time < cutoff_time:
                continue
            
            # Check if post contains any of our keywords
            text = f"{submission.title} {submission.selftext}".lower()
            matched_keywords = [kw for kw in all_keywords if kw.lower() in text]
            
            if matched_keywords:
                post_data = {
                    'subreddit': subreddit_name,
                    'keywords': ', '.join(matched_keywords),
                    'post_id': submission.id,
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'created_utc': post_time,
                    'permalink': f"https://reddit.com{submission.permalink}"
                }
                recent_posts.append(post_data)
    
    if recent_posts:
        df = pd.DataFrame(recent_posts)
        filename = base_path / f'reddit_recent_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"✓ Collected {len(df)} recent posts")
        return df
    
    return None

if __name__ == "__main__":
    print(""" done
    """)
    
    # Check if credentials are set
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        print("\n⚠️  SETUP REQUIRED ⚠️")
        print("\nBefore running this script, you need to:")
        print("\n1. Create a Reddit app at: https://www.reddit.com/prefs/apps")
        print("2. Choose 'script' as the app type")
        print("3. Create a .env file in the project root with:")
        print("   REDDIT_CLIENT_ID=your_client_id_here")
        print("   REDDIT_CLIENT_SECRET=your_client_secret_here")
        print("   REDDIT_USER_AGENT=StockResearchBot/1.0")
        print("\nSee README.md for detailed instructions.")
    else:
        # Collect historical data
        data = collect_all_reddit_data()
        
        if data is not None:
            print("\n✓ Reddit data collection completed successfully!")
            print("\nNext steps:")
            print("1. Review the collected data in data_raw/reddit/")
            print("2. Run sentiment_analyzer.py to analyze sentiment")
            print("3. Collect news data using collect_news_data.py")
        else:
            print("\n✗ Data collection failed.")
