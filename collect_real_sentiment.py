"""
Collect Real Sentiment Data from Free Sources
Yahoo Finance News + Google News + Reddit (web scraping)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Initialize sentiment analyzers
vader = SentimentIntensityAnalyzer()

# Stocks to collect
STOCKS = {
    'AAPL': 'Apple',
    'GOOGL': 'Google',
    'TSLA': 'Tesla',
    'AMZN': 'Amazon',
    'MSFT': 'Microsoft',
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'TCS',
    'INFY.NS': 'Infosys',
    'CSEALL': 'Sri Lanka Stock Market'
}

def get_yahoo_finance_news(symbol, company_name):
    """Scrape Yahoo Finance news headlines"""
    print(f"  Collecting Yahoo Finance news for {symbol}...")
    
    news_data = []
    
    try:
        # Yahoo Finance news URL
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news headlines
            headlines = soup.find_all('h3', limit=20)
            
            for headline in headlines:
                text = headline.get_text(strip=True)
                if text and len(text) > 10:
                    # Analyze sentiment
                    vader_score = vader.polarity_scores(text)
                    
                    news_data.append({
                        'stock': symbol,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'yahoo_finance',
                        'text': text,
                        'sentiment_score': vader_score['compound'],
                        'positive': vader_score['pos'],
                        'negative': vader_score['neg'],
                        'neutral': vader_score['neu']
                    })
            
            print(f"    ✓ Found {len(news_data)} Yahoo Finance headlines")
        else:
            print(f"    ⚠ Yahoo Finance failed (status {response.status_code})")
            
    except Exception as e:
        print(f"    ✗ Yahoo Finance error: {e}")
    
    return news_data

def get_google_news(symbol, company_name):
    """Get Google News headlines via RSS"""
    print(f"  Collecting Google News for {company_name}...")
    
    news_data = []
    
    try:
        # Google News RSS feed
        query = company_name.replace(' ', '+')
        url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item', limit=15)
            
            for item in items:
                title = item.title.text if item.title else ''
                pub_date = item.pubDate.text if item.pubDate else datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
                
                if title:
                    # Parse date
                    try:
                        date_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                    except:
                        date_obj = datetime.now()
                    
                    # Analyze sentiment
                    vader_score = vader.polarity_scores(title)
                    
                    news_data.append({
                        'stock': symbol,
                        'date': date_obj.strftime('%Y-%m-%d'),
                        'source': 'google_news',
                        'text': title,
                        'sentiment_score': vader_score['compound'],
                        'positive': vader_score['pos'],
                        'negative': vader_score['neg'],
                        'neutral': vader_score['neu']
                    })
            
            print(f"    ✓ Found {len(news_data)} Google News articles")
        else:
            print(f"    ⚠ Google News failed (status {response.status_code})")
            
    except Exception as e:
        print(f"    ✗ Google News error: {e}")
    
    return news_data

def get_reddit_posts(symbol, company_name):
    """Scrape Reddit posts (public, no API)"""
    print(f"  Collecting Reddit posts for {symbol}...")
    
    reddit_data = []
    
    try:
        # Reddit search URL (public)
        search_term = symbol.replace('.NS', '')
        url = f"https://www.reddit.com/search/?q={search_term}%20stock&sort=new&t=month"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Extract post titles from HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find post titles (Reddit's HTML structure)
            titles = soup.find_all('h3', limit=10)
            
            for title in titles:
                text = title.get_text(strip=True)
                if text and len(text) > 10:
                    # Analyze sentiment
                    vader_score = vader.polarity_scores(text)
                    
                    reddit_data.append({
                        'stock': symbol,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'reddit',
                        'text': text,
                        'sentiment_score': vader_score['compound'],
                        'positive': vader_score['pos'],
                        'negative': vader_score['neg'],
                        'neutral': vader_score['neu']
                    })
            
            print(f"    ✓ Found {len(reddit_data)} Reddit posts")
        else:
            print(f"    ⚠ Reddit failed (status {response.status_code})")
            
    except Exception as e:
        print(f"    ✗ Reddit error: {e}")
    
    return reddit_data

def get_finviz_news(symbol):
    """Scrape Finviz news (free financial news)"""
    print(f"  Collecting Finviz news for {symbol}...")
    
    news_data = []
    
    try:
        # Remove .NS suffix for Indian stocks
        ticker = symbol.replace('.NS', '')
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news table
            news_table = soup.find('table', {'id': 'news-table'})
            
            if news_table:
                rows = news_table.find_all('tr', limit=15)
                
                for row in rows:
                    link = row.find('a')
                    if link:
                        headline = link.get_text(strip=True)
                        
                        # Analyze sentiment
                        vader_score = vader.polarity_scores(headline)
                        
                        news_data.append({
                            'stock': symbol,
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'finviz',
                            'text': headline,
                            'sentiment_score': vader_score['compound'],
                            'positive': vader_score['pos'],
                            'negative': vader_score['neg'],
                            'neutral': vader_score['neu']
                        })
                
                print(f"    ✓ Found {len(news_data)} Finviz headlines")
            else:
                print(f"    ⚠ No Finviz news table found")
        else:
            print(f"    ⚠ Finviz failed (status {response.status_code})")
            
    except Exception as e:
        print(f"    ✗ Finviz error: {e}")
    
    return news_data

def collect_all_sentiment():
    """Collect sentiment from all sources"""
    print("=" * 80)
    print("COLLECTING REAL SENTIMENT DATA FROM MULTIPLE SOURCES")
    print("=" * 80)
    print(f"\nSources: Yahoo Finance + Google News + Reddit + Finviz")
    print(f"Stocks: {len(STOCKS)}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    all_sentiment = []
    
    for i, (symbol, company) in enumerate(STOCKS.items(), 1):
        print(f"\n[{i}/{len(STOCKS)}] {symbol} - {company}")
        print("-" * 60)
        
        # Collect from all sources
        yahoo_news = get_yahoo_finance_news(symbol, company)
        all_sentiment.extend(yahoo_news)
        time.sleep(1)  # Be polite to servers
        
        google_news = get_google_news(symbol, company)
        all_sentiment.extend(google_news)
        time.sleep(1)
        
        reddit_posts = get_reddit_posts(symbol, company)
        all_sentiment.extend(reddit_posts)
        time.sleep(1)
        
        # Finviz (mainly for US stocks)
        if '.NS' not in symbol and symbol != 'CSEALL':
            finviz_news = get_finviz_news(symbol)
            all_sentiment.extend(finviz_news)
            time.sleep(1)
        
        total = len(yahoo_news) + len(google_news) + len(reddit_posts)
        print(f"\n  Total for {symbol}: {total} items")
    
    # Create DataFrame
    df = pd.DataFrame(all_sentiment)
    
    if len(df) > 0:
        # Add sentiment label
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        
        # Save
        output_dir = Path('data_raw/sentiment')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'real_sentiment_{timestamp}.csv'
        
        df.to_csv(output_file, index=False)
        
        # Summary
        print("\n" + "=" * 80)
        print("COLLECTION COMPLETE")
        print("=" * 80)
        print(f"\n✓ Total items collected: {len(df)}")
        print(f"✓ Saved to: {output_file}")
        
        print("\n📊 Distribution by Source:")
        print(df['source'].value_counts().to_string())
        
        print("\n📊 Distribution by Stock:")
        print(df['stock'].value_counts().to_string())
        
        print("\n📊 Sentiment Distribution:")
        print(df['sentiment_label'].value_counts().to_string())
        
        print("\n📊 Average Sentiment by Stock:")
        avg_sentiment = df.groupby('stock')['sentiment_score'].mean().sort_values(ascending=False)
        for stock, score in avg_sentiment.items():
            sentiment_type = 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
            print(f"  {stock}: {score:.3f} ({sentiment_type})")
        
        print("\n" + "=" * 80)
        
        return output_file
    else:
        print("\n⚠ No sentiment data collected. Check your internet connection.")
        return None

if __name__ == "__main__":
    try:
        collect_all_sentiment()
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
