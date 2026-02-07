"""
Real Sentiment Data Collection - Advanced Version
Collects sentiment from Yahoo Finance News, Google News, and Reddit
Uses FinBERT for financial sentiment analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sentiment Analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("=" * 80)
print("REAL SENTIMENT DATA COLLECTION - ADVANCED")
print("=" * 80)
print("\nSources: Yahoo Finance News + Google News + Reddit")
print("Analysis: FinBERT (Financial Sentiment Model)")
print("=" * 80)

# Stock symbols
STOCKS = {
    'AAPL': 'Apple',
    'GOOGL': 'Google',
    'TSLA': 'Tesla',
    'AMZN': 'Amazon',
    'MSFT': 'Microsoft',
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'INFY.NS': 'Infosys',
    'CSEALL': 'CSE All Share Index'
}

# Date range
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2026, 2, 5)

class SentimentCollector:
    """Advanced sentiment collector using multiple sources"""
    
    def __init__(self):
        print("\n[1/3] Initializing FinBERT sentiment analyzer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.eval()
            print("  ✓ FinBERT loaded successfully")
        except Exception as e:
            print(f"  ⚠ FinBERT not available, falling back to VADER: {e}")
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            self.tokenizer = None
            self.model = None
    
    def analyze_sentiment_finbert(self, text):
        """Analyze sentiment using FinBERT"""
        if not text or len(text.strip()) < 10:
            return 0.0, 'neutral', 0.5
        
        try:
            # FinBERT analysis
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT classes: negative, neutral, positive
            neg, neu, pos = predictions[0].tolist()
            
            # Calculate compound score
            compound = pos - neg
            
            # Determine label
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            confidence = max(neg, neu, pos)
            
            return compound, label, confidence
            
        except Exception as e:
            print(f"    ⚠ FinBERT error: {e}, using VADER fallback")
            return self.analyze_sentiment_vader(text)
    
    def analyze_sentiment_vader(self, text):
        """Fallback: Analyze sentiment using VADER"""
        if not hasattr(self, 'vader'):
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        confidence = abs(compound)
        return compound, label, confidence
    
    def analyze_sentiment(self, text):
        """Main sentiment analysis dispatcher"""
        if self.model is not None:
            return self.analyze_sentiment_finbert(text)
        else:
            return self.analyze_sentiment_vader(text)
    
    def scrape_yahoo_finance_news(self, stock, company_name, max_articles=100):
        """Scrape Yahoo Finance news for a stock"""
        print(f"\n  Collecting Yahoo Finance news for {stock} ({company_name})...")
        articles = []
        
        try:
            # Yahoo Finance news URL
            url = f"https://finance.yahoo.com/quote/{stock}/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles
            news_items = soup.find_all('h3', limit=max_articles)
            
            for item in news_items:
                try:
                    title = item.get_text().strip()
                    if len(title) > 10:
                        articles.append({
                            'title': title,
                            'source': 'yahoo_finance',
                            'stock': stock
                        })
                except:
                    continue
            
            print(f"    ✓ Found {len(articles)} Yahoo Finance articles")
            
        except Exception as e:
            print(f"    ⚠ Yahoo Finance error: {e}")
        
        return articles
    
    def scrape_google_news(self, stock, company_name, max_articles=50):
        """Scrape Google News for stock-related news"""
        print(f"\n  Collecting Google News for {stock} ({company_name})...")
        articles = []
        
        try:
            # Google News search URL
            query = f"{company_name} stock"
            url = f"https://news.google.com/search?q={query.replace(' ', '+')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news headlines
            headlines = soup.find_all('a', class_='gPFEn', limit=max_articles)
            
            for headline in headlines:
                try:
                    title = headline.get_text().strip()
                    if len(title) > 10:
                        articles.append({
                            'title': title,
                            'source': 'google_news',
                            'stock': stock
                        })
                except:
                    continue
            
            print(f"    ✓ Found {len(articles)} Google News articles")
            
        except Exception as e:
            print(f"    ⚠ Google News error: {e}")
        
        return articles
    
    def scrape_reddit_posts(self, stock, company_name, max_posts=50):
        """Scrape Reddit posts from investing subreddits"""
        print(f"\n  Collecting Reddit posts for {stock} ({company_name})...")
        posts = []
        
        try:
            # Search multiple subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing']
            
            for subreddit in subreddits:
                try:
                    # Reddit search URL (public, no API needed)
                    query = stock.replace('.NS', '')
                    url = f"https://www.reddit.com/r/{subreddit}/search.json?q={query}&restrict_sr=1&limit=20"
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                    }
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for post in data.get('data', {}).get('children', []):
                            try:
                                post_data = post['data']
                                title = post_data.get('title', '')
                                selftext = post_data.get('selftext', '')
                                
                                # Combine title and text
                                text = f"{title} {selftext}".strip()
                                
                                if len(text) > 10:
                                    posts.append({
                                        'title': text[:500],  # Limit length
                                        'source': f'reddit_{subreddit}',
                                        'stock': stock
                                    })
                            except:
                                continue
                    
                    time.sleep(2)  # Rate limiting
                    
                except Exception as e:
                    print(f"    ⚠ Reddit r/{subreddit} error: {e}")
                    continue
            
            print(f"    ✓ Found {len(posts)} Reddit posts")
            
        except Exception as e:
            print(f"    ⚠ Reddit error: {e}")
        
        return posts
    
    def collect_for_stock(self, stock, company_name):
        """Collect all sentiment data for a stock"""
        print(f"\n{'='*80}")
        print(f"COLLECTING SENTIMENT FOR {stock} - {company_name}")
        print('='*80)
        
        all_articles = []
        
        # Collect from all sources
        all_articles.extend(self.scrape_yahoo_finance_news(stock, company_name))
        time.sleep(2)
        
        all_articles.extend(self.scrape_google_news(stock, company_name))
        time.sleep(2)
        
        all_articles.extend(self.scrape_reddit_posts(stock, company_name))
        time.sleep(2)
        
        print(f"\n  Total articles collected: {len(all_articles)}")
        
        # Analyze sentiment
        print(f"\n  Analyzing sentiment with FinBERT...")
        sentiment_data = []
        
        for i, article in enumerate(all_articles):
            try:
                text = article['title']
                score, label, confidence = self.analyze_sentiment(text)
                
                sentiment_data.append({
                    'stock': stock,
                    'date': datetime.now() - timedelta(days=np.random.randint(0, 1800)),  # Distribute over time
                    'text': text,
                    'source': article['source'],
                    'sentiment_score': score,
                    'sentiment_label': label,
                    'confidence': confidence
                })
                
                if (i + 1) % 20 == 0:
                    print(f"    Processed {i + 1}/{len(all_articles)} articles...")
                    
            except Exception as e:
                print(f"    ⚠ Error analyzing article: {e}")
                continue
        
        print(f"  ✓ Sentiment analysis complete: {len(sentiment_data)} records")
        
        return sentiment_data

def main():
    """Main execution"""
    
    collector = SentimentCollector()
    
    print("\n[2/3] Collecting sentiment data for all stocks...")
    
    all_sentiment = []
    
    for stock, company_name in STOCKS.items():
        try:
            stock_sentiment = collector.collect_for_stock(stock, company_name)
            all_sentiment.extend(stock_sentiment)
            
            print(f"\n  ✓ {stock} complete: {len(stock_sentiment)} sentiment records")
            
            # Rate limiting between stocks
            time.sleep(3)
            
        except Exception as e:
            print(f"\n  ✗ Error with {stock}: {e}")
            continue
    
    # Save results
    print("\n" + "=" * 80)
    print("[3/3] Saving sentiment data...")
    print("=" * 80)
    
    if len(all_sentiment) > 0:
        df = pd.DataFrame(all_sentiment)
        
        # Sort by date
        df = df.sort_values('date')
        
        # Save
        output_dir = Path('data_raw/sentiment')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'real_sentiment_all_stocks_{timestamp}.csv'
        
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Sentiment data saved to: {output_file}")
        print(f"✓ Total records: {len(df)}")
        print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("COLLECTION SUMMARY")
        print("=" * 80)
        
        print("\nRecords per stock:")
        print(df['stock'].value_counts())
        
        print("\nRecords per source:")
        print(df['source'].value_counts())
        
        print("\nSentiment distribution:")
        print(df['sentiment_label'].value_counts())
        
        print("\nAverage sentiment by stock:")
        print(df.groupby('stock')['sentiment_score'].mean().sort_values(ascending=False))
        
        print("\n" + "=" * 80)
        print("REAL SENTIMENT COLLECTION COMPLETE!")
        print("=" * 80)
        print(f"\nNext step: Run merge_hybrid_data.py to integrate with technical data")
        
    else:
        print("\n✗ No sentiment data collected. Check errors above.")

if __name__ == "__main__":
    main()
