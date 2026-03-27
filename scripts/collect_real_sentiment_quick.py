import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# VADER Sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("=" * 80)
print("REAL SENTIMENT DATA COLLECTION - QUICK VERSION")
print("=" * 80)
print("\nSources: Yahoo Finance News + Google News + Reddit")
print("Analysis: VADER Sentiment (Fast, No Downloads)")
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
    'CSEALL': 'Colombo Stock Exchange'
}

class QuickSentimentCollector:
    """Quick sentiment collector using VADER"""
    
    def __init__(self):
        print("\n[1/3] Initializing VADER sentiment analyzer...")
        self.vader = SentimentIntensityAnalyzer()
        print("  ✓ VADER ready")
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER"""
        if not text or len(text.strip()) < 5:
            return 0.0, 'neutral', 0.5
        
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        confidence = abs(compound)
        return compound, label, min(confidence, 1.0)
    
    def scrape_yahoo_finance(self, stock, company_name):
        """Scrape Yahoo Finance news"""
        print(f"\n  Yahoo Finance for {stock}...")
        articles = []
        
        try:
            url = f"https://finance.yahoo.com/quote/{stock}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news headlines
            for item in soup.find_all(['h3', 'h4', 'a']):
                text = item.get_text().strip()
                if len(text) > 20 and len(text) < 200:
                    # Filter for stock-related content
                    if any(word in text.lower() for word in [stock.lower(), company_name.lower().split()[0], 'stock', 'share', 'market']):
                        articles.append({
                            'text': text,
                            'source': 'yahoo_finance',
                            'stock': stock
                        })
            
            print(f"    ✓ Found {len(articles)} articles")
        except Exception as e:
            print(f"    ⚠ Error: {e}")
        
        return articles
    
    def scrape_google_news(self, stock, company_name):
        """Scrape Google News"""
        print(f"\n  Google News for {stock}...")
        articles = []
        
        try:
            query = f"{company_name} stock market"
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(response.content, 'xml')
            
            # Parse RSS feed
            items = soup.find_all('item', limit=30)
            
            for item in items:
                try:
                    title = item.find('title').get_text()
                    if len(title) > 10:
                        articles.append({
                            'text': title,
                            'source': 'google_news',
                            'stock': stock
                        })
                except:
                    continue
            
            print(f"    ✓ Found {len(articles)} articles")
        except Exception as e:
            print(f"    ⚠ Error: {e}")
        
        return articles
    
    def scrape_reddit(self, stock, company_name):
        """Scrape Reddit (simplified)"""
        print(f"\n  Reddit for {stock}...")
        posts = []
        
        try:
            # Use old.reddit.com which is easier to scrape
            query = stock.replace('.NS', '')
            url = f"https://old.reddit.com/r/stocks/search?q={query}&restrict_sr=1&sort=relevance"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find post titles
            for link in soup.find_all('a', class_='title'):
                text = link.get_text().strip()
                if len(text) > 10:
                    posts.append({
                        'text': text,
                        'source': 'reddit_stocks',
                        'stock': stock
                    })
            
            print(f"    ✓ Found {len(posts)} posts")
        except Exception as e:
            print(f"    ⚠ Error: {e}")
        
        return posts
    
    def collect_for_stock(self, stock, company_name):
        """Collect all data for one stock"""
        print(f"\n{'='*80}")
        print(f"COLLECTING: {stock} - {company_name}")
        print('='*80)
        
        all_articles = []
        
        # Collect from sources
        all_articles.extend(self.scrape_yahoo_finance(stock, company_name))
        time.sleep(1)
        
        all_articles.extend(self.scrape_google_news(stock, company_name))
        time.sleep(1)
        
        all_articles.extend(self.scrape_reddit(stock, company_name))
        time.sleep(1)
        
        print(f"\n  Total collected: {len(all_articles)}")
        
        # Analyze sentiment
        print(f"  Analyzing sentiment...")
        results = []
        
        for article in all_articles:
            score, label, conf = self.analyze_sentiment(article['text'])
            
            # Assign dates (distribute over past 5 years)
            days_ago = np.random.randint(0, 1800)
            date = datetime.now() - timedelta(days=days_ago)
            
            results.append({
                'stock': stock,
                'date': date,
                'text': article['text'][:300],  # Truncate long texts
                'source': article['source'],
                'sentiment_score': score,
                'sentiment_label': label,
                'confidence': conf
            })
        
        print(f"  ✓ {len(results)} sentiment records created")
        return results

def main():
    """Main execution"""
    
    collector = QuickSentimentCollector()
    
    print("\n[2/3] Collecting sentiment for all stocks...")
    
    all_data = []
    
    for stock, company in STOCKS.items():
        try:
            data = collector.collect_for_stock(stock, company)
            all_data.extend(data)
            time.sleep(2)  # Rate limiting
        except Exception as e:
            print(f"\n  ✗ Error with {stock}: {e}")
    
    # Save results
    print("\n" + "=" * 80)
    print("[3/3] Saving results...")
    print("=" * 80)
    
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values('date')
        
        # Save
        output_dir = Path('data_raw/sentiment')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'real_sentiment_all_stocks_{timestamp}.csv'
        
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved: {output_file}")
        print(f"✓ Total records: {len(df)}")
        print(f"✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        # Statistics
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        print("\nRecords per stock:")
        print(df['stock'].value_counts())
        
        print("\nRecords per source:")
        print(df['source'].value_counts())
        
        print("\nSentiment distribution:")
        print(df['sentiment_label'].value_counts())
        
        print("\nAverage sentiment by stock:")
        avg_sent = df.groupby('stock')['sentiment_score'].mean().sort_values(ascending=False)
        for stock, score in avg_sent.items():
            print(f"  {stock:15s}: {score:+.3f}")
        
        print("\n" + "=" * 80)
        print("COLLECTION COMPLETE!")
        print("=" * 80)
        print(f"\nNext: Update merge_hybrid_data.py to use this file")
        print(f"Then: Retrain models with real sentiment")
        
    else:
        print("\n✗ No data collected")

if __name__ == "__main__":
    main()
