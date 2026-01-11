"""
Collect Historical Sentiment Data using web scraping
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from pathlib import Path
import random

# Stock symbols and search terms
STOCKS = {
    'AAPL': ['Apple', 'AAPL', 'iPhone', 'Tim Cook'],
    'GOOGL': ['Google', 'GOOGL', 'Alphabet', 'Sundar Pichai'],
    'TSLA': ['Tesla', 'TSLA', 'Elon Musk', 'electric vehicle'],
    'AMZN': ['Amazon', 'AMZN', 'Jeff Bezos', 'AWS'],
    'MSFT': ['Microsoft', 'MSFT', 'Azure', 'Satya Nadella'],
    'RELIANCE.NS': ['Reliance Industries', 'Mukesh Ambani', 'Jio'],
    'TCS.NS': ['TCS', 'Tata Consultancy'],
    'INFY.NS': ['Infosys', 'Infy'],
    'CSEALL': ['Sri Lanka stock market', 'Colombo Stock Exchange', 'CSE']
}

def scrape_google_news(query, start_date, end_date):
    """
    Scrape Google News for historical articles
    """
    articles = []
    
    # Format dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Google News search URL
    search_query = query.replace(' ', '+')
    
    # Search by year to get historical data
    current_year = start.year
    while current_year <= end.year:
        try:
            url = f"https://news.google.com/search?q={search_query}+when:1y&hl=en-US&gl=US&ceid=US:en"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article elements
                article_elements = soup.find_all('article')
                
                for article in article_elements[:20]:  # Limit per search
                    try:
                        title_elem = article.find('h3') or article.find('h4')
                        title = title_elem.get_text() if title_elem else ''
                        
                        link_elem = article.find('a')
                        link = link_elem.get('href', '') if link_elem else ''
                        if link.startswith('./'):
                            link = 'https://news.google.com' + link[1:]
                        
                        time_elem = article.find('time')
                        pub_date = time_elem.get('datetime', '') if time_elem else str(current_year)
                        
                        if title:
                            articles.append({
                                'title': title,
                                'link': link,
                                'published': pub_date,
                                'source': 'Google News',
                                'search_query': query
                            })
                    except:
                        continue
            
            time.sleep(random.uniform(2, 4))  # Be polite
            current_year += 1
            
        except Exception as e:
            print(f"      Error scraping {query} for {current_year}: {e}")
            continue
    
    return articles

def generate_synthetic_sentiment(stock_symbol, start_date, end_date, num_samples=500):
    """
    Generate synthetic sentiment data based on actual price movements
    """
    print(f"    Generating synthetic sentiment based on market patterns...")
    
    # Load actual stock prices
    try:
        price_file = 'data_raw/stock_prices/all_stocks_with_cse_20260104_232250.csv'
        df = pd.read_csv(price_file)
        stock_data = df[df['Ticker'] == stock_symbol].copy()
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data = stock_data.sort_values('Date')
        
        # Calculate returns and volatility
        stock_data['Returns'] = stock_data['Close'].pct_change()
        stock_data['Volatility'] = stock_data['Returns'].rolling(window=5).std()
        
        # Generate sentiment scores based on price action
        sentiment_data = []
        for _, row in stock_data.iterrows():
            if pd.notna(row['Returns']):
                # Positive returns → positive sentiment (with noise)
                base_sentiment = row['Returns'] * 10  # Scale to [-1, 1]
                noise = random.gauss(0, 0.2)
                
                sentiment_score = max(-1, min(1, base_sentiment + noise))
                
                # Classify
                if sentiment_score > 0.2:
                    label = 'positive'
                elif sentiment_score < -0.2:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                sentiment_data.append({
                    'date': row['Date'].strftime('%Y-%m-%d'),
                    'sentiment_score': sentiment_score,
                    'sentiment_label': label,
                    'confidence': abs(sentiment_score),
                    'source': 'market_derived'
                })
        
        print(f"    Generated {len(sentiment_data)} sentiment datapoints")
        return pd.DataFrame(sentiment_data)
        
    except Exception as e:
        print(f"    Error generating synthetic sentiment: {e}")
        return pd.DataFrame()

def main():
    print("=" * 80)
    print("HISTORICAL SENTIMENT DATA COLLECTION")
    print("=" * 80)
    print("\nApproaches:")
    print("  1. Web scraping Google News (recent data)")
    print("  2. Synthetic sentiment from price movements")
    print()
    
    choice = input("Choose approach:\n1. Scrape Google News (slower, recent only)\n2. Generate synthetic sentiment (faster, complete historical)\n3. Both\n\nChoice (1/2/3): ")
    
    output_dir = Path('data_raw/sentiment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = '2021-01-01'
    end_date = '2024-12-31'
    
    all_sentiment_data = []
    
    for i, (symbol, search_terms) in enumerate(STOCKS.items(), 1):
        print(f"\n[{i}/{len(STOCKS)}] Processing {symbol}")
        print("-" * 60)
        
        stock_sentiment = pd.DataFrame()
        
        if choice in ['1', '3']:
            # Web scraping approach
            print(f"  Scraping news for {symbol}...")
            all_articles = []
            
            for term in search_terms[:2]:  # Limit terms to avoid being blocked
                print(f"    Searching: {term}")
                articles = scrape_google_news(term, start_date, end_date)
                all_articles.extend(articles)
                time.sleep(random.uniform(3, 5))
            
            if all_articles:
                news_df = pd.DataFrame(all_articles)
                news_df['stock'] = symbol
                print(f"    Collected {len(news_df)} articles")
                
                # Save
                news_file = output_dir / f'scraped_news_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                news_df.to_csv(news_file, index=False)
                print(f"    ✓ Saved: {news_file}")
        
        if choice in ['2', '3']:
            # Synthetic sentiment approach
            stock_sentiment = generate_synthetic_sentiment(symbol, start_date, end_date)
            
            if not stock_sentiment.empty:
                stock_sentiment['stock'] = symbol
                all_sentiment_data.append(stock_sentiment)
                
                # Save individual
                sent_file = output_dir / f'sentiment_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                stock_sentiment.to_csv(sent_file, index=False)
                print(f"    ✓ Saved: {sent_file}")
    
    # Combine all sentiment data
    if all_sentiment_data:
        print("\n" + "=" * 80)
        print("COMBINING ALL SENTIMENT DATA")
        print("=" * 80)
        
        combined = pd.concat(all_sentiment_data, ignore_index=True)
        combined_file = output_dir / f'sentiment_all_stocks_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        combined.to_csv(combined_file, index=False)
        
        print(f"\n✓ Total sentiment records: {len(combined)}")
        print(f"  Saved to: {combined_file}")
        print(f"\n✓ Coverage by stock:")
        print(combined.groupby('stock').size())
        
        print("\n" + "=" * 80)
        print("NEXT STEP: Merge with technical indicators")
        print("This sentiment data is ready for hybrid model training!")
        print("=" * 80)

if __name__ == "__main__":
    main()
