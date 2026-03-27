import re
import time
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("tradex.news_ingestion")


def _parse_rss_feed(feed_url: str, timeout: int = 10) -> List[Dict]:
    """
    Parse RSS/Atom feed and return list of items.
    Uses feedparser if available, falls back to basic XML parsing.
    """
    items = []

    try:
        import feedparser
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:20]:  # Limit to 20 most recent
            published = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6])
            else:
                published = datetime.utcnow()

            items.append({
                "title": entry.get("title", ""),
                "content": entry.get("summary", entry.get("description", "")),
                "url": entry.get("link", ""),
                "published_at": published,
            })

        return items

    except ImportError:
        logger.warning("feedparser not installed. Attempting basic XML parsing.")

    # Fallback: basic urllib + XML parsing
    try:
        import urllib.request
        import xml.etree.ElementTree as ET

        req = urllib.request.Request(feed_url, headers={
            'User-Agent': 'TradeXY/1.0 (Research Project)'
        })

        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read().decode('utf-8', errors='ignore')

        root = ET.fromstring(content)

        # Handle RSS 2.0
        for item in root.findall('.//item')[:20]:
            title = item.findtext('title', '')
            desc = item.findtext('description', '')
            link = item.findtext('link', '')
            pub_date = item.findtext('pubDate', '')

            published = datetime.utcnow()
            if pub_date:
                try:
                    from email.utils import parsedate_to_datetime
                    published = parsedate_to_datetime(pub_date).replace(tzinfo=None)
                except Exception:
                    pass

            items.append({
                "title": title,
                "content": _strip_html(desc),
                "url": link,
                "published_at": published,
            })

        # Handle Atom feeds
        if not items:
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            for entry in root.findall('.//atom:entry', ns)[:20]:
                title = entry.findtext('atom:title', '', ns)
                content = entry.findtext('atom:summary', '', ns)
                link_el = entry.find('atom:link', ns)
                link = link_el.get('href', '') if link_el is not None else ''

                items.append({
                    "title": title,
                    "content": _strip_html(content),
                    "url": link,
                    "published_at": datetime.utcnow(),
                })

    except Exception as e:
        logger.error(f"Failed to parse feed {feed_url}: {e}")

    return items

def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text).strip()


def _fetch_reddit_posts(subreddit: str, limit: int = 10) -> List[Dict]:
    """
    Fetch recent posts from a Reddit subreddit using the public JSON API.
    No API key required - uses .json endpoint.
    """
    items = []

    try:
        import urllib.request
        import json

        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
        req = urllib.request.Request(url, headers={
            'User-Agent': 'TradeXY/1.0 (Research Project by student)'
        })

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))

        for post in data.get('data', {}).get('children', []):
            post_data = post.get('data', {})

            title = post_data.get('title', '')
            selftext = post_data.get('selftext', '')[:500]
            permalink = post_data.get('permalink', '')
            created = post_data.get('created_utc', 0)
            score = post_data.get('score', 0)

            items.append({
                "title": title,
                "content": selftext or title,
                "url": f"https://reddit.com{permalink}",
                "published_at": datetime.utcfromtimestamp(created) if created else datetime.utcnow(),
                "score": score,
            })

    except Exception as e:
        logger.error(f"Failed to fetch Reddit r/{subreddit}: {e}")

    return items


class NewsIngestionManager:
    """
    Manages the ingestion of news from multiple sources.
    Feeds items into the NewsRiskFilter for scoring and risk state updates.
    """

    # Default subreddits
    DEFAULT_SUBREDDITS = [
        "cryptocurrency", "bitcoin", "ethereum", "forex",
        "stocks", "investing", "wallstreetbets"
    ]

    def __init__(self, rss_feeds: Optional[List[Dict]] = None,
                 subreddits: Optional[List[str]] = None):
        from tradex.config import RSS_FEEDS

        self.rss_feeds = rss_feeds or RSS_FEEDS
        self.subreddits = subreddits or self.DEFAULT_SUBREDDITS

        self._last_fetch: Dict[str, datetime] = {}
        self._fetch_interval = timedelta(minutes=15)

        # Stats
        self.total_fetched = 0
        self.total_ingested = 0
        self.total_duplicates = 0
        self.errors = []

    def fetch_all_rss(self) -> List[Dict]:
        """Fetch items from all configured RSS feeds."""
        all_items = []

        for feed_config in self.rss_feeds:
            url = feed_config["url"]
            source = feed_config["source"]

            # Rate limiting
            last = self._last_fetch.get(url)
            if last and (datetime.utcnow() - last) < self._fetch_interval:
                continue

            logger.info(f"Fetching RSS: {source}")
            items = _parse_rss_feed(url)

            for item in items:
                item["source"] = source
                item["feed_category"] = feed_config.get("category", "general")

            all_items.extend(items)
            self._last_fetch[url] = datetime.utcnow()
            self.total_fetched += len(items)

            # Be polite - don't hammer RSS servers
            time.sleep(0.5)

        return all_items

    def fetch_reddit(self) -> List[Dict]:
        """Fetch posts from configured subreddits."""
        all_items = []

        for subreddit in self.subreddits:
            source = f"reddit.com/r/{subreddit}"

            last = self._last_fetch.get(source)
            if last and (datetime.utcnow() - last) < self._fetch_interval:
                continue

            logger.info(f"Fetching Reddit: r/{subreddit}")
            items = _fetch_reddit_posts(subreddit, limit=10)

            for item in items:
                item["source"] = source

            all_items.extend(items)
            self._last_fetch[source] = datetime.utcnow()
            self.total_fetched += len(items)

            time.sleep(1)  # Reddit rate limiting

        return all_items

    def ingest_into_filter(self, news_filter, items: List[Dict]) -> int:
        """
        Process fetched items through the NewsRiskFilter.

        Returns number of successfully ingested (non-duplicate) items.
        """
        ingested = 0

        for item in items:
            result = news_filter.ingest_item(
                title=item.get("title", ""),
                content=item.get("content", ""),
                source=item.get("source", "unknown"),
                url=item.get("url", ""),
                published_at=item.get("published_at", datetime.utcnow())
            )

            if result:
                ingested += 1
                self.total_ingested += 1
            else:
                self.total_duplicates += 1

        return ingested

    def run_full_ingestion(self, news_filter) -> Dict:
        """
        Run a complete ingestion cycle: RSS + Reddit -> NewsRiskFilter.
        Returns ingestion statistics.
        """
        start = datetime.utcnow()

        # Fetch RSS
        rss_items = self.fetch_all_rss()
        rss_ingested = self.ingest_into_filter(news_filter, rss_items)

        # Fetch Reddit
        reddit_items = self.fetch_reddit()
        reddit_ingested = self.ingest_into_filter(news_filter, reddit_items)

        elapsed = (datetime.utcnow() - start).total_seconds()

        stats = {
            "timestamp": start.isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "rss_items_fetched": len(rss_items),
            "rss_items_ingested": rss_ingested,
            "reddit_items_fetched": len(reddit_items),
            "reddit_items_ingested": reddit_ingested,
            "total_ingested": rss_ingested + reddit_ingested,
            "total_duplicates": self.total_duplicates,
            "risk_states": news_filter.get_summary(),
        }

        logger.info(f"Ingestion complete: {stats['total_ingested']} new items in {elapsed:.1f}s")

        return stats

    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        return {
            "total_fetched": self.total_fetched,
            "total_ingested": self.total_ingested,
            "total_duplicates": self.total_duplicates,
            "feeds_configured": len(self.rss_feeds),
            "subreddits_configured": len(self.subreddits),
            "errors": self.errors[-10:],  # Last 10 errors
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("""
        TradeXY - News Ingestion Pipeline
        RSS + Reddit -> Score -> Risk States
    """)

    from tradex.engines.news_risk_filter import NewsRiskFilter

    nrf = NewsRiskFilter()
    manager = NewsIngestionManager()

    # Test with simulated items (no network calls)
    test_items = [
        {"title": "Fed Raises Interest Rates",
         "content": "The Federal Reserve raises interest rates by 25 basis points",
         "source": "reuters.com", "url": "https://example.com/1",
         "published_at": datetime.utcnow()},
        {"title": "Bitcoin ETF Sees Record Inflows",
         "content": "Institutional investors pour billions into Bitcoin ETF products",
         "source": "coindesk.com", "url": "https://example.com/2",
         "published_at": datetime.utcnow()},
        {"title": "Tesla Earnings Beat Expectations",
         "content": "Tesla reports record revenue and strong delivery numbers",
         "source": "cnbc.com", "url": "https://example.com/3",
         "published_at": datetime.utcnow()},
    ]

    ingested = manager.ingest_into_filter(nrf, test_items)
    logger.info("  Ingested: {ingested} items")

    stats = manager.get_stats()
    logger.info("  Stats: {stats}")

    # Show risk states
    summary = nrf.get_summary()
    logger.info("\n  Risk State Summary:")
    logger.info("    Total news: {summary[")
    logger.info("    Blocked: {summary[")
    logger.info("    Caution: {summary[")
    logger.info("    Clear: {summary[")

    logger.info("\nNews Ingestion test complete.")
