"""
Stores signals, news items, risk states, and system configuration
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import asdict

logger = logging.getLogger("tradex.db")

class MongoDBManager:
    """
    MongoDB connection and CRUD operations for TradeXY.
    
    Collections:
        - signals: Trading signals (A-grade)
        - news_items: Ingested news with scores
        - risk_states: Per-symbol risk states
        - settings: User-configurable thresholds
        - system_log: Audit trail
    
    Usage:
        db = MongoDBManager("mongodb://localhost:27017")
        db.save_signal(signal_dict)
        signals = db.get_signals(symbol="BTCUSDT", limit=10)
    """
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017",
                 db_name: str = "tradex"):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self._client = None
        self._db = None
        self._connected = False
        
        self._connect()
    
    def _connect(self):
        """Attempt MongoDB connection."""
        try:
            from pymongo import MongoClient
            
            self._client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000
            )
            # Test connection
            self._client.admin.command('ping')
            self._db = self._client[self.db_name]
            self._connected = True
            
            # Create indexes
            self._create_indexes()
            
            logger.info(f"Connected to MongoDB: {self.db_name}")
        except Exception as e:
            logger.warning(f"MongoDB not available: {e}. Using in-memory fallback.")
            self._connected = False
            self._memory_store = {
                "signals": [],
                "news_items": [],
                "risk_states": {},
                "settings": {},
                "system_log": [],
            }
    
    def _create_indexes(self):
        """Create MongoDB indexes for efficient queries."""
        if not self._connected:
            return
        
        try:
            self._db.signals.create_index([("symbol", 1), ("timestamp", -1)])
            self._db.signals.create_index([("timestamp", -1)])
            self._db.signals.create_index([("grade", 1)])
            
            self._db.news_items.create_index([("published_at", -1)])
            self._db.news_items.create_index([("asset_mentions", 1)])
            self._db.news_items.create_index([("content_hash", 1)], unique=True)
            
            self._db.risk_states.create_index([("symbol", 1)], unique=True)
            
            logger.info("MongoDB indexes created")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    # --- Signals ---
    
    def save_signal(self, signal_dict: Dict) -> str:
        """Save a trading signal. Returns the signal ID."""
        signal_dict["_created_at"] = datetime.utcnow()
        
        if self._connected:
            result = self._db.signals.insert_one(signal_dict)
            return str(result.inserted_id)
        else:
            self._memory_store["signals"].append(signal_dict)
            return f"mem_{len(self._memory_store['signals'])}"
    
    def get_signals(self, symbol: Optional[str] = None,
                    grade: str = "A",
                    hours: int = 168,
                    limit: int = 50) -> List[Dict]:
        """Get recent signals, optionally filtered by symbol."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        if self._connected:
            query = {"timestamp": {"$gte": cutoff.isoformat()}}
            if symbol:
                query["symbol"] = symbol
            if grade:
                query["grade"] = grade
            
            return list(
                self._db.signals
                .find(query, {"_id": 0})
                .sort("timestamp", -1)
                .limit(limit)
            )
        else:
            items = self._memory_store["signals"]
            if symbol:
                items = [s for s in items if s.get("symbol") == symbol]
            return items[-limit:]
    
    def get_signal_count(self, hours: int = 24) -> Dict:
        """Get signal counts by grade and symbol."""
        if self._connected:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            pipeline = [
                {"$match": {"_created_at": {"$gte": cutoff}}},
                {"$group": {
                    "_id": {"symbol": "$symbol", "grade": "$grade"},
                    "count": {"$sum": 1}
                }}
            ]
            results = list(self._db.signals.aggregate(pipeline))
            return {"counts": results}
        else:
            return {"counts": len(self._memory_store["signals"])}
    
    # --- News Items ---
    
    def save_news_item(self, item_dict: Dict) -> bool:
        """Save a news item. Returns False if duplicate."""
        if self._connected:
            try:
                self._db.news_items.insert_one(item_dict)
                return True
            except Exception:
                return False  # Likely duplicate
        else:
            self._memory_store["news_items"].append(item_dict)
            return True
    
    def get_news(self, symbol: Optional[str] = None,
                 impact: Optional[str] = None,
                 hours: int = 24,
                 limit: int = 50) -> List[Dict]:
        """Get recent news items with optional filters."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        if self._connected:
            query = {"published_at": {"$gte": cutoff.isoformat()}}
            if symbol:
                query["asset_mentions"] = symbol
            if impact:
                query["impact_level"] = impact
            
            return list(
                self._db.news_items
                .find(query, {"_id": 0})
                .sort("published_at", -1)
                .limit(limit)
            )
        else:
            items = self._memory_store["news_items"]
            return items[-limit:]
    
    # --- Risk States ---
    
    def save_risk_state(self, symbol: str, state_dict: Dict):
        """Upsert risk state for a symbol."""
        state_dict["_updated_at"] = datetime.utcnow()
        
        if self._connected:
            self._db.risk_states.update_one(
                {"symbol": symbol},
                {"$set": state_dict},
                upsert=True
            )
        else:
            self._memory_store["risk_states"][symbol] = state_dict
    
    def get_risk_states(self) -> Dict[str, Dict]:
        """Get all risk states."""
        if self._connected:
            states = {}
            for doc in self._db.risk_states.find({}, {"_id": 0}):
                states[doc.get("symbol", "unknown")] = doc
            return states
        else:
            return self._memory_store["risk_states"]
    
    # --- Settings ---
    
    def save_settings(self, settings: Dict):
        """Save user settings."""
        if self._connected:
            self._db.settings.update_one(
                {"_id": "user_settings"},
                {"$set": settings},
                upsert=True
            )
        else:
            self._memory_store["settings"] = settings
    
    def get_settings(self) -> Dict:
        """Load user settings."""
        if self._connected:
            doc = self._db.settings.find_one({"_id": "user_settings"})
            return doc if doc else {}
        else:
            return self._memory_store["settings"]
    
    # --- System Log ---
    
    def log_event(self, event_type: str, details: Dict):
        """Log a system event."""
        event = {
            "type": event_type,
            "details": details,
            "timestamp": datetime.utcnow()
        }
        
        if self._connected:
            self._db.system_log.insert_one(event)
        else:
            self._memory_store["system_log"].append(event)
            # Keep only last 1000 events
            if len(self._memory_store["system_log"]) > 1000:
                self._memory_store["system_log"] = self._memory_store["system_log"][-500:]
    
    # --- Dashboard Stats ---
    
    def get_dashboard_stats(self) -> Dict:
        """Get aggregated stats for the dashboard overview."""
        if self._connected:
            now = datetime.utcnow()
            day_ago = now - timedelta(hours=24)
            week_ago = now - timedelta(days=7)
            
            return {
                "signals_24h": self._db.signals.count_documents(
                    {"_created_at": {"$gte": day_ago}}
                ),
                "signals_7d": self._db.signals.count_documents(
                    {"_created_at": {"$gte": week_ago}}
                ),
                "news_24h": self._db.news_items.count_documents(
                    {"_created_at": {"$gte": day_ago}}
                ),
                "active_risk_blocks": self._db.risk_states.count_documents(
                    {"state": "BLOCK"}
                ),
                "db_connected": True,
            }
        else:
            return {
                "signals_24h": len(self._memory_store["signals"]),
                "news_24h": len(self._memory_store["news_items"]),
                "active_risk_blocks": sum(
                    1 for rs in self._memory_store["risk_states"].values()
                    if rs.get("state") == "BLOCK"
                ),
                "db_connected": False,
                "note": "Using in-memory storage (MongoDB not available)"
            }
    
    def close(self):
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")

# --- Standalone Test ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("""
        TradeXY - MongoDB Layer
        Signals + News + Risk States + Settings
    """)
    
    db = MongoDBManager()
    
    print(f"  Connected: {db.is_connected}")
    
    # Test signal
    db.save_signal({
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "grade": "A",
        "timestamp": datetime.utcnow().isoformat(),
        "entry_price": 65000,
        "stop_loss": 63500,
        "tp1": 66500,
        "reason": "Test signal"
    })
    
    signals = db.get_signals()
    print(f"  Signals stored: {len(signals)}")
    
    # Test risk state
    db.save_risk_state("BTCUSDT", {
        "symbol": "BTCUSDT",
        "state": "CLEAR",
        "reasons": [],
    })
    
    states = db.get_risk_states()
    print(f"  Risk states: {len(states)}")
    
    # Dashboard stats
    stats = db.get_dashboard_stats()
    print(f"  Dashboard stats: {stats}")
    
    db.close()
    print("\nMongoDB layer test complete.")
