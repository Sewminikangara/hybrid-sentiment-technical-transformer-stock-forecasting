"""
FinBERT Sentiment Integration
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:


    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = "cpu", max_length: int = 512):
        """
        Args:
            device: 'cpu' or 'cuda' for GPU inference.
            max_length: Maximum token length for input truncation.
        """
        self.device = device
        self.max_length = max_length
        self._pipeline = None
        self._available = False
        self._load_model()

    def _load_model(self):
        """Attempt to load the FinBERT model. Fail silently if not available."""
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                tokenizer=self.MODEL_NAME,
                device=self.device,
                truncation=True,
                max_length=self.max_length,
            )
            self._available = True
            logger.info("FinBERT model loaded successfully.")
        except ImportError:
            logger.warning(
                "transformers or torch not installed. "
                "FinBERT unavailable; using rule-based fallback."
            )
        except Exception as e:
            logger.warning(f"FinBERT loading failed: {e}. Using fallback.")

    @property
    def is_available(self) -> bool:
        return self._available

    def score(self, text: str) -> float:
        """
        Score text sentiment from -1.0 (negative) to +1.0 (positive).
        Maps FinBERT labels: positive -> +score, negative -> -score,
        neutral -> score near 0.

        Args:
            text: Input text to classify.

        Returns:
            Sentiment score in [-1.0, +1.0].
        """
        if not self._available or not text:
            return 0.0

        try:
            text = str(text).strip()[:self.max_length * 4]
            result = self._pipeline(text)[0]
            label = result["label"].lower()
            confidence = result["score"]

            if label == "positive":
                return confidence
            elif label == "negative":
                return -confidence
            else:
                return 0.0
        except Exception as e:
            logger.debug(f"FinBERT inference error: {e}")
            return 0.0

    def score_batch(self, texts: list) -> list:

        if not self._available or not texts:
            return [0.0] * len(texts)

        try:
            cleaned = [str(t).strip()[:self.max_length * 4] for t in texts]
            results = self._pipeline(cleaned)
            scores = []
            for result in results:
                label = result["label"].lower()
                conf = result["score"]
                if label == "positive":
                    scores.append(conf)
                elif label == "negative":
                    scores.append(-conf)
                else:
                    scores.append(0.0)
            return scores
        except Exception as e:
            logger.debug(f"FinBERT batch error: {e}")
            return [0.0] * len(texts)

    def analyze(self, text: str) -> Dict:

        if not self._available or not text:
            return {"label": "neutral", "confidence": 0.5, "score": 0.0}

        try:
            text = str(text).strip()[:self.max_length * 4]
            result = self._pipeline(text)[0]
            label = result["label"].lower()
            confidence = result["score"]

            if label == "positive":
                score = confidence
            elif label == "negative":
                score = -confidence
            else:
                score = 0.0

            return {
                "label": label,
                "confidence": confidence,
                "score": score,
            }
        except Exception as e:
            logger.debug(f"FinBERT analysis error: {e}")
            return {"label": "neutral", "confidence": 0.5, "score": 0.0}


class HybridSentimentScorer:


    def __init__(self, finbert_weight: float = 0.7, device: str = "cpu"):

        self.finbert_weight = finbert_weight
        self.finbert = FinBERTSentimentAnalyzer(device=device)
        self._rule_scorer = None

    def _get_rule_scorer(self):
        """Lazy import to avoid circular dependency."""
        if self._rule_scorer is None:
            from tradex.engines.news_risk_filter import SimpleSentimentScorer
            self._rule_scorer = SimpleSentimentScorer()
        return self._rule_scorer

    def score(self, text: str) -> float:
        """
        Compute hybrid sentiment score.

        Args:
            text: Input text.

        Returns:
            Score in [-1.0, +1.0].
        """
        rule_score = self._get_rule_scorer().score(text)

        if not self.finbert.is_available:
            return rule_score

        finbert_score = self.finbert.score(text)
        w = self.finbert_weight
        hybrid = w * finbert_score + (1.0 - w) * rule_score
        return max(-1.0, min(1.0, hybrid))

    def score_detailed(self, text: str) -> Dict:
        """
        Return detailed breakdown of both scoring methods.

        Returns:
            Dict with rule_score, finbert_score, hybrid_score, finbert_available.
        """
        rule_score = self._get_rule_scorer().score(text)
        finbert_result = self.finbert.analyze(text)

        if self.finbert.is_available:
            w = self.finbert_weight
            hybrid = w * finbert_result["score"] + (1.0 - w) * rule_score
        else:
            hybrid = rule_score

        return {
            "rule_score": rule_score,
            "finbert_score": finbert_result["score"],
            "finbert_label": finbert_result["label"],
            "finbert_confidence": finbert_result["confidence"],
            "finbert_available": self.finbert.is_available,
            "hybrid_score": max(-1.0, min(1.0, hybrid)),
            "finbert_weight": self.finbert_weight,
        }


if __name__ == "__main__":
    logger.info("FinBERT Sentiment Integration Test")
    logger.info("=")

    analyzer = FinBERTSentimentAnalyzer()
    logger.info("  FinBERT available: {analyzer.is_available}")

    test_texts = [
        "Federal Reserve raises interest rates by 50 basis points",
        "Bitcoin surges to new all-time high as ETF inflows accelerate",
        "Major exchange hack leads to $100M in stolen funds",
        "Company reports steady quarterly earnings in line with expectations",
    ]

    for text in test_texts:
        result = analyzer.analyze(text)
        logger.info("\n  Text: {text[:60]}...")
        logger.info("  Label: {result[")

    logger.info("\nHybrid Scorer Test")
    logger.info("-")
    hybrid = HybridSentimentScorer()

    for text in test_texts:
        detail = hybrid.score_detailed(text)
        logger.info("\n  Text: {text[:60]}...")
        print(f"  Rule: {detail['rule_score']:+.3f}, "
              f"FinBERT: {detail['finbert_score']:+.3f}, "
              f"Hybrid: {detail['hybrid_score']:+.3f}")

    logger.info("\nFinBERT Sentiment Integration test complete.")
