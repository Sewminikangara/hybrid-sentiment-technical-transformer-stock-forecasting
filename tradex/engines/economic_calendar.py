"""
Economic Calendar Integration

"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EventImpact(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EventType(Enum):
    INTEREST_RATE = "INTEREST_RATE"
    INFLATION = "INFLATION"
    EMPLOYMENT = "EMPLOYMENT"
    GDP = "GDP"
    CENTRAL_BANK_SPEECH = "CENTRAL_BANK_SPEECH"
    EARNINGS = "EARNINGS"
    HALVING = "HALVING"
    OTHER = "OTHER"


@dataclass
class EconomicEvent:
    """A single scheduled economic event."""
    name: str
    event_type: EventType
    impact: EventImpact
    scheduled_time: datetime
    currency_affected: List[str]
    description: str = ""
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None


@dataclass
class CalendarCheckResult:
    """Result of checking the calendar for upcoming events."""
    has_upcoming_high_impact: bool
    has_upcoming_medium_impact: bool
    upcoming_events: List[EconomicEvent]
    next_high_impact: Optional[EconomicEvent]
    minutes_to_next_high: Optional[int]
    should_block_signals: bool
    should_reduce_confidence: bool
    affected_symbols: List[str]
    summary: str


class EconomicCalendar:
    """
    Manages a schedule of economic events and provides blocking
    recommendations for the signal engine.

    Block window: signals are blocked for a configurable period
    before and after HIGH-impact events.
    Caution window: confidence is reduced around MEDIUM-impact events.
    """

    # Known recurring events with their typical schedule
    RECURRING_EVENTS = [
        {
            "name": "FOMC Interest Rate Decision",
            "event_type": EventType.INTEREST_RATE,
            "impact": EventImpact.HIGH,
            "currency_affected": ["USD", "EURUSD", "GBPUSD", "USDJPY",
                                   "BTCUSDT", "ETHUSDT"],
            "description": "Federal Reserve interest rate announcement",
        },
        {
            "name": "US CPI (Inflation)",
            "event_type": EventType.INFLATION,
            "impact": EventImpact.HIGH,
            "currency_affected": ["USD", "EURUSD", "GBPUSD", "USDJPY",
                                   "BTCUSDT", "ETHUSDT"],
            "description": "US Consumer Price Index release",
        },
        {
            "name": "US Non-Farm Payrolls",
            "event_type": EventType.EMPLOYMENT,
            "impact": EventImpact.HIGH,
            "currency_affected": ["USD", "EURUSD", "GBPUSD", "USDJPY"],
            "description": "Monthly US employment report",
        },
        {
            "name": "ECB Interest Rate Decision",
            "event_type": EventType.INTEREST_RATE,
            "impact": EventImpact.HIGH,
            "currency_affected": ["EUR", "EURUSD"],
            "description": "European Central Bank rate decision",
        },
        {
            "name": "BOE Interest Rate Decision",
            "event_type": EventType.INTEREST_RATE,
            "impact": EventImpact.HIGH,
            "currency_affected": ["GBP", "GBPUSD"],
            "description": "Bank of England rate decision",
        },
        {
            "name": "US GDP (Quarterly)",
            "event_type": EventType.GDP,
            "impact": EventImpact.MEDIUM,
            "currency_affected": ["USD", "EURUSD"],
            "description": "US Gross Domestic Product quarterly release",
        },
        {
            "name": "US Jobless Claims (Weekly)",
            "event_type": EventType.EMPLOYMENT,
            "impact": EventImpact.LOW,
            "currency_affected": ["USD"],
            "description": "Weekly initial jobless claims",
        },
    ]

    # Symbol to currency mapping for filtering
    SYMBOL_CURRENCIES = {
        "EURUSD": ["EUR", "USD"],
        "GBPUSD": ["GBP", "USD"],
        "USDJPY": ["USD", "JPY"],
        "AUDUSD": ["AUD", "USD"],
        "USDCAD": ["USD", "CAD"],
        "USDCHF": ["USD", "CHF"],
        "BTCUSDT": ["BTC", "USD"],
        "ETHUSDT": ["ETH", "USD"],
        "AAPL": ["USD"],
        "GOOGL": ["USD"],
        "TSLA": ["USD"],
        "AMZN": ["USD"],
        "MSFT": ["USD"],
    }

    def __init__(self, block_minutes_before: int = 60,
                 block_minutes_after: int = 30,
                 caution_minutes_before: int = 120,
                 caution_minutes_after: int = 60):
        """
        Args:
            block_minutes_before: Block signals this many minutes before HIGH events.
            block_minutes_after: Block signals this many minutes after HIGH events.
            caution_minutes_before: Reduce confidence before MEDIUM events.
            caution_minutes_after: Reduce confidence after MEDIUM events.
        """
        self.block_before = block_minutes_before
        self.block_after = block_minutes_after
        self.caution_before = caution_minutes_before
        self.caution_after = caution_minutes_after
        self._events: List[EconomicEvent] = []

    def add_event(self, event: EconomicEvent):
        """Add a scheduled event to the calendar."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.scheduled_time)

    def add_events_from_schedule(self, events: List[Dict]):
        """
        Bulk add events from a list of dicts.

        Each dict must have: name, event_type, impact, scheduled_time,
        currency_affected.
        """
        for ev in events:
            self.add_event(EconomicEvent(
                name=ev["name"],
                event_type=EventType[ev.get("event_type", "OTHER")],
                impact=EventImpact[ev.get("impact", "LOW")],
                scheduled_time=ev["scheduled_time"],
                currency_affected=ev.get("currency_affected", []),
                description=ev.get("description", ""),
                forecast_value=ev.get("forecast_value"),
                previous_value=ev.get("previous_value"),
            ))

    def load_recurring_events(self, dates: List[datetime]):
        """
        Create events from the recurring template for given dates.
        Useful for generating a test calendar.
        """
        for date in dates:
            for template in self.RECURRING_EVENTS:
                event = EconomicEvent(
                    name=template["name"],
                    event_type=template["event_type"],
                    impact=template["impact"],
                    scheduled_time=date,
                    currency_affected=template["currency_affected"],
                    description=template["description"],
                )
                self.add_event(event)

    def check(self, symbol: str,
              current_time: Optional[datetime] = None) -> CalendarCheckResult:
        """
        Check if any upcoming events should block or adjust signals
        for the given symbol.

        Args:
            symbol: Trading symbol to check.
            current_time: Current timestamp (defaults to now).

        Returns:
            CalendarCheckResult with blocking recommendations.
        """
        now = current_time or datetime.now(timezone.utc)

        # Find relevant currencies
        currencies = self.SYMBOL_CURRENCIES.get(symbol, ["USD"])

        # Filter events relevant to this symbol
        relevant = []
        for ev in self._events:
            # Check if any currency overlaps
            if any(c in ev.currency_affected for c in currencies) or \
               symbol in ev.currency_affected:
                # Check if event is in the relevant window
                window_start = ev.scheduled_time - timedelta(
                    minutes=self.caution_before
                )
                window_end = ev.scheduled_time + timedelta(
                    minutes=self.block_after
                )
                if window_start <= now <= window_end:
                    relevant.append(ev)
                elif now < ev.scheduled_time:
                    # Future event within the next 24 hours
                    if (ev.scheduled_time - now).total_seconds() < 86400:
                        relevant.append(ev)

        # Determine blocking
        should_block = False
        should_caution = False
        next_high = None
        minutes_to_high = None

        high_events = [e for e in relevant if e.impact == EventImpact.HIGH]
        medium_events = [e for e in relevant if e.impact == EventImpact.MEDIUM]

        for ev in high_events:
            block_start = ev.scheduled_time - timedelta(minutes=self.block_before)
            block_end = ev.scheduled_time + timedelta(minutes=self.block_after)
            if block_start <= now <= block_end:
                should_block = True
            if next_high is None or ev.scheduled_time < next_high.scheduled_time:
                next_high = ev
                minutes_to_high = int(
                    (ev.scheduled_time - now).total_seconds() / 60
                )

        for ev in medium_events:
            caution_start = ev.scheduled_time - timedelta(minutes=self.caution_before)
            caution_end = ev.scheduled_time + timedelta(minutes=self.caution_after)
            if caution_start <= now <= caution_end:
                should_caution = True

        # Affected symbols
        affected = set()
        for ev in relevant:
            for sym, curs in self.SYMBOL_CURRENCIES.items():
                if any(c in ev.currency_affected for c in curs):
                    affected.add(sym)

        # Summary
        parts = []
        if should_block:
            parts.append(f"BLOCKED: {next_high.name}")
        elif should_caution:
            parts.append(f"CAUTION: {len(medium_events)} medium-impact event(s)")
        else:
            parts.append("CLEAR")

        if next_high and minutes_to_high is not None:
            parts.append(f"Next HIGH in {minutes_to_high}min")

        summary = " | ".join(parts)

        return CalendarCheckResult(
            has_upcoming_high_impact=len(high_events) > 0,
            has_upcoming_medium_impact=len(medium_events) > 0,
            upcoming_events=relevant,
            next_high_impact=next_high,
            minutes_to_next_high=minutes_to_high,
            should_block_signals=should_block,
            should_reduce_confidence=should_caution,
            affected_symbols=sorted(affected),
            summary=summary,
        )


if __name__ == "__main__":
    logger.info("Economic Calendar Integration Test")
    logger.info("=")

    now = datetime.utcnow()

    calendar = EconomicCalendar()

    # Add some test events
    calendar.add_event(EconomicEvent(
        name="FOMC Rate Decision",
        event_type=EventType.INTEREST_RATE,
        impact=EventImpact.HIGH,
        scheduled_time=now + timedelta(minutes=30),
        currency_affected=["USD", "EURUSD", "BTCUSDT"],
        description="Fed rate decision - expected hold",
        forecast_value=5.25,
        previous_value=5.25,
    ))

    calendar.add_event(EconomicEvent(
        name="US GDP Q4",
        event_type=EventType.GDP,
        impact=EventImpact.MEDIUM,
        scheduled_time=now + timedelta(hours=3),
        currency_affected=["USD"],
        description="Quarterly GDP revision",
    ))

    calendar.add_event(EconomicEvent(
        name="Weekly Jobless Claims",
        event_type=EventType.EMPLOYMENT,
        impact=EventImpact.LOW,
        scheduled_time=now + timedelta(hours=5),
        currency_affected=["USD"],
    ))

    # Check for different symbols
    for sym in ["EURUSD", "BTCUSDT", "AAPL", "GBPUSD"]:
        result = calendar.check(sym, current_time=now)
        logger.info("\n  {sym}:")
        logger.info("    Block: {result.should_block_signals}")
        logger.info("    Caution: {result.should_reduce_confidence}")
        logger.info("    Upcoming: {len(result.upcoming_events)} events")
        if result.next_high_impact:
            print(f"    Next HIGH: {result.next_high_impact.name} "
                  f"in {result.minutes_to_next_high}min")
        logger.info("    Summary: {result.summary}")

    logger.info("\nEconomic calendar test complete.")
