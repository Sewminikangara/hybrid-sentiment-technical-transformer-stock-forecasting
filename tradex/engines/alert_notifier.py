"""
Alert Notification System

"""

import json
import logging
import os
import smtplib
import urllib.request
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AlertMessage:
    """Structured alert message."""
    title: str
    symbol: str
    direction: str
    grade: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    confidence: int
    checklist: List[str]
    timestamp: str
    extra: str = ""

    def to_text(self) -> str:
        """Plain text format."""
        checks = "\n".join(f"  {c}" for c in self.checklist)
        return (
            f"{self.title}\n"
            f"{'=' * 40}\n"
            f"Symbol:     {self.symbol}\n"
            f"Direction:  {self.direction}\n"
            f"Grade:      {self.grade}\n"
            f"Confidence: {self.confidence}%\n"
            f"\n"
            f"Entry:  {self.entry_price:.4f}\n"
            f"SL:     {self.stop_loss:.4f}\n"
            f"TP1:    {self.take_profit_1:.4f}\n"
            f"TP2:    {self.take_profit_2:.4f}\n"
            f"TP3:    {self.take_profit_3:.4f}\n"
            f"\nChecklist:\n{checks}\n"
            f"\nTime: {self.timestamp}\n"
            f"{self.extra}"
        )

    def to_html(self) -> str:
        """HTML format for email."""
        checks = "".join(f"<li>{c}</li>" for c in self.checklist)
        color = "#22c55e" if self.direction == "LONG" else "#ef4444"
        return f"""
        <div style="font-family: Arial, sans-serif; max-width: 500px;
                    border: 1px solid #ddd; border-radius: 8px; padding: 20px;">
            <h2 style="color: {color}; margin-top: 0;">{self.title}</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><strong>Symbol</strong></td>
                    <td>{self.symbol}</td></tr>
                <tr><td><strong>Direction</strong></td>
                    <td style="color: {color};">{self.direction}</td></tr>
                <tr><td><strong>Grade</strong></td>
                    <td>{self.grade}</td></tr>
                <tr><td><strong>Confidence</strong></td>
                    <td>{self.confidence}%</td></tr>
            </table>
            <hr style="margin: 15px 0;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><strong>Entry</strong></td>
                    <td>{self.entry_price:.4f}</td></tr>
                <tr><td><strong>Stop Loss</strong></td>
                    <td style="color: #ef4444;">{self.stop_loss:.4f}</td></tr>
                <tr><td><strong>TP1</strong></td>
                    <td>{self.take_profit_1:.4f}</td></tr>
                <tr><td><strong>TP2</strong></td>
                    <td>{self.take_profit_2:.4f}</td></tr>
                <tr><td><strong>TP3</strong></td>
                    <td>{self.take_profit_3:.4f}</td></tr>
            </table>
            <hr style="margin: 15px 0;">
            <strong>Checklist:</strong>
            <ul style="margin: 5px 0;">{checks}</ul>
            <p style="color: #888; font-size: 12px;">
                {self.timestamp}
            </p>
        </div>
        """


class TelegramNotifier:
    """Send alerts via Telegram Bot API."""

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: Optional[str] = None,
                 chat_id: Optional[str] = None):
        """
        Args:
            token: Bot API token (or set TRADEX_TELEGRAM_TOKEN env var).
            chat_id: Target chat ID (or set TRADEX_TELEGRAM_CHAT_ID env var).
        """
        self.token = token or os.environ.get("TRADEX_TELEGRAM_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TRADEX_TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.token and self.chat_id)

        if self._enabled:
            logger.info("Telegram notifications enabled.")
        else:
            logger.info("Telegram not configured (token/chat_id missing).")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def send(self, message: str) -> bool:
        """
        Send a text message via Telegram.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self._enabled:
            logger.debug("Telegram not configured; message not sent.")
            return False

        try:
            url = self.API_URL.format(token=self.token)
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }).encode("utf-8")

            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                if result.get("ok"):
                    logger.info("Telegram message sent successfully.")
                    return True
                else:
                    logger.warning(f"Telegram API error: {result}")
                    return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_alert(self, alert: AlertMessage) -> bool:
        """Send a structured alert message."""
        return self.send(alert.to_text())


class EmailNotifier:
    """Send alerts via SMTP email."""

    def __init__(self, smtp_server: Optional[str] = None,
                 smtp_port: Optional[int] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 to_address: Optional[str] = None):
        """
        All parameters can be set via environment variables:
            TRADEX_EMAIL_SMTP, TRADEX_EMAIL_PORT, TRADEX_EMAIL_USER,
            TRADEX_EMAIL_PASS, TRADEX_EMAIL_TO
        """
        self.smtp_server = smtp_server or os.environ.get("TRADEX_EMAIL_SMTP", "")
        self.smtp_port = smtp_port or int(os.environ.get("TRADEX_EMAIL_PORT", "587"))
        self.username = username or os.environ.get("TRADEX_EMAIL_USER", "")
        self.password = password or os.environ.get("TRADEX_EMAIL_PASS", "")
        self.to_address = to_address or os.environ.get("TRADEX_EMAIL_TO", "")
        self._enabled = bool(
            self.smtp_server and self.username and
            self.password and self.to_address
        )

        if self._enabled:
            logger.info("Email notifications enabled.")
        else:
            logger.info("Email not configured (SMTP credentials missing).")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def send(self, subject: str, body_text: str,
             body_html: Optional[str] = None) -> bool:
        """
        Send an email.

        Args:
            subject: Email subject line.
            body_text: Plain text body.
            body_html: Optional HTML body.

        Returns:
            True if sent, False otherwise.
        """
        if not self._enabled:
            logger.debug("Email not configured; message not sent.")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.username
            msg["To"] = self.to_address

            msg.attach(MIMEText(body_text, "plain"))
            if body_html:
                msg.attach(MIMEText(body_html, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email sent to {self.to_address}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def send_alert(self, alert: AlertMessage) -> bool:
        """Send a structured alert via email."""
        return self.send(
            subject=f"TradeXY Signal: {alert.symbol} {alert.direction}",
            body_text=alert.to_text(),
            body_html=alert.to_html(),
        )


class AlertManager:
    """
    Unified alert manager that dispatches to all configured
    notification channels.
    """

    def __init__(self):
        self.telegram = TelegramNotifier()
        self.email = EmailNotifier()
        self._history: List[Dict] = []

    @property
    def any_enabled(self) -> bool:
        return self.telegram.is_enabled or self.email.is_enabled

    def send_signal_alert(self, alert: AlertMessage) -> Dict[str, bool]:
        """
        Send alert through all enabled channels.

        Returns:
            Dict mapping channel name to success status.
        """
        results = {}

        if self.telegram.is_enabled:
            results["telegram"] = self.telegram.send_alert(alert)

        if self.email.is_enabled:
            results["email"] = self.email.send_alert(alert)

        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": alert.symbol,
            "direction": alert.direction,
            "channels": results,
        })

        return results

    def get_history(self) -> List[Dict]:
        """Return alert dispatch history."""
        return self._history.copy()


if __name__ == "__main__":
    logger.info("Alert Notification System Test")
    logger.info("=")

    # Create a test alert
    alert = AlertMessage(
        title="A-GRADE LONG Signal",
        symbol="BTCUSDT",
        direction="LONG",
        grade="A",
        entry_price=45000.00,
        stop_loss=44000.00,
        take_profit_1=46500.00,
        take_profit_2=48000.00,
        take_profit_3=50000.00,
        confidence=87,
        checklist=[
            "[PASS] Trend: Bullish (EMA200)",
            "[PASS] Structure: BOS confirmed + retest",
            "[PASS] Elliott: Wave 2->3, conf=87%",
            "[PASS] News: CLEAR",
            "[PASS] Risk: 1.5 ATR SL, 2R target",
        ],
        timestamp=datetime.utcnow().isoformat(),
    )

    logger.info("\nPlain text format:")
    logger.info(alert.to_text())

    logger.info("\nHTML format (truncated):")
    html = alert.to_html()
    logger.info(html[:200] + "...")

    # Test channels (will report not configured)
    manager = AlertManager()
    logger.info("\nTelegram enabled: {manager.telegram.is_enabled}")
    logger.info("Email enabled: {manager.email.is_enabled}")

    if manager.any_enabled:
        results = manager.send_signal_alert(alert)
        logger.info("Send results: {results}")
    else:
        logger.info("No channels configured. Set environment variables to enable.")

    logger.info("\nAlert system test complete.")
