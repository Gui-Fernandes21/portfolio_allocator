from __future__ import annotations

from datetime import date
from typing import Any

from orchestration.base import Agent
from orchestration.llm import LLMUnavailable, complete_json
from orchestration.messages import AgentContext, AgentOutput

TICKER_LIST = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT"]
HEADLINES_PER_TICKER = 4
VALID_SENTIMENTS = {"negative", "neutral", "positive"}

SYSTEM_PROMPT = (
    "You classify the tone of financial news headlines. For each headline "
    "return a sentiment of exactly 'negative', 'neutral' or 'positive', and "
    "one short clause justifying it. Judge only the text given: do not use "
    "outside knowledge, do not speculate about price direction, and do not "
    "give investment advice. Reply with JSON only."
)

_cache: dict[date, dict] = {}

# Normalise one yfinance news item.
def parse_news_item(item: dict) -> dict[str, Any] | None:
    content = item.get("content") if isinstance(item.get("content"), dict) else item

    title = content.get("title") or item.get("title")

    if not title:
        return None

    provider = content.get("provider")

    if isinstance(provider, dict):
        publisher = provider.get("displayName")
    else:
        publisher = provider or item.get("publisher")

    url = None
    for key in ("canonicalUrl", "clickThroughUrl"):
        candidate = content.get(key)
        if isinstance(candidate, dict) and candidate.get("url"):
            url = candidate["url"]
            break
    url = url or item.get("link")

    published = content.get("pubDate") or content.get("displayTime")
    if published is None and item.get("providerPublishTime"):
        from datetime import datetime, timezone
        published = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc).isoformat()

    return {
        "title": title.strip(),
        "publisher": publisher or "unknown",
        "published": published,
        "url": url,
    }

# Recent headlines per ticker. 
def fetch_headlines(tickers = TICKER_LIST, per_ticker = HEADLINES_PER_TICKER):
    import yfinance as yf

    headlines = []
    
    for ticker in tickers:
        try:
            items = yf.Ticker(ticker).get_news(count=per_ticker) or []
        except Exception:
            continue  # skip failures
        for item in items[:per_ticker]:
            parsed = parse_news_item(item)
            if parsed:
                parsed["ticker"] = ticker
                headlines.append(parsed)
    return headlines


def build_prompt(headlines):
    lines = ["Classify each headline. Reply with a JSON array of objects with keys: index, sentiment, reason.", ""]
    
    for index, headline in enumerate(headlines):
        lines.append(f"{index}. [{headline['ticker']}] {headline['title']}")
    
    return "\n".join(lines)

# Aggregate classified headlines by ticker.
def aggregate_by_ticker(headlines):
    """Roll classified headlines up to one reading per ETF."""
    per_asset = {}
    
    for headline in headlines:
        sentiment = headline.get("sentiment")

        if sentiment not in VALID_SENTIMENTS:
            continue

        entry = per_asset.setdefault(
            headline["ticker"],
            {"negative": 0, "neutral": 0, "positive": 0, "headline_count": 0},
        )
        entry[sentiment] += 1
        entry["headline_count"] += 1

    for ticker, counts in per_asset.items():
        positive, negative = counts["positive"], counts["negative"]
        
        if positive > negative:
            counts["overall"] = "positive"
        elif negative > positive:
            counts["overall"] = "negative"
        else:
            counts["overall"] = "neutral"
    
    return per_asset


class SentimentAnalyserAgent(Agent):
    name = "sentiment_analyser"
    uses_llm = True

    def run(self, context: AgentContext) -> AgentOutput:
        today = date.today()
        
        if today in _cache:
            cached = _cache[today]
            return self.output(payload=cached, summary=cached["summary_line"])

        try:
            headlines = fetch_headlines()
        except Exception as error:
            return self.output(
                error=f"{type(error).__name__}: {error}",
                summary="News sentiment unavailable. Headlines could not be retrieved.",
            )

        if not headlines:
            return self.output(
                payload={"per_asset": {}, "sources": [],
                         "classified": False,
                         "classification_error": "no headlines returned"},
                summary="No recent headlines were available for these ETFs.",
            )

        # Classification is optional
        classification_error = None
        
        try:
            classified = complete_json(
                build_prompt(headlines),
                role="sentiment_analyser",
                system=SYSTEM_PROMPT,
                max_tokens=900,
            )
            for entry in classified if isinstance(classified, list) else []:
                index = entry.get("index")
                sentiment = entry.get("sentiment", "").lower().strip()
                if (isinstance(index, int) and 0 <= index < len(headlines) and sentiment in VALID_SENTIMENTS):
                    headlines[index]["sentiment"] = sentiment
                    headlines[index]["reason"] = entry.get("reason")
        except LLMUnavailable as error:
            classification_error = str(error)

        per_asset = aggregate_by_ticker(headlines)
        classified_count = sum(1 for h in headlines if h.get("sentiment"))

        if classified_count:
            leaning = [f"{t} {value['overall']}" for t, value in sorted(per_asset.items())]
            
            summary_line = (
                f"Classified {classified_count} recent headlines across "
                f"{len(per_asset)} ETFs: " + ", ".join(leaning) + "."
            )
        else:
            summary_line = (
                f"Collected {len(headlines)} recent headlines, but tone classification was unavailable."
            )

        payload = {
            "per_asset": per_asset,
            "sources": headlines,          # full attribution for every claim
            "classified": bool(classified_count),
            "classification_error": classification_error,
            "summary_line": summary_line,
        }
        _cache.clear()
        _cache[today] = payload
        return self.output(payload=payload, summary=summary_line)
