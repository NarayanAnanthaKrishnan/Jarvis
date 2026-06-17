import feedparser


RSS_FEEDS = {
    "general": "http://feeds.bbci.co.uk/news/rss.xml",
    "tech": "https://feeds.feedburner.com/TechCrunch",
    "science": "https://www.sciencedaily.com/rss/top.xml",
    "us": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
}


def get_news(topic: str = "general") -> str:
    topic = topic.lower().strip()
    url = RSS_FEEDS.get(topic, RSS_FEEDS["general"])
    try:
        feed = feedparser.parse(url)
        entries = feed.entries[:5]
        if not entries:
            return "Could not fetch news."
        lines = [f"{i+1}. {e.title}" for i, e in enumerate(entries)]
        return "\n".join(lines)
    except Exception:
        return "Could not fetch news."
