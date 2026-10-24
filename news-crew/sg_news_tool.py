"""
SG property news tool (RSS-backed) for the daily brief crew.

Drop-in replacement for the Tavily search tool: it keeps the same tool
name (tavily_news_search) and the same (query) signature, so the rest of
the crew is unchanged. Instead of semantic web search it reads a curated
set of Singapore property-specialist RSS feeds, so every item is on-topic
by construction.
"""

import re
import html
import time
from datetime import datetime, timezone, timedelta

import feedparser
from crewai.tools import BaseTool

# ---- EDIT THIS LIST ---------------------------------------------------
# Singapore property-specialist feeds. The tool tries each one, skips any
# that fail, and prints a per-feed status line (FEED STATUS -> ...) so you
# can see coverage and prune dead entries. Add or remove freely.
FEEDS = [
    ("Stacked Homes",  "https://stackedhomes.com/feed"),
    ("PropertySoul",   "https://propertysoul.com/feed"),
    ("PropertyNet SG", "https://propertynet.sg/feed/?x=1"),
    ("99.co Insider",  "https://www.99.co/singapore/insider/feed"),
    ("Jack Sheo",      "https://jacksheo.com/feed"),
]
# -----------------------------------------------------------------------

LOOKBACK_DAYS = 7    # only items published within this many days
MAX_ITEMS     = 10   # items returned per call
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")

feedparser.USER_AGENT = UA


def _clean(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _published_dt(entry):
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
    return None


def _fetch_all():
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    items, status = [], []
    for name, url in FEEDS:
        try:
            d = feedparser.parse(url, agent=UA)
            n = 0
            for e in d.entries:
                dt = _published_dt(e)
                if dt and dt < cutoff:
                    continue
                link = (e.get("link") or "").strip()
                title = _clean(e.get("title", ""))
                if not link or not title:
                    continue
                summary = _clean(e.get("summary", e.get("description", "")))[:400]
                items.append({
                    "source": name,
                    "title": title,
                    "url": link,
                    "dt": dt,
                    "published": dt.strftime("%a, %d %b %Y") if dt else "n/a",
                    "snippet": summary,
                })
                n += 1
            status.append(f"{name}: {n}")
        except Exception as ex:
            status.append(f"{name}: ERROR ({type(ex).__name__})")

    # de-dupe by url, then sort newest first
    seen, deduped = set(), []
    for it in items:
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        deduped.append(it)
    deduped.sort(key=lambda it: (it["dt"] or datetime.min.replace(tzinfo=timezone.utc)),
                 reverse=True)
    return deduped, status


def _score(item, terms):
    hay = (item["title"] + " " + item["snippet"]).lower()
    return sum(1 for t in terms if t in hay)


def _format(items):
    if not items:
        return ("NO ITEMS: no Singapore property articles were found in the "
                "configured RSS feeds within the lookback window. Do not invent any.")
    out = []
    for i, it in enumerate(items, 1):
        out.append(
            f"[{i}] {it['title']}\n"
            f"URL: {it['url']}\n"
            f"PUBLISHED: {it['published']} ({it['source']})\n"
            f"SNIPPET: {it['snippet']}"
        )
    return "\n\n".join(out)


class SGPropertyNewsTool(BaseTool):
    name: str = "tavily_news_search"
    description: str = (
        "Search recent Singapore property news. Input a short query for the "
        "angle you want (for example: HDB resale, SORA mortgage rates, new "
        "launch, cooling measures). Returns recent items from Singapore "
        "property-specialist sources with headline, URL, publish date and a "
        "short snippet. Use only the snippet text; never add facts not present."
    )

    def _run(self, query: str = "") -> str:
        items, status = _fetch_all()
        terms = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower())
                 if len(w) > 2]
        if terms:
            ranked = sorted(items, key=lambda it: _score(it, terms), reverse=True)
            if ranked and _score(ranked[0], terms) == 0:
                ranked = items  # query matched nothing; fall back to freshest
        else:
            ranked = items
        print("FEED STATUS -> " + " | ".join(status))
        return _format(ranked[:MAX_ITEMS])
