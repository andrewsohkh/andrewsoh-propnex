"""Wrap today's brief into market-news.json for the website."""
import json, pathlib, sys
from datetime import datetime
from zoneinfo import ZoneInfo

now = datetime.now(ZoneInfo("Asia/Singapore"))
brief = pathlib.Path(__file__).parent / "out" / f"daily_{now:%Y-%m-%d}.md"
if not brief.exists():
    sys.exit(f"No fresh brief for {now:%Y-%m-%d} - not publishing a stale one.")
md = brief.read_text(encoding="utf-8")
out = pathlib.Path(__file__).parent.parent / "market-news.json"
out.write_text(json.dumps({
    "date": f"{now:%Y-%m-%d}",
    "updated": f"{now.day} {now:%b %Y}",
    "markdown": md,
}, ensure_ascii=False), encoding="utf-8")
print("wrote", out, f"({len(md)} chars)")
