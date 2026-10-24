"""
news_crew.py
-------------
Daily Singapore property + finance market briefs for Andrew's PropNex AI concierge.

Now supports selectable news TYPES:
  python news_crew.py --type daily --dry-run     # mixed property + finance (default)
  python news_crew.py --type hdb                 # HDB-focused, its own collection
  python news_crew.py --type rates               # interest rates / MAS / SORA
  python news_crew.py --list                     # show all available types

Flow per run:
  Researcher agent  -> searches the selected type's angles (Tavily)
  Curator agent     -> picks 3-5 client-relevant items, plain English, source link
                       on every item, invents no numbers
  Script (not an agent) -> writes the brief locally, then UPLOADS to that type's
                       RAG collection as a single rolling document.

Publishing stays in code, gated behind --dry-run. The crew only writes; the
script decides whether it goes live. No agent ever auto-publishes.
"""

import os
import sys
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Type

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

load_dotenv()

# ----------------------------------------------------------------------
# Config (all from .env / environment)
# ----------------------------------------------------------------------
RAG_BASE_URL = os.environ.get("RAG_BASE_URL", "https://rag.gradientedgeai.com").rstrip("/")
RAG_USERNAME = os.environ.get("RAG_USERNAME", "")
RAG_PASSWORD = os.environ.get("RAG_PASSWORD", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

OUT_DIR = "out"
SGT = ZoneInfo("Asia/Singapore")

# ----------------------------------------------------------------------
# News types: pick one at runtime with --type.
# Each profile = the search angles, the target RAG collection, and the
# rolling filename. Add your own types here freely - nothing else changes.
# ----------------------------------------------------------------------
NEWS_TYPES = {
    "daily": {
        "label": "Daily Brief (property + finance)",
        "collection": "market_news",
        "filename": "market_brief_latest.md",
        "angles": [
            "Singapore private residential property prices / URA data / new launches",
            "HDB resale market / BTO / housing policy",
            "Interest rates / mortgage rates / MAS / SORA",
            "Property cooling measures / stamp duty (ABSD, BSD) / regulation",
        ],
    },
    "hdb": {
        "label": "HDB Brief",
        "collection": "market_news_hdb",
        "filename": "hdb_brief_latest.md",
        "angles": [
            "Singapore HDB resale prices and transaction volumes",
            "HDB BTO launches and supply",
            "HDB grants, eligibility and MOP rules",
            "HDB policy changes and announcements",
        ],
    },
    "newlaunch": {
        "label": "New Launch Brief",
        "collection": "market_news_launch",
        "filename": "newlaunch_brief_latest.md",
        "angles": [
            "Singapore new condo launches and preview / booking weekends",
            "URA new private home sales figures",
            "Executive Condominium (EC) launches",
            "Showflat take-up rates and project pricing",
        ],
    },
    "rates": {
        "label": "Rates & Financing Brief",
        "collection": "market_news_rates",
        "filename": "rates_brief_latest.md",
        "angles": [
            "Singapore SORA and home loan rates",
            "Singapore mortgage fixed vs floating rate trends",
            "MAS monetary policy and outlook",
            "Bank home loan package changes in Singapore",
        ],
    },
    "investor": {
        "label": "Investor Brief",
        "collection": "market_news_investor",
        "filename": "investor_brief_latest.md",
        "angles": [
            "Singapore ABSD and foreign buyer rules",
            "Singapore residential rental market and rental yields",
            "Singapore en bloc / collective sale activity",
            "Singapore property investment outlook",
        ],
    },
}


# ----------------------------------------------------------------------
# Tool: Tavily news search (the only thing the researcher agent calls)
# ----------------------------------------------------------------------
class TavilyNewsInput(BaseModel):
    query: str = Field(..., description="A focused news search query, e.g. 'Singapore HDB resale prices'.")


from sg_news_tool import SGPropertyNewsTool


class TavilyNewsTool(BaseTool):
    name: str = "tavily_news_search"
    description: str = (
        "Search recent news. Returns items with TITLE, URL, PUBLISHED date and a SNIPPET. "
        "Use the SNIPPET text only; do not invent facts or figures. Always preserve the URL."
    )
    args_schema: Type[BaseModel] = TavilyNewsInput

    def _run(self, query: str) -> str:
        try:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            resp = client.search(
                query=query,
                topic="news",
                days=7,
                max_results=8,
                search_depth="advanced",
            )
        except Exception as e:
            return f"SEARCH_ERROR for '{query}': {e}"

        results = resp.get("results", []) if isinstance(resp, dict) else []
        if not results:
            return f"No results for '{query}'."

        lines = []
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            published = (r.get("published_date") or "n/a")
            snippet = (r.get("content") or "").strip().replace("\n", " ")
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."
            lines.append(f"[{i}] {title}\nURL: {url}\nPUBLISHED: {published}\nSNIPPET: {snippet}\n")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Build the crew for a given type
# ----------------------------------------------------------------------
def build_crew(today_str: str, angles_text: str) -> Crew:
    llm = LLM(model="anthropic/claude-sonnet-4-6", temperature=0.2)
    news_tool = SGPropertyNewsTool()

    researcher = Agent(
        role="Singapore property & finance news researcher",
        goal="Find the freshest, most relevant SG news for a property agent's clients.",
        backstory=(
            "You scan the Singapore market for what genuinely affects buyers, sellers, landlords "
            "and investors. You care about accuracy and always keep the source link. You never "
            "state a number that is not in the search snippet you read."
        ),
        tools=[news_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    curator = Agent(
        role="Market brief editor (CEA-aware)",
        goal="Turn raw news items into a short, accurate, plain-English brief a client can trust.",
        backstory=(
            "You write for a CEA-licensed PropNex salesperson, not a financial adviser. You report "
            "facts and link sources; you do not give buy/sell or investment recommendations. You never "
            "invent figures, dates or quotes. If a detail is not in the source, you leave it out."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "Today is {today}. Using the tavily_news_search tool, gather recent Singapore news across "
            "these angles (run a separate search for each, adjust wording as needed):\n"
            "{angles}\n\n"
            "Compile a de-duplicated shortlist of the most relevant, freshest items. For EACH item give: "
            "the headline, the URL, and one sentence on what it says drawn ONLY from the snippet. "
            "Do not add facts that are not in the snippets."
        ),
        expected_output=(
            "A bullet list of 6-10 candidate news items, each with headline, URL, and a one-line summary "
            "taken only from the snippet text."
        ),
        agent=researcher,
    )

    curate_task = Task(
        description=(
            "Today is {today}. From the researcher's shortlist, select the 3 to 5 items that matter most "
            "to a property client right now. Write the final brief in GitHub-flavoured Markdown, with no "
            "preamble and no code fences. Use exactly this header and footer:\n\n"
            "# Singapore Property & Finance - {type_label}\n"
            "_Auto-generated {today} (SGT). Source-linked. General information only; exact figures and "
            "eligibility are confirmed case by case._\n\n"
            "<one to three short sections with sensible headings for the items, e.g. a heading then bullets>\n\n"
            "Each bullet looks like:\n"
            "- **<plain headline>** - <one or two sentence plain-English summary>. [Source](URL)\n\n"
            "---\n"
            "_Shared by Andrew Soh, PropNex Realty. News and general context only - not financial advice._\n\n"
            "Rules:\n"
            "- Every bullet MUST end with a working [Source](URL). Copy each URL from the shortlist exactly, character for character - never retype, shorten, or edit a URL.\n"
            "- Use only facts present in the shortlist. Invent nothing - no numbers, no dates, no quotes.\n"
            "- Plain English. State only what the source reports - no guidance, no implications for the reader, avoid words like should consider or factor in. No sentence may tell the reader what to do.\n"
            "- Group items under headings that fit today's items; omit anything thin rather than padding."
        ),
        expected_output="The finished Markdown brief only, starting with '# Singapore Property & Finance'.",
        agent=curator,
        context=[research_task],
    )

    return Crew(
        agents=[researcher, curator],
        tasks=[research_task, curate_task],
        process=Process.sequential,
        verbose=True,
    )


# ----------------------------------------------------------------------
# Publish (deterministic, in code - not an agent decision)
# ----------------------------------------------------------------------
def upload_brief_to_rag(markdown: str, collection: str, filename: str) -> str:
    """Overwrite the single rolling brief in the given collection.

    Upload contract (confirmed): POST /upload?collection=<name>, multipart field 'file'.
    Cleanup of the previous doc is BEST-EFFORT and assumes DELETE /documents/{filename}
    accepts a 'collection' query param. If your delete contract differs, the upload still
    succeeds - just adjust or remove the delete call below.
    """
    if not RAG_USERNAME or not RAG_PASSWORD:
        raise RuntimeError("RAG_USERNAME / RAG_PASSWORD not set in environment.")

    auth = HTTPBasicAuth(RAG_USERNAME, RAG_PASSWORD)

    try:
        d = requests.delete(
            f"{RAG_BASE_URL}/documents/{filename}",
            params={"collection": collection},
            auth=auth,
            timeout=30,
        )
        cleanup = f"cleanup={d.status_code}"
    except Exception as e:
        cleanup = f"cleanup-skipped ({e})"

    files = {"file": (filename, markdown.encode("utf-8"), "text/markdown")}
    r = requests.post(
        f"{RAG_BASE_URL}/upload",
        params={"collection": collection},
        files=files,
        auth=auth,
        timeout=120,
    )
    r.raise_for_status()
    return f"uploaded {filename} -> {collection} (status={r.status_code}, {cleanup})"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Selectable SG property/finance market brief crew.")
    parser.add_argument("--type", default="daily", choices=list(NEWS_TYPES.keys()),
                        help="Which news type to generate (default: daily).")
    parser.add_argument("--dry-run", action="store_true", help="Generate and print; upload nothing.")
    parser.add_argument("--list", action="store_true", help="List available news types and exit.")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable --type values:\n")
        for key, p in NEWS_TYPES.items():
            print(f"  {key:<10} {p['label']:<34} -> collection: {p['collection']}")
        print()
        return

    for key in ("TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.environ.get(key):
            print(f"[abort] {key} is not set. Edit your .env file.")
            sys.exit(1)

    profile = NEWS_TYPES[args.type]
    today_str = datetime.now(SGT).strftime("%d %b %Y")
    angles_text = "\n".join(f"  {i}. {a}" for i, a in enumerate(profile["angles"], 1))

    crew = build_crew(today_str, angles_text)

    print(f"\n=== Generating '{args.type}' brief for {today_str} (SGT) ===\n")
    result = crew.kickoff(inputs={
        "today": today_str,
        "angles": angles_text,
        "type_label": profile["label"],
    })
    markdown = getattr(result, "raw", str(result)).strip()

    if markdown.startswith("```"):
        markdown = markdown.strip("`")
        markdown = markdown.split("\n", 1)[1] if "\n" in markdown else markdown

    if len(markdown) < 80 or not markdown.startswith("#"):
        print("[abort] Brief looks empty or malformed - not uploading.\n")
        print(markdown)
        sys.exit(2)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(SGT).strftime("%Y-%m-%d")
    with open(os.path.join(OUT_DIR, profile["filename"]), "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(os.path.join(OUT_DIR, f"{args.type}_{stamp}.md"), "w", encoding="utf-8") as f:
        f.write(markdown)

    print("\n----- BRIEF -----\n")
    print(markdown)
    print("\n-----------------\n")

    if args.dry_run:
        print(f"[dry-run] Nothing uploaded. Local copies written to ./{OUT_DIR}/.")
        return

    msg = upload_brief_to_rag(markdown, profile["collection"], profile["filename"])
    print(f"[done] {msg}")


if __name__ == "__main__":
    main()
