import html
import os
import re
from typing import Any
from tavily import TavilyClient
from mcp.types import Tool

web_search_tool = Tool(
    name="web_search",
    description="Search the web for current information using Tavily AI search. Use this when the user asks about current events, recent news, weather, real-time information, or anything that requires up-to-date data.",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5, max: 10)",
                "default": 5,
                "minimum": 1,
                "maximum": 10
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "description": "Search depth: 'basic' for quick results, 'advanced' for more comprehensive search",
                "default": "basic"
            }
        },
        "required": ["query"]
    }
)


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    return text


def format_results_for_llm(query: str, results: list[dict], answer: str = None) -> str:
    lines = []

    if answer:
        lines.append(f"Summary: {answer}")
        lines.append("")

    if results:
        lines.append(f"Search results for \"{query}\":")
        lines.append("")

        for i, r in enumerate(results, 1):
            title = clean_text(r.get("title", "No title"))
            content = clean_text(r.get("content", ""))
            url = r.get("url", "")
            score = r.get("score", 0)

            if len(content) > 400:
                content = content[:397] + "..."

            lines.append(f"{i}. {title}")
            if content:
                lines.append(f"   {content}")
            if url:
                lines.append(f"   Source: {url}")
            lines.append("")

    if not lines:
        return f"No search results found for: {query}"

    return "\n".join(lines)


async def execute_web_search(arguments: dict[str, Any]) -> str:
    query = arguments.get("query")
    max_results = min(arguments.get("max_results", 5), 10)
    search_depth = arguments.get("search_depth", "basic")

    if not query:
        return "Error: Query is required"

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable not set."

    try:

        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=True,
            include_raw_content=False
        )

        results = response.get("results", [])
        answer = response.get("answer", "")

        if not results and not answer:
            return f"No search results found for: {query}"

        return format_results_for_llm(query, results, answer)

    except Exception as e:
        return f"Search error: {str(e)}"
