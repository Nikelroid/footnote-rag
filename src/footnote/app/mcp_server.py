"""The search, offered as a tool to any agent that speaks the Model Context Protocol.

    python -m footnote.app.mcp_server

An MCP client, such as Claude Desktop or another agent framework, can then call
`search_filings`. The tool only searches. It calls no model and costs nothing,
so the calling agent pays for its own reasoning.
"""

from __future__ import annotations

from functools import lru_cache

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    "footnote",
    instructions="Search SEC filings (10-K, 10-Q, 8-K, earnings releases) of large US companies. Returns whole pages with their document and page number.",
)


@lru_cache(maxsize=1)
def assistant():
    from ..rag.assistant import Assistant

    return Assistant()


@server.tool()
def search_filings(query: str, pages: int = 5) -> list[dict]:
    """Find the filing pages most relevant to a question.

    Name the company and the year in the query, and use the wording of a filing
    where you can, for example "purchases of property, plant and equipment" for
    capital expenditure. Each result has the document name, the page number
    counted from 0, a relevance score and the full text of the page.
    """
    return assistant().search(query, pages=max(1, min(pages, 10)))


if __name__ == "__main__":
    server.run()
