import pytest

from padwan_llm.mcp import McpStreamable

pytestmark = pytest.mark.e2e

DATAGOUV_URL = "https://mcp.data.gouv.fr/mcp"


class TestMcpStreamableDataGouv:
    async def test_list_tools(self):
        async with McpStreamable(url=DATAGOUV_URL) as client:
            assert len(client.tools) > 0
            names = {t.name for t in client.tools}
            assert "search_datasets" in names

    async def test_tool_defs_valid(self):
        async with McpStreamable(url=DATAGOUV_URL) as client:
            for tool in client.tools:
                td = tool.to_tool_def()
                assert td["name"] == tool.name
                assert isinstance(td["description"], str)
                assert td["parameters"]["type"] == "object"

    async def test_call_search_datasets(self):
        async with McpStreamable(url=DATAGOUV_URL) as client:
            search = next(t for t in client.tools if t.name == "search_datasets")
            result = await search.handler({"query": "transport"})
            texts = [c["text"] for c in result["content"]]
            combined = "\n".join(texts)
            assert "transport" in combined.lower()
