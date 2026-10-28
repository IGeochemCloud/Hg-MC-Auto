import asyncio
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

root = Path.cwd()
params = StdioServerParameters(
    command=str(root / "hg-mc-auto/bin/hgmc"),
    args=["mcp", "--action", "stdio"],
    cwd=root,
)


async def main():
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            print("server=" + init.server_info.name + "@" + init.server_info.version)
            print("tools=" + ",".join(tool.name for tool in tools.tools))
            info = await session.call_tool("get_element_info", {"name": "hg"})
            print("hg_info=" + str(info.structured_content))


asyncio.run(main())