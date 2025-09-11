from mcp.server.fastmcp import FastMCP
"""
간단한 날씨체크 더미 툴을 만들어서 mcp에 등록해서 툴로 쓰는 코드 (이 코드는 streamable-http로 구현하였기 때문에,
인터넷을 사용하는 형식이다)
"""
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")