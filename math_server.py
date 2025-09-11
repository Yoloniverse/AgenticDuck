from mcp.server.fastmcp import FastMCP
"""
간단한 수학 툴을 만들어서 mcp에 등록해서 툴로 쓰는 코드
"""
mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")