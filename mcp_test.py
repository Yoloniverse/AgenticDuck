import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient
from pprint import pprint 
from langchain_ollama import ChatOllama
# 1. mcp JSON 정보
mcp_json_data = """
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"]
    },
    "duckduckgo-mcp-server": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@nickclyde/duckduckgo-mcp-server",
        "--key",
        "c4718807-23a0-4979-a1f2-348ee6e37b20"
      ]
    }
  }
}
"""

# 2. JSON 데이터 파싱
mcp_info = json.loads(mcp_json_data)


# 3. 각 서버별 도구 생성
mcp_tools = []
server_data = mcp_info["mcpServers"]

server_data.keys()



# 도구 함수의 입력 스키마를 정의합니다.
class McpToolInput(BaseModel):
    """mcp-server의 인자들을 나타내는 Pydantic 모델."""
    args: List[str] = Field(description="실행할 mcp-server에 전달할 인자 목록.")


def create_mcp_tool(name: str, server_info: Dict[str, Any]):
    """주어진 mcp 서버 정보로 동적으로 @tool 함수를 생성합니다."""
    
    # 함수 이름과 설명을 동적으로 만듭니다.
    func_name = f"run_{name}_server"
    docstring = f"Runs the {name} mcp-server with provided arguments."
    
    # 실제 mcp 서버를 실행하는 핵심 로직
    def tool_func(args: List[str]):
        """Runs the mcp-server. This part needs to be implemented."""
        print(f"--- Calling mcp-server: '{name}' ---")
        print(f"Command: {server_info['command']}")
        print(f"Static args: {server_info['args']}")
        print(f"Dynamic args: {args}")
        
        # 실제 실행 로직: subprocess.run() 등을 사용하여 외부 명령어를 실행
        # 예: result = subprocess.run([server_info['command']] + server_info['args'] + args, capture_output=True, text=True)
        
        # 실제 mcp-server 실행 결과를 반환해야 합니다.
        return f"mcp-server '{name}' executed successfully with args: {args}"

    # @tool 데코레이터를 사용하여 함수를 도구로 등록합니다.
    # 함수 이름과 독스트링을 동적으로 할당하고, Pydantic 모델을 통해 입력 스키마를 정의합니다.
    tool_func.__name__ = func_name
    tool_func.__doc__ = docstring
    return tool(args_schema=McpToolInput)(tool_func)


# 4. 각 서버에 대해 도구 함수를 생성하고 리스트에 추가
for server_name, server_config in server_data.items():
    new_tool = create_mcp_tool(server_name, server_config)
    mcp_tools.append(new_tool)





"""
1. 남이 만든 MCP 서버들을 json으로 정보를 가져와서 MultiServerMCPClient을 사용하여 일반적인 툴로 등록하는 방법
https://langchain-ai.github.io/langgraph/agents/mcp/
"""

# ## mcp tool들 명세서 json형식으로 load
# with open("/home/sdt/Workspace/mvai/AgenticRAG/mcp_config.json", "r") as f:
#     mcp_config_websearch = json.load(f)

## mcp tool들 명세서 json형식으로 load
with open("/home/sdt/Workspace/mvai/AgenticRAG/mcp_config_websearch.json", "r") as f:
    mcp_config_websearch = json.load(f)

def create_server_config(mcp_json):
    config = mcp_json
    server_config = {}

    if config and "mcpServers" in config:
        for server_name, server_config_data in config["mcpServers"].items():
            # command가 있으면 stdio 방식
            if "command" in server_config_data:
                server_config[server_name] = {
                    "command": server_config_data.get("command"),
                    "args": server_config_data.get("args", []),
                    "transport": "stdio",
                }
            # url이 있으면 sse 방식
            elif "url" in server_config_data:
                server_config[server_name] = {
                    "url": server_config_data.get("url"),
                    "transport": "sse",
                }

    return server_config


mcp_config_websearch = create_server_config(mcp_config_websearch)
pprint(mcp_config_websearch)
mcp_config_websearch.keys()

## mcp server들을 LangChain의 mcp client adapter로 연결
websearch_client = MultiServerMCPClient(mcp_config_websearch)

## 연결된 툴들 조회 
websearch_tools = await websearch_client.get_tools()

pprint(websearch_tools)

llm = ChatOllama(model="qwen3:8b", base_url="http://127.0.0.1:11434")


## llm에 tool 등록 하면 됨 
websearch_agent = create_react_agent(
    model=llm,
    tools=websearch_tools,
    prompt=(
        "You are a websearch agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with web searching tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="websearch_agent",
)



"""
2. 남이 만든 MCP 서버들을 json으로 정보를 가져와서 MCP 서버의 호출 로직을 직접 구현하고, 이를 @tool 데코레이터로 감싸는 방식로 등록하는 방법
"""



import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# 1. mcp JSON 정보
mcp_json_data = """
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"]
    },
    "duckduckgo-mcp-server": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@nickclyde/duckduckgo-mcp-server",
        "--key",
        "c4718807-23a0-4979-a1f2-348ee6e37b20"
      ]
    }
  }
}
"""

# 2. JSON 데이터 파싱 및 'playwright' 서버 정보만 추출
mcp_info = json.loads(mcp_json_data)
playwright_server_info = mcp_info["mcpServers"]["playwright"]

# 3. 'playwright' 서버를 위한 Tool 정의
# Tool 함수의 입력 스키마를 정의합니다.
class PlaywrightToolInput(BaseModel):
    """Playwright mcp-server에 전달할 인자들을 나타내는 Pydantic 모델."""
    args: List[str] = Field(description="실행할 playwright 서버에 전달할 인자 목록.")

@tool(args_schema=PlaywrightToolInput)
def run_playwright_server(args: List[str]):
    """Runs the Playwright mcp-server with provided arguments."""
    print(f"--- Calling mcp-server: 'playwright' ---")
    print(f"Command: {playwright_server_info['command']}")
    print(f"Static args: {playwright_server_info['args']}")
    print(f"Dynamic args: {args}")

    # 실제 subprocess.run() 등을 사용하여 외부 명령어를 실행하는 로직이 여기에 들어갑니다.
    # 예: result = subprocess.run([playwright_server_info['command']] + playwright_server_info['args'] + args, capture_output=True, text=True)

    # 실제 서버 실행 결과를 반환해야 합니다.
    return f"mcp-server 'playwright' executed successfully with dynamic args: {args}"

# 4. LangGraph에 LLM과 도구 연결
# ChatOpenAI 모델을 인스턴스화하고, 생성된 도구 함수를 bind_tools 메서드로 연결합니다.
# tools는 항상 리스트 형태여야 하므로, [run_playwright_server]와 같이 리스트로 감싸줍니다.
llm_with_tools = ChatOpenAI(model="gpt-4o").bind_tools([run_playwright_server])

# 5. LLM에 요청을 보내 도구 사용을 유도하는 예제
prompt = "Use the playwright tool to navigate to a website. The argument should be 'https://example.com'."

# LLM을 호출하여 도구 호출 결과를 확인
response = llm_with_tools.invoke(prompt)

# LLM의 응답에서 도구 호출 정보 확인
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print("\n--- LLM's Tool Call ---")
    print(f"Tool Name: {tool_call['name']}")
    print(f"Tool Arguments: {tool_call['args']}")

    # 도구 호출 실행
    if tool_call['name'] == "run_playwright_server":
        result = run_playwright_server.invoke(tool_call['args'])
        print(f"--- Tool Execution Result: {result} ---")







"""
3.  그럼 MultiServerMCPClient활용해서 langgraph에 등록하는 전체 코드를 줘
"""


import os
import json
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from pprint import pprint

# 1. 환경 설정: MCP JSON 파일 경로와 API 키 설정
# 가정: 이 파일은 mcp_config_websearch.json이라는 이름으로 존재합니다.
# 이 파일을 직접 생성하거나, 아래 예시 JSON을 사용하세요.
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# 예시 MCP JSON 파일 내용
"""
{
  "mcpServers": {
    "duckduckgo-mcp-server": {
      "command": "npx",
      "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@nickclyde/duckduckgo-mcp-server",
        "--key",
        "c4718807-23a0-4979-a1f2-348ee6e37b20"
      ]
    }
  }
}
"""

# mcp_config_websearch.json 파일이 존재하지 않는 경우를 대비한 가상 경로
mcp_config_path = "./mcp_config_websearch.json"

# 가상 MCP 클라이언트 클래스 (LangChain 라이브러리에서 가져와야 함)
class MultiServerMCPClient:
    def __init__(self, config: dict):
        self.config = config
    
    async def get_tools(self):
        # 실제 구현에서는 MCP 서버에 연결하여 툴 명세를 가져옵니다.
        # 여기서는 JSON 설정에 기반한 가상 툴을 반환합니다.
        from langchain_core.tools import tool
        from langchain_core.pydantic_v1 import BaseModel, Field

        class SearchToolInput(BaseModel):
            query: str = Field(description="The query to search for.")

        @tool(args_schema=SearchToolInput)
        def duckduckgo_search(query: str):
            """Searches for a query using the DuckDuckGo search engine."""
            print(f"--- MCP Client is running search for: {query} ---")
            # 실제 MCP 서버 호출 로직이 여기에 들어갑니다.
            return f"Mock search result for '{query}': LangGraph is a library for building LLM applications."
        
        return [duckduckgo_search]

# 2. MCP 클라이언트를 초기화하고 툴을 가져옵니다.
async def get_mcp_tools():
    # 실제로는 파일에서 로드합니다.
    # with open(mcp_config_path, "r") as f:
    #    mcp_config = json.load(f)
    
    # 예시용으로 직접 JSON 데이터를 사용
    mcp_config = {
      "mcpServers": {
        "duckduckgo-mcp-server": {
          "command": "npx",
          "args": [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@nickclyde/duckduckgo-mcp-server",
            "--key",
            "c4718807-23a0-4979-a1f2-348ee6e37b20"
          ]
        }
      }
    }
    
    # MultiServerMCPClient 인스턴스 생성
    websearch_client = MultiServerMCPClient(mcp_config)
    
    # 툴을 비동기적으로 가져옵니다.
    websearch_tools = await websearch_client.get_tools()
    print("--- MCP 툴 로드 완료 ---")
    pprint(websearch_tools)
    return websearch_tools

# 3. LangGraph를 위한 에이전트 및 상태 정의
class AgentState(TypedDict):
    messages: List[BaseMessage]
    
# 4. LangGraph 에이전트 노드 생성
async def create_agent_node():
    # 툴을 비동기적으로 가져옵니다.
    tools = await get_mcp_tools()
    
    # LLM 모델 초기화
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # create_react_agent를 사용하여 에이전트 생성
    # 이 함수는 툴 명세를 LLM에게 전달하고, 툴을 실행하는 로직을 포함합니다.
    websearch_agent_node = create_react_agent(
        llm=llm,
        tools=tools,
        checkpointer=None,
        messages_channel="messages"
    )
    return websearch_agent_node

# 5. LangGraph 그래프 구성 및 실행
async def main():
    # 에이전트 노드 생성
    websearch_agent_node = await create_agent_node()

    # LangGraph 워크플로우 정의
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("websearch_agent", websearch_agent_node)
    
    # 시작점 설정
    workflow.set_entry_point("websearch_agent")
    
    # 단일 노드이므로, 노드에서 END로 연결
    workflow.add_edge("websearch_agent", END)
    
    # 그래프 컴파일
    app = workflow.compile()
    
    # 그래프 실행
    print("\n--- LangGraph 실행 ---")
    inputs = {"messages": [HumanMessage(content="LangGraph가 무엇인지 찾아줘.")]}
    
    for s in app.stream(inputs, stream_mode="values"):
        print(s)
        
    print("\n--- LangGraph 실행 완료 ---")

# main 함수 실행
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())