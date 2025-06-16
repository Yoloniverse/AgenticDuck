
import os
## 호준 Tavily API Key
os.environ["TAVILY_API_KEY"] = "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"
from langchain.llms import Ollama
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

from pydantic import BaseModel, Field
import getpass
from typing import List
import logging.handlers


######################################################################
#                             Save Log                               #
######################################################################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_max_size = 1024000
log_file_count = 3
log_fileHandler = logging.handlers.RotatingFileHandler(
        filename=f"/home/sdt/Workspace/mvai/AgenticRAG/logs/agent.log",
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode='a')

log_fileHandler.setFormatter(formatter)
logger.addHandler(log_fileHandler)





# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")
# _set_env("ANTHROPIC_API_KEY")


## 더미 날씨 툴 함수 만들기
@tool
def get_current_weather(city: str) -> dict:
    """Get the current weather for a specified city."""
    ## 실제 날씨 API 호출 로직
    if city == "Seoul":
        return {"city": "Seoul", "temperature": "25C", "conditions": "Sunny"}
    else:
        return {"city": city, "temperature": "N/A", "conditions": "Unknown"}





"""
https://python.langchain.com/docs/integrations/tools/tavily_search/
"""
## 타빌리 서치엔진 초기화
# search_tool = TavilySearch(max_results=2)
# result = search_tool.invoke("2025 June 9th, there was a final round of nations league for football. Who won?")
# result = search_tool.invoke("대한민국 선릉역 근처에서 가장 가봐야 할 맛집은 어디야??")
# result['results'][0]['content']

## 타빌리 서치엔진 툴 객체 만들기
@tool
def taviliy_web_search_tool(query: str) -> str:
    """Search the web using Tavily."""
    
    search_tool = TavilySearch(max_results=1)
    result = search_tool.invoke(query)
    logger.info(f"result: {result}")
    answer = result['results'][0]['content']
    return answer

## 더미 유저 검증 툴 함수 만들기
@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.
    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True






## 위 세 툴들을 리스트로 묶기
tools = [get_current_weather, taviliy_web_search_tool, validate_user]



## Ollama LLM 객체 만들기
llm = ChatOllama(model="llama3.1", temperature=0.1)
## LLM에 툴 바인딩하기
llm_with_tools = llm.bind_tools(tools)

## LLM에 툴 바인딩한 후, invoke 메소드로 툴 호출하여 툴 사용 테스트 하기 
result = llm_with_tools.invoke("2025 June 9th, there was a final round of nations league for football. Who won?")
# result = llm_with_tools.invoke("선릉역 근처에 있는 SDT라는 회사에서 가장 가까운 맛집을 알려줘")

## 툴을 사용했는지 확인
# result.tool_calls

##이것은 미완성임 툴을 LLM이 잘 사용하게 하려면, LLM의 템플릿을 준수 하면서 AgentExecutor를 사용해야 함
## 프롬프트 정의 (툴 호출 에이전트에 적합한 형식)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

## 툴 호출 에이전트 생성
agent = create_tool_calling_agent(llm, tools, prompt)

## AgentExecutor 생성
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) ## verbose=True로 설정하면 실행 과정확인 가능

## AgentExecutor 실행
response_from_agent = agent_executor.invoke({"input": "2025 June 9th, there was a final round of nations league for football. Who won?"})
response_from_agent = agent_executor.invoke({"input": "What happened to Donald Trump at 10th of June in 2025?"})
response_from_agent = agent_executor.invoke({"input": "What is the best restaurant near Gangnam station?"})
response_from_agent = agent_executor.invoke({"input": "강남에 있는 선릉역 근처에 SDT라는 회사에 대해 말해줘"})
print("\n--- Final Answer from AgentExecutor ---")
print(response_from_agent["output"]) 

##현재 agent_executor에 등록되어 있는 툴 확인
agent_executor.tools 


