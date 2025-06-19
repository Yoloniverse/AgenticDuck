
from langchain.agents import tool
from langchain_tavily import TavilySearch
from typing import List
import logging.handlers
import os
## 호준 Tavily API Key
os.environ["TAVILY_API_KEY"] = "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"

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
