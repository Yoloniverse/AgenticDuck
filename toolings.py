from langchain.agents import tool
from langchain_tavily import TavilySearch
from typing import List
import logging
import os
from dotenv import load_dotenv

######################################################################
#                             Save Log                               #
######################################################################
load_dotenv()
# main에서 만든 것과 동일한 이름 사용
logger = logging.getLogger("Agent")


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
    """For information that changes in real time—such as the latest news or weather—that is not captured within the LLM’s training data, relevant results are provided through web search."""
    search_tool = TavilySearch(max_results=3)
    result = search_tool.invoke(query)
    
    logger.info(f"result: {result}")
    #answer = result['results'][0]['content']
    return result

## 더미 유저 검증 툴 함수 만들기
@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.
    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True

@tool
def get_menual_info(query: str) -> str:
    """Searches equipment manuals for usage instructions, specifications, and maintenance methods."""
    answer = "Restart the device."
    return answer

@tool
def get_db_info(query: str) -> int:
    """Search the equipment raw data from database"""
    answer = 10
    return answer
