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
def get_comp_info(query: str) -> str:
    """Search for basic introductions and general information about the company"""
    answer = "Restart the device."
    return answer

@tool
def get_get_hr_processdb_info(query: str) -> str:
    """Search for internal HR policies and expense reimbursement regulations"""
    answer = ""
    return answer


@tool
def get_staff_info(query: str) -> str:
    """Search for the company’s organizational chart and employee contact directory"""
    answer = ""
    return answer

@tool
def get_doc_apprl_info(query: str) -> str:
    """Search for instructions and procedures on using the internal electronic approval system (Amaranth)"""
    answer = ""
    return answer

