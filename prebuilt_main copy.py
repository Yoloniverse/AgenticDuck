
## 호준 Tavily API Key
# os.environ["TAVILY_API_KEY"] = "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"

## Langchain libraries
from langchain.llms import Ollama
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
# from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser
from langchain_mcp_adapters.client import MultiServerMCPClient



## LangGraph libraries
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

## other libraries
import chromadb
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Any, TypedDict, Annotated, Literal
import getpass
from typing import List
import logging.handlers
import requests
import asyncio
import json
from pprint import pprint

## custom made libraries
from toolings import get_current_weather, validate_user, taviliy_web_search_tool
from utils import create_server_config

load_dotenv()  # .env 파일의 환경변수를 불러옵니다
os.environ.get('TAVILY_API_KEY')




### 멀티에이전트 
"""
1. Subgrapgh를 사용.
2. 각 Subgraph를 독립적으로 memory cache에 저장.
3. 다양한 유저의 요청을 처리하기 위해 asynce로 함수 호출

subgraph_builder = StateGraph(...)
subgraph = subgraph_builder.compile(checkpointer=True)
"""


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

## Ollama LLM 객체 만들기
llm = ChatOllama(model="llama3.1", temperature=0.1) ## qwen3:8b 다운받아놓음. 한국어 실력이 더 좋다고 함.
# tools = [get_current_weather, taviliy_web_search_tool, validate_user]
# llm = ChatOllama(model="qwen3:8b", temperature=0.1) ## qwen3:8b 다운받아놓음. 한국어 실력이 더 좋다고 함.

## LLM에 툴 바인딩하기
# llm_with_tools = llm.bind_tools(tools)




"""
############################################################
############################################################
LangGraph with Pure atonomous LLM tool calls
############################################################
############################################################
"""

## Thread (세션)의 대화내용을 저장하기 위한 체크포인터 (short-term memory)
checkpointer = InMemorySaver()
## long-term memory를 위한 스토어
store = InMemoryStore()


## mcp tool들 명세서 json형식으로 load
with open("/home/sdt/Workspace/mvai/AgenticRAG/mcp_config_websearch.json", "r") as f:
    mcp_config_websearch = json.load(f)


pprint(mcp_config_websearch)
## LangGraph에서 사용할 수 있는 형식으로 mcp tool json 전처리
mcp_config_websearch = create_server_config(mcp_config_websearch)
pprint(mcp_config_websearch)

## mcp server들을 LangChain의 mcp client adapter로 연결
websearch_client = MultiServerMCPClient(mcp_config_websearch)
## 연결된 툴들 조회 
tools = await websearch_client.get_tools()
pprint(tools)


###############
## LangGraph 에이전트 구축

## 1. LangGraph의 pre-built ReAct agent 생성 (가장 기초적)
agent = create_react_agent(
    llm,
    tools,
)
    # checkpointer=checkpointer)

## 
await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

await agent.stream({"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]})
await agent.astream({"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]})




websearch_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)




SQL_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)




doc_retriever_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)


## LangGraph Supervisor 형식의 
### Creating top supervisor 
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

supervisor = create_supervisor(
    model=init_chat_model(model="llama3.1", temperature=0.1),
    agents=[research_agent, math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()