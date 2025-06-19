
import os
## 호준 Tavily API Key
os.environ["TAVILY_API_KEY"] = "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"

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
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage


## LangGraph libraries
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState, START, END, StateGraph




## custom made libraries
from toolings import get_current_weather, validate_user, taviliy_web_search_tool

## other libraries
from pydantic import BaseModel, Field
from typing import Any, TypedDict, Annotated
import getpass
from typing import List
import logging.handlers


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










## 위 세 툴들을 리스트로 묶기
tools = [get_current_weather, taviliy_web_search_tool, validate_user]



## Ollama LLM 객체 만들기
llm = ChatOllama(model="llama3.1", temperature=0.1) ## qwen3:8b 다운받아놓음. 한국어 실력이 더 좋다고 함.
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



## simple한 llm invoke function
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}

class CustomStateTypedDict(TypedDict):  
    messages: list[str]
    user_class: dict[str, Any]
    user_email: str


test_dict: CustomStateTypedDict = {"messages" : ["hi?"],
                                    "user_age" : {"animal": "duck"},
                                    "user_email" : "lunchduck@sdt.inc"}

test_dict.get("messages")                
test_dict["user_age"]

class CustomStatePyDantic(BaseModel):  
    messages: list[str]
    user_class: dict[str, Any]
    user_email: str

test_dantic = CustomStatePyDantic(messages = ['hi?'],
                                  user_class = {"what!?" : "Was!?"},
                                  user_email = "DonaldTrump@tesla.com")

test_dantic.user_class

## 우리의 랭그래프 빌딩
builder = StateGraph(MessagesState)
builder.add_node("inference", call_model)
builder.add_edge(START, "inference")

## 다 빌딩 했으면 컴파일로 마무리
graph = builder.compile(checkpointer=checkpointer, store=store)


## invocation test 
graph.invoke({"messages": [{"role": "user", "content": "What is 3 + 4?"}]},
             {"configurable": {"thread_id": "duck"}})





