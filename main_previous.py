
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
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
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


## other libraries
import chromadb
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Any, TypedDict, Annotated, Literal
import getpass
from typing import List
import logging.handlers


## custom made libraries
from toolings import get_current_weather, validate_user, taviliy_web_search_tool


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











## 위 세 툴들을 리스트로 묶기
tools = [get_current_weather, taviliy_web_search_tool, validate_user]



## Ollama LLM 객체 만들기
llm = ChatOllama(model="llama3.1", temperature=0.1) ## qwen3:8b 다운받아놓음. 한국어 실력이 더 좋다고 함.
# llm = ChatOllama(model="qwen3:8b", temperature=0.1) ## qwen3:8b 다운받아놓음. 한국어 실력이 더 좋다고 함.
## LLM에 툴 바인딩하기
llm_with_tools = llm.bind_tools(tools)

## LLM에 툴 바인딩한 후, invoke 메소드로 툴 호출하여 툴 사용 테스트 하기 
# result = llm_with_tools.invoke("2025 June 9th, there was a final round of nations league for football. Who won?")
result = llm_with_tools.invoke("선릉역 근처에 있는 SDT라는 회사에서 가장 가까운 맛집을 알려줘")

## 툴을 사용했는지 확인
result.tool_calls

## 이것은 미완성임 툴을 LLM이 잘 사용하게 하려면, LLM의 템플릿을 준수 하면서 AgentExecutor를 사용해야 함
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


##db = Chroma.from_documents(documents, OpenAIEmbeddings())



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



class SQLAgentState(BaseModel):  
    messages: list[str]
    user_class: dict[str, Any]
    user_email: str




class WebsearchAgentState(BaseModel):  
    messages: list[str]
    user_class: dict[str, Any]
    user_email: str




class RAGsearchAgentState(BaseModel):  
    messages: list[str]
    user_class: dict[str, Any]
    user_email: str





test_dantic = CustomStatePyDantic(messages = ['hi?'],
                                  user_class = {"what!?" : "Was!?"},
                                  user_email = "DonaldTrump@tesla.com")

test_dantic.user_class


import requests


class MCPQueryTool(Tool):
    name = "mcp_sqlite_query"
    description = "Execute a SQL query via MCP SQLite API"

    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def run(self, query: str) -> str:
        resp = requests.post(f"{self.base_url}/query", json={"query": query})
        resp.raise_for_status()
        data = resp.json()
        # MCP 응답 구조에 맞게 포맷팅
        return data.get("result", "")




## 우리의 랭그래프 빌딩
builder = StateGraph(MessagesState)
builder.add_node("inference", call_model)
builder.add_edge(START, "inference")

## 다 빌딩 했으면 컴파일로 마무리
graph = builder.compile(checkpointer=checkpointer, store=store)


## invocation test 
graph.invoke({"messages": [{"role": "user", "content": "What is 3 + 4?"}]},
             {"configurable": {"thread_id": "duck"}})




## 멀티에이전트 스켈레톤 랭그래프 구축 

class SupervisorOutput(BaseModel):
    next_agent: Literal["SQLAgent", "WebsearchAgent", "RAGSearchAgent", END]

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


## LangChain 모델 wrapper를 활용하면 형식이 자동 처리됨.
## LangGraph에서 LLM 노드를 정의할 때 LangChain의 ChatOpenAI, ChatAnthropic 등을 사용하면 내부적으로 프롬프트 형식을 자동으로 맞춰줌.
## 예시####################################################################################################################################
"""
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")  # OpenAI의 형식을 자동 사용
LangGraph 노드에서는 llm.invoke(prompt) 형태로 사용하면 프롬프트 포맷 걱정 없이 사용 가능.
"""
####################################################################################################################################


prompt = SystemMessage(content="You are a nice pirate")
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="{input}"),
    AIMessage(content='blah')
])



parser = PydanticOutputParser(pydantic_object=SupervisorOutput)
dir(parser)

prompt = ChatPromptTemplate.from_messages([
    ##Use one of 'human', 'user', 'ai', 'assistant', or 'system'.
    ("system", "You are a supervisor. Decide which agent to call next based on the conversation."),
    ("human", "{messages}"),
])




prompt = ChatPromptTemplate([
    ##Use one of 'human', 'user', 'ai', 'assistant', or 'system'.
    ("system", "You are a supervisor. Decide which agent to call next based on the conversation."),
    ("human", "{messages}")])

filled_prompt = prompt.format(name="Alice")
v.to_string()


prompt_template = ChatPromptTemplate([
    ####Use one of 'human', 'user', 'ai', 'assistant', or 'system'.
    ("system", "You are a supervisor. Decide which agent to call next based on the conversation."),
    ("human", "{messages}"),
    # ("assistant", f"Output in this format:\n{parser.get_format_instructions()}")
    ("assistant", f"Output in this format:\n{parser.to_json()}")
])
chain = prompt | llm_with_tools | parser
chain.invoke({"messages": "cats"})


from pprint import pprint

print("""You are the top supervisor of multiple agents.
    You MUST select only one subsequent agent among 3 agents.
    The only agent options we have is these 3 as blow
    
    - SQLAgent, WebsearchAgent, RAGSearchAgent
    
       Like I mentioned earlier, you MUST choose one agent out of these and you cannot skip choosing one. 
    """)

def topSupervisor(topSupervisorState: MessagesState) -> Command[Literal["SQLAgent", "WebsearchAgent", "RAGSearchAgent",END]]:
    """You are the top supervisor of multiple agents.
    You MUST select only one subsequent agent among 3 agents.
    The only agent options we have is these 3 as blow

    - SQLAgent, WebsearchAgent, RAGSearchAgent

    Like I mentioned earlier, you MUST choose one agent out of these and you cannot skip choosing one. 
    """
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which agent to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_agent" field)
    chain = prompt | llm_with_tools | parser
    response = chain.invoke({'messages': topSupervisorState["messages"]})
    # route to one of the agents or exit based on the supervisor's decision
    # if the supervisor returns "__end__", the graph will finish execution
    return Command(goto=response.next_agent)

def SQLAgent(state: MessagesState) -> Command[Literal["topSupervisor"]]:
    """
    You are a SQL agent.
    """
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # and add any additional logic (different models, custom prompts, structured output, etc.)
    response = llm_with_tools.invoke(...)
    return Command(
        goto="topSupervisor",
        update={"messages": [response]},
    )

def WebsearchAgent(state: MessagesState) -> Command[Literal["topSupervisor"]]:
    response = llm_with_tools.invoke(...)
    return Command(
        goto="topSupervisor",
        update={"messages": [response]},
    )

def RAGSearchAgent(state: MessagesState) -> Command[Literal["topSupervisor"]]:
    response = llm_with_tools.invoke(...)
    return Command(
        goto="topSupervisor",
        update={"messages": [response]},
    )

builder = StateGraph(MessagesState)
builder.add_node(topSupervisor)
builder.add_node(SQLAgent)
builder.add_node(WebsearchAgent)
builder.add_node(RAGSearchAgent)
builder.add_edge(START, "topSupervisor")
supervisor = builder.compile(checkpointer=checkpointer, store=store)




## 위 세 툴들을 리스트로 묶기
tools = [get_current_weather, taviliy_web_search_tool, validate_user]



## Ollama LLM 객체 만들기
llm = ChatOllama(model="llama3.1", temperature=0.1) ## qwen3:8b 다운받아놓음. 한국어 실력이 더 좋다고 함.
# llm = ChatOllama(model="qwen3:8b", temperature=0.1) ## qwen3:8b 다운받아놓음. 한국어 실력이 더 좋다고 함.
## LLM에 툴 바인딩하기
llm_with_tools = llm.bind_tools(tools)


builder = StateGraph(MessagesState)

supervisor = builder.compile(checkpointer=checkpointer, store=store)