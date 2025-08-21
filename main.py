import os
## Langchain libraries
from langchain.llms import Ollama
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
#from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately

# from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

## LangGraph libraries
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState, START, END, StateGraph

## custom made libraries
from toolings import get_current_weather, validate_user, taviliy_web_search_tool, get_menual_info, get_db_info
## other libraries
from pydantic import BaseModel, Field
from typing import Any, TypedDict, Annotated, Literal, List
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
os.makedirs('./logs', exist_ok=True)
logger = logging.getLogger("Agent")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_max_size = 1024000
log_file_count = 3
log_fileHandler = logging.handlers.RotatingFileHandler(
        filename=f"./logs/agent.log",
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode='a')

log_fileHandler.setFormatter(formatter)
logger.addHandler(log_fileHandler)


# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")
# _set_env("ANTHROPIC_API_KEY")

# pre_model_hook = summarization_node 으로 agent 생성 시 제공. 대화내용을 요약해서 유지함으로써 LLM 컨텍스트 윈도우 초과 방지.
# summarization_node = SummarizationNode(
#     token_counter=count_tokens_approximately,
#     model=model,
#     max_tokens=384,
#     max_summary_tokens=128,
#     output_messages_key="llm_input_messages",
# )


## Ollama LLM 객체 만들기
# ollama pul qwen3:8b 
llm = ChatOllama(model="qwen3:8b", temperature=0.1) ## qwen3:8b 다운받아놓음. 한국어 실력이 더 좋다고 함.
## LLM에 툴 바인딩하기 test
#llm_with_tools = llm.bind_tools(tools)

## LLM에 툴 바인딩한 후, invoke 메소드로 툴 호출하여 툴 사용 테스트 하기 
#result = llm_with_tools.invoke("2025 June 9th, there was a final round of nations league for football. Who won?")
#result = llm_with_tools.invoke("선릉역 근처에 있는 SDT라는 회사에서 가장 가까운 맛집을 알려줘")
#print(result)
## 툴을 사용했는지 확인
#result.tool_calls


# Search Agent Gen
tools = [get_current_weather, taviliy_web_search_tool, validate_user, get_menual_info, get_db_info]
## 프롬프트 정의 (툴 호출 에이전트에 적합한 형식)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
## 툴 호출 에이전트 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# 멀티 agent 생성 시 사용
# route node 생성
# class GraphState(TypedDict, total=False):
#     messages: List[AnyMessage]      # 대화 메시지 목록
#     route: Literal["agent1", "agent2"]  # 라우팅 결과(상위에서 결정)

# class RouteDecision(BaseModel):
#     route: Literal["agent1", "agent2"]

# # llm이 구조화 출력 지원한다고 가정(예: .with_structured_output)
# route_llm = llm.with_structured_output(RouteDecision)




## AgentExecutor 생성 - 단일 agnet에 사용하는 편
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) ## verbose=True로 설정하면 실행 과정확인 가능
## AgentExecutor 실행 - 단일 agnet에 사용하는 편
response_from_agent = agent_executor.invoke({"input": "What's the weather like in Seoul, Korea today?"})
#print(response_from_agent['output'])




## Thread (세션)의 대화내용을 저장하기 위한 체크포인터 . 에이전트에 checkpointer를 전달하면서 여러 호출 간 상태 유지(short-term memory)
checkpointer = InMemorySaver()
## long-term memory를 위한 스토어. 모든 스레드에서 재활용할 수 있는 지식이 필요할 때 씀. 서비스가 running중일 때만 유지됨. 영구 저장은 DB를 사용해야 함.
store = InMemoryStore()


## agent invoke function
def call_model(state: MessagesState):
    response_from_agent = agent_executor.invoke({"input": state["messages"]})
    response = response_from_agent['output'].split('</think>\n\n')[-1]
    #response = llm.invoke(state["messages"])
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
res_fine=graph.invoke({"messages": [{"role": "user", "content": "What is the weather like in Seoul"}]},
             {"configurable": {"thread_id": "duck"}})
print(res_fine['messages'][-1].content)



# graph visualization
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass