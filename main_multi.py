'''
진행사항: thread에서 대화내용을 저장하면서 대화하도록 하였으며, 이전 대화 기록을 필요한 소스만 담도록 함(요약도 필요할듯)
할 일: 
 1. lunchduck 업데이트 갈기기
 2. multi agent 구성을 위해, hallucination 검증하는 agent를 추가.
 3. 사내 규정 RAG tool에 사용할 모듈(db 연결하고 검색해서 결과 반환하는 기능) 만들기(사내 규정 데이터 가공이 필요함)
 4. RAG 툴만들면 문서 종류별로 collection 찾아서 검색 기능 사용하는 함수 만들기
'''


import os
## Langchain libraries
from langchain.llms import Ollama
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
#from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import create_react_agent

# from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

## LangGraph libraries
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.graph.message import add_messages
## custom made libraries
from toolings import taviliy_web_search_tool, get_menual_info, get_db_info

## other libraries
from pydantic import BaseModel, Field
from typing import Any, TypedDict, Annotated, Literal, List, Dict
import getpass
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


## Thread (세션)의 대화내용을 저장하기 위한 체크포인터 . 에이전트에 checkpointer를 전달하면서 여러 호출 간 상태 유지(short-term memory)
checkpointer = InMemorySaver()
## long-term memory를 위한 스토어. 모든 스레드에서 재활용할 수 있는 지식이 필요할 때 씀. 서비스가 running중일 때만 유지됨. 영구 저장은 DB를 사용해야 함.
store = InMemoryStore()

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

tools = [taviliy_web_search_tool, get_menual_info, get_db_info]

## 프롬프트 정의 (툴 호출 에이전트에 적합한 형식)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 사용자 질문에 한국어로 대답해주는 어시스턴트야."),
        ("human", "{user_question}"),
        MessagesPlaceholder("messages"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

prompt_ch = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 AI가 답변한 내용이 사용자의 질문에 맞는지 검증하고 맞지 않는 답변을 걸러서 최종 답변을 생성하는 역할이야. 어떤식으로 검증했는지는 다 생략하고 사용자에게 제공할 최종 답변만 간결하게 해줘."),
        ("human", "사용자 질문: {user_question}\n\nAI 답변: {ai_answer}"),
    ]
)
## 툴 호출 에이전트 생성
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) 
# 검증 체인
reviewer_chain = prompt_ch | llm


class AppState(TypedDict, total=False):
    # 대화(필요하면 계속 누적)
    # messages: List[Dict[str, Any]]
    messages: Annotated[list, add_messages]
    # Executor가 생성한 초안
    #draft_answer: str
    # Reviewer가 후처리한 최종 답
    final_answer: str


def executor_node(state: AppState) -> AppState:
    # 마지막 사용자 메시지에서 질문 꺼내기 (or 별도 필드 사용)
    user_q = state["messages"][-1].content
    out = agent_executor.invoke({"user_question": user_q, "messages": state["messages"], })
    #print("out : ", out['user_question'] + out['output'].split('</think>\n\n')[-1])
    draft = out['output'].split('</think>\n\n')[-1]
    #draft = out.content["output"].split('</think>\n\n')[-1] if hasattr(out, "content") else str(out)
    #draft = getattr(out, "content", str(out))

    # ({
    #     "messages": state["messages"] + [{"role": "assistant", "content": draft}],
    #     "draft_answer": draft
    # })
    print("draft: ",draft)
    print("state['messages']: ",state['messages'])
    print('stats의 messages: ',state['messages'])
    return {
        "messages": [str(draft)],
        "final_answer": str(draft)
    }


# def reviewer_node(state: AppState) -> AppState:
#     user_q = state["messages"][-1].content  # 간단히 마지막을 질문으로 사용
#     ai_ans = state.get("draft_answer", "")
#     out = reviewer_chain.invoke({"user_question": user_q, "ai_answer": ai_ans})
#     #final = out.content if hasattr(out, "content") else str(out)
#     final = getattr(out, "content", str(out))
#     # return {
#     #     "messages": state["messages"] + [{"role": "assistant", "content": final}],
#     #     "final_answer": final
#     # }
#     return {
#         "messages": [{"role": "assistant", "content": final}],
#         "final_answer": final,
#     }

builder = StateGraph(AppState)
builder.add_node("executor", executor_node)
# builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "executor")
# builder.add_edge("executor", "reviewer")
# builder.add_edge("reviewer", END)
builder.add_edge("executor", END)

#checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer, store=store)

# 실행 예시
out = graph.invoke(
    {"messages": [{"role":"user","content":"선릉역 근처에 있는 SDT라는 회사에서 가장 가까운 맛집을 알려줘"}]},
    {"configurable": {"thread_id": "t1"}}
)
print(out["final_answer"])


out = graph.invoke(
    {"messages": [{"role":"user","content":"너가 알려줬던 2번 맛집의 상세 주소가 어디니"}]},
    {"configurable": {"thread_id": "t1"}}
)
print(out["final_answer"])

out = graph.invoke(
    {"messages": [{"role":"user","content":"그럼 그 맛집에서 추천 메뉴 하나만 알려줄래?"}]},
    {"configurable": {"thread_id": "t1"}}
)
print(out["final_answer"])










#--------------------------------------------------------------
# 멀티 agent 생성 시 사용
#route node 생성
class GraphState(TypedDict, total=False):
    messages: List[AnyMessage]      # 대화 메시지 목록
    route: Literal["agent1", "agent2"]  # 라우팅 결과(상위에서 결정)

class RouteDecision(BaseModel):
    route: Literal["agent1", "agent2"]

# llm이 구조화 출력 지원한다고 가정(예: .with_structured_output)
route_llm = llm.with_structured_output(RouteDecision)

def router_node(state: GraphState) -> GraphState:
    system_prompt = (
        "You are a router. Read the conversation and choose exactly one route: "
        "'agent1' or 'agent2'. "
        "Return JSON with key 'route' only."
    )
    # 최신 메시지 컨텍스트에 라우팅 지시를 덧붙여 판단
    messages = (state.get("messages") or []) + [{"role": "system", "content": system_prompt}]
    decision: RouteDecision = route_llm.invoke(messages)
    return {"route": decision.route}  # 상태에 route만 갱신


# 하위 에이전트 실행 노드
def run_agent1(state: GraphState) -> GraphState:
    # agent1은 messages 기반으로 동작/갱신
    out = agent1.invoke({"messages": state["messages"]})
    return {"messages": out["messages"]}

def run_agent2(state: GraphState) -> GraphState:
    out = agent2.invoke({"messages": state["messages"]})
    return {"messages": out["messages"]}


# 라우터의 route 값에 따라 분기
def choose_route(state: GraphState) -> str:
    # 반드시 "agent1" 또는 "agent2" 반환
    return state["route"]

# 그래프 구성
builder = StateGraph(GraphState)

builder.add_node("router", router_node)
builder.add_node("agent1", run_agent1)
builder.add_node("agent2", run_agent2)

# 시작 → 라우터
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    choose_route,
    {
        "agent1": "agent1",
        "agent2": "agent2",
    },
)

# 각 하위 에이전트 실행 후 종료
builder.add_edge("agent1", END)
builder.add_edge("agent2", END)
# 컴파일
graph = builder.compile(checkpointer=checkpointer, store=store)

initial_state: GraphState = {
    "messages": [{"role": "user", "content": "이 입력을 처리해줘"}]
}

# 스레드/대화 ID는 필요 시 지정
result_state = graph.invoke(
    initial_state,
    config={"configurable": {"thread_id": "demo-thread"}}
)

# 최종 메시지 열람
for m in result_state["messages"]:
    print(m["role"], ":", m["content"])




# graph visualization
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass