
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
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, AnyMessage

## LangGraph libraries
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

## custom made libraries
#from toolings_hr import get_staff_info, get_doc_apprl_info
## other libraries
from pydantic import BaseModel, Field
from typing import Any, TypedDict, Annotated, Literal, List, Dict, Optional
import getpass
from typing import List
import logging.handlers

#sql node
from sqlalchemy import create_engine, text, inspect

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
        filename=f"./logs/HR_agent.log",
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode='a')

log_fileHandler.setFormatter(formatter)
logger.addHandler(log_fileHandler)
## db 연결
db_user = "admin"
db_password = "sdt251327"
db_host = "127.0.0.1"
db_name = "langgraph" 

# Const`ruct the connection string
connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(connection_string)
connection = engine.connect()
inspector = inspect(engine)

## Thread (세션)의 대화내용을 저장하기 위한 체크포인터 . 에이전트에 checkpointer를 전달하면서 여러 호출 간 상태 유지(short-term memory)
#checkpointer = InMemorySaver()
## long-term memory를 위한 스토어. 모든 스레드에서 재활용할 수 있는 지식이 필요할 때 씀. 서비스가 running중일 때만 유지됨. 영구 저장은 DB를 사용해야 함.
store = InMemoryStore()

## Ollama LLM 객체 만들기
# ollama pul qwen3:8b 
# 4bit 모델: "qwen3:8b-q4_K_M"
# 8bit 모델: "qwen3:8b-q8_0"
# 16bit 모델 : "qwen3:8b-fp16"
llm = ChatOllama(model="qwen3:8b", temperature=0.1) ## qwen3:8b 다운받아놓음. 한국어 실력이 더 좋다고 함.


prompt_router = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 사용자 질문을 읽고 사내 결재 기안 규정에 대한 질문인지, 사내 직원 정보에 대한 질문인지 판단하는 역할이야."),
        ("human", "{user_question}"),
        MessagesPlaceholder("messages")
    ]
)


class AppState(TypedDict, total=False):
    # 대화(필요하면 계속 누적)
    messages: Annotated[list[dict], add_messages]
    # router_node의 분기 결과
    path_results: str
    # Executor가 생성한 초안
    RAG_result: str
    error_RAG: str
    SQL_draft: str
    SQL_result: str
    error_SQL: Optional[str]    
    hallucination_check: bool # source를 사용한 답변 생성 후 hallucination 검토 
    # 검토 후 통과한 최종 답변
    final_answer: str 

# 출력 스키마 사용
class OutputState(TypedDict):
    # 검토 후 통과한 최종 답변
    final_answer: str                      # 외부로 반환하는 데이터

class RouteOut(BaseModel):
    route: Literal["policy", "employee", "unclear"]  # 결재규정 / 직원정보 / 불명확
    confidence: float = Field(ge=0, le=1)


# 라우터 체인
# 온도 0으로 결정성 높이기
router_chain = prompt_router | llm.with_config({'temperature': 0}).with_structured_output(RouteOut)

def router_node(state: AppState) -> Command[Literal["rag_node", "sql_gen_node", "general"]]:
    '''
    사용자 질문을 보고 다음 스텝 분기를 정하는 노드
    '''
    output = router_chain.invoke({"user_question": state['messages'], "messages": state["messages"]})
    # 분기
    path = output.route if output.confidence >= 0.6 else "unclear"
    if path == "policy":
        return Command(
            goto="rag_node",
            update={
                "path_results": path,
            }
        )
    elif path == "employee":
        return Command(
            goto="sql_gen_node",
            update={
                "path_results": path,
            }
        )
    elif path == "unclear":
        return Command(
            goto="general",
            update={
                "path_results": path,
            }
        )

def general(state: AppState) -> AppState:
    return {
        "final_answer":"사내 규정 및 직원 정보를 찾을 수 없습니다."
        }

def sql_gen_node(state: AppState) -> AppState:
    prompt_db_structure= """"""
    # 스키마 순회
    for schema_name in inspector.get_schema_names():
        if schema_name == db_name:
            for table_name in inspector.get_table_names(schema=schema_name):
                # 컬럼 이름과 타입 수집
                columns = inspector.get_columns(table_name, schema=schema_name)
                column_list = [
                    f"{col['name']}"
                    for col in columns
                ]
                prompt_db_structure += f'\n[DB 스키마]\n1.table_name: {table_name}\n2.columns: {column_list}'

    prompt_sql = ChatPromptTemplate.from_messages(
    [
        ("system", f"너는 사용자 질문에 대한 답을 얻기 위해 아래 [DB 스키마]구조의 데이터베이스에서 필요한 데이터를 얻기위한 SQL 문을 만드는 역할이야. 절대 다른 문장을 붙이지 말고, SQL문으로만 답변해.{prompt_db_structure}"),
        ("human", "{user_question}"),
        MessagesPlaceholder("messages"),
    ])
    sql_chain = prompt_sql | llm.with_config({'temperature': 0})
    output = sql_chain.invoke({"user_question": state['messages'], "messages": state["messages"]})
    return {
        "SQL_draft": output.content.split('</think>\n\n')[-1]
    }

def sql_execute_node(state: AppState) -> AppState:
    sql_draft = state["SQL_draft"]
    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql_draft))
            rows = result.fetchall() 
            for row in result:
                print(row)
        return {
        "SQL_result": rows
    }
    except Exception as e:
         return {
        "error_SQL": e
    }
            
def rag_node(state: AppState) -> AppState:
    return {
        "RAG_result": "test"
    }


builder = StateGraph(AppState)
# node
builder.add_node("router", router_node)
builder.add_node("rag_node", rag_node)
builder.add_node("general", general)
builder.add_node("sql_gen_node", sql_gen_node)
builder.add_node("sql_execute_node", sql_execute_node)
# edge
builder.add_edge(START, "router")

# builder.add_conditional_edges(
#     "router",
#     router_node,
#     {
#         "policy": "rag_node",
#         "employee": "sql_gen_node",
#         "unclear": "general"
#     },
# )
builder.add_edge("sql_gen_node", "sql_execute_node")
builder.add_edge("sql_execute_node", END)
builder.add_edge("general", END)
builder.add_edge("rag_node", END)

checkpointer = MemorySaver()

#compile
graph = builder.compile(checkpointer=checkpointer)
# [{"role":"user","content":"김다연은 어느 부서, 어느 팀 사원이니?"}]
# [HumanMessage(content="김다연은 어느 부서, 어느 팀 사원이니?")]
# 실행 예시
out = graph.invoke(
    {"messages": [{"role":"user","content":"김다연은 어느 부서, 어느 팀 사원이니?"}], "error_SQL": None},
    {"configurable": {"thread_id": "t1"}}
)
print(out.keys())
print(out.keys())
print(out['error_SQL'])
print(out['path_results'])
print(out['SQL_draft'])
print(out['SQL_result'])

# 할 일
# 에러가 난 경우, 해당 메세지를 참고해서 다시 sql문을 만드는 사이클 필요
# AI 답변도 message 리스트에 저장하는 부분 추후에 적용 필요