
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
import logging.handlers

#sql node
from sqlalchemy import create_engine, text, inspect
# for rag
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma

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


## Thread (세션)의 대화내용을 저장하기 위한 체크포인터 . 에이전트에 checkpointer를 전달하면서 여러 호출 간 상태 유지(short-term memory)
#checkpointer = InMemorySaver()
## long-term memory를 위한 스토어. 모든 스레드에서 재활용할 수 있는 지식이 필요할 때 씀. 서비스가 running중일 때만 유지됨. 영구 저장은 DB를 사용해야 함.
store = InMemoryStore()
## Ollama LLM 객체 만들기
# ollama pul qwen3:8b 
# 4bit 모델: "qwen3:8b-q4_K_M"
# 8bit 모델: "qwen3:8b-q8_0"
# 16bit 모델 : "qwen3:8b-fp16"
llm = ChatOllama(model="qwen3:8b", base_url="http://127.0.0.1:11434", temperature=0.1)

prompt_router = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 사용자 질문을 읽고 사내 결재 기안 규정에 대한 질문인지, 사내 직원 정보에 대한 질문인지 판단하는 역할이야."),
        ("human", "{user_question}"),
        MessagesPlaceholder("messages")
    ]
)

# 메시지를 최대 3세트(6개)로 제한하는 함수
def add_messages_with_limit(left: list, right: list, max_messages: int = 6) -> list:
    """메시지를 추가하되, 최대 메시지 수를 제한"""
    # 기본 add_messages 동작 수행
    combined = add_messages(left, right)
    
    # 최근 max_messages개의 메시지만 유지 (Human-AI 번갈아가는 구조)
    if len(combined) > max_messages:
        return combined[-max_messages:]
    
    return combined

class AppState(TypedDict, total=False):
    # 대화(필요하면 계속 누적)
    #messages: Annotated[list[dict], add_messages]
    messages: Annotated[list[dict], lambda left, right: add_messages_with_limit(left, right, max_messages=6)]
    # router_node의 분기 결과
    path_results: str
    # for RAG
    rag_retrieved_docs: List[str]
    rag_reranked_docs: List[str]
    rag_check_cnt: int
    rag_error: str
    # for DB search
    sql_db_schema:  str
    sql_draft: str
    sql_result: str
    sql_error_cnt: int
    sql_error_node: str
    sql_error: Optional[str]    
    hallucination_check: bool # source를 사용한 답변 생성 후 hallucination 검토 
    # 검토 후 통과한 최종 답변
    final_answer: str 

# 출력 스키마 사용
class OutputState(TypedDict):
    # 검토 후 통과한 최종 답변
    final_answer: str # 외부로 반환하는 데이터

class RouteOut(BaseModel):
    route: Literal["policy", "employee", "unclear"]  # 결재규정 / 직원정보 / 불명확
    confidence: float = Field(ge=0, le=1)

class HallucinationState(TypedDict):
    result: bool

# 라우터 체인
# 온도 0으로 결정성 높이기
router_chain = prompt_router | llm.with_config({'temperature': 0}).with_structured_output(RouteOut)

def router_node(state: AppState) -> Command[Literal["rag_init_node", "get_schema", "general"]]:
    '''
    사용자 질문을 보고 다음 스텝 분기를 정하는 노드
    '''
    output = router_chain.invoke({"user_question": state['messages'][-1].content, "messages": state["messages"]})
    # 분기
    path = output.route if output.confidence >= 0.6 else "unclear"
    if path == "policy":
        return Command(
            goto="rag_init_node",
            update={
                "path_results": path,
            }
        )
    elif path == "employee":
        return Command(
            goto="get_schema",
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
    prompt_gen = ChatPromptTemplate.from_messages(
    [
        ("system", f"""너는 사용자의 일반적인 질문에 답변하는 Assistant야."""),
        ("human", "{user_question}"),
        MessagesPlaceholder("messages"),
    ])
    gen_chain = prompt_gen | llm.with_config({'temperature': 0.5})
    output = gen_chain.invoke({"user_question": state['messages'][-1].content, "messages": state["messages"]})
    return {
        "messages": AIMessage(content=output.content.split('</think>\n\n')[-1]),
        "final_answer": output.content.split('</think>\n\n')[-1]
        }

class SQLManager:
    def __init__(self):
        ## db 연결
        self.db_user = "admin"
        self.db_password = "sdt251327"
        self.db_host = "127.0.0.1"
        self.db_name = "langgraph" 
        self.connection_string = f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}"
        self.engine = create_engine(self.connection_string)
        self.connection = self.engine.connect()
        self.inspector = inspect(self.engine)


    def get_schema(self, state: AppState) -> AppState:
        try:
            db_structure= """"""
            # 스키마 순회
            for schema_name in self.inspector.get_schema_names():
                if schema_name == db_name:
                    for idx, table_name in enumerate(self.inspector.get_table_names(schema=schema_name)):
                        # 컬럼 이름과 타입 수집
                        columns = self.inspector.get_columns(table_name, schema=schema_name)
                        column_list = [
                            f"{col['name']}"
                            for col in columns
                        ]
                        db_structure += f'\n[DB 스키마]\n{idx}.table_name: {table_name}\n{idx}.columns: {column_list}'
            return {
                "sql_db_schema": db_structure
            }
        except Exception as e:
            return {
            "sql_error_node": "get_schema",
            "sql_error": e
            }

    def sql_gen_node(self, state: AppState) -> Command[Literal["sql_execute_node", END]]:
        try:
            # error 메세지가 있으면 참고해서 생성
            sql_error = state["sql_error"]
            sql_error_cnt = state["sql_error_cnt"]
            print(f"sql_error_cnt: {sql_error_cnt}")

            if sql_error == None:
                prompt_sql = ChatPromptTemplate.from_messages(
                [
                    ("system", f"""너는 사용자 질문에 대한 답을 얻기 위해 아래 [DB 스키마]구조의 데이터베이스에서 필요한 데이터를 얻기위한 SQL 문을 만드는 역할이야. 
                                가장 답변을 잘 이끌어 낼 수있는 SQL 조건이 무엇일지 step by step으로 충분히 생각해. 절대 다른 문장을 붙이지 말고, SQL문으로만 답변해.
                                \n[DB 스키마]\n
                                {state['sql_db_schema']}"""),
                    ("human", "{user_question}"),
                ])
                sql_chain = prompt_sql | llm.with_config({'temperature': 0})
                output = sql_chain.invoke({"user_question": state['messages'][-1].content})
                return Command(
                    goto="sql_execute_node",
                    update={
                        "sql_draft": output.content.split('</think>\n\n')[-1]
                    }
                )

            # 에러 발생하여 최대 3번 다시 시도
            elif sql_error != None and sql_error_cnt < 4:
                print(f"sql_draft: {state['sql_draft']}")
                prompt_sql = ChatPromptTemplate.from_messages(
                [
                    ("system", f"""너는 사용자 질문에 대한 답을 얻기 위해 아래 [DB 스키마]구조의 데이터베이스에서 필요한 데이터를 얻기위한 SQL 문을 만드는 역할이야. 
                                너는 방금 잘못된 SQL을 만들어서 에러가 발생했어. 아래 사항들 참고해서 올바른 SQL문을 다시 만들어.
                                절대 다른 문장을 붙이지 말고, SQL문으로만 답변해.
                                
                                [중요한 제약사항]
                                - 사용자 질문: {state['messages'][-1].content}
                                - 이전에 실패한 SQL: {state['sql_draft']}
                                - 발생한 에러: {state['sql_error']}
                                - 이번은 {sql_error_cnt}번째 시도입니다.
                                - 절대 이전에 실패한 SQL과 같은 구조를 반복하지 말고, 완전히 다른 방식으로 접근해.
                                - SQL문으로만 답변해.

                                [DB 스키마]
                                {state['sql_db_schema']}"""),
                    ("human", "{user_question}"),
                ])
                sql_chain = prompt_sql | llm.with_config({'temperature': 0.1 * sql_error_cnt})
                output = sql_chain.invoke({"user_question": state['messages'][-1].content}, config={"timeout": 30})
                return Command(
                    goto="sql_execute_node",
                    update={
                        "sql_draft": output.content.split('</think>\n\n')[-1]
                    }
                )

            elif sql_error != None and sql_error_cnt == 4:
                return Command(
                    goto=END,
                    update={
                        "final_answer": "데이터 검색에 실패하였습니다. 다시 질문해주십시오.",
                    }
                )

        except Exception as e:
            return {
                "sql_error_node": "sql_gen_node",
                "sql_error": e
            }

    def sql_execute_node(self, state: AppState) -> Command[Literal["sql_gen_node", "sql_final_answer_gen"]]:
        sql_draft = state["sql_draft"]
        try:
            with engine.connect() as self.connection:
                result = self.connection.execute(text(sql_draft))
                rows = result.fetchall() 
                # for row in result:
                #     print(row)
            if not rows or None in rows:
                raise Exception("EMPTY_RESULT") 
                
            # 빈 결과도 에러로 처리 
            else:
                return Command(
                goto="sql_final_answer_gen",
                update={
                    "sql_result": rows,
                    "sql_error": None,
                    }
                )

        # 실행 에러 발생 시 sql_gen_node 노드로 돌아가며, error 메시지를 저장하고 error count를 올림.
        except Exception as e:
            return Command(
                goto="sql_gen_node",
                update={
                    "sql_error_node": "sql_execute_node",
                    "sql_error": str(e),
                    "sql_error_cnt": state['sql_error_cnt'] + 1
                    }
                )

    def sql_final_answer_gen(self,state: AppState) -> AppState:
        sql_result = state['sql_result']
        sql_draft = state['sql_draft']
        prompt_final = ChatPromptTemplate.from_messages(
        [
            ("system", f"""너는 사용자 질문에 대한 답변을 생성하는 역할이야. 아래 [정보]는 사용자의 질문을 기반으로 [SQL문]으로 필요한 정보를 DB에서 조회한 결과야. 정보에 없는 내용을 덧붙이거나 변형하여 환각을 일으키지 마.
                        \n[SQL문]\n {sql_draft}
                        \n[정보]\n {sql_result}"""),
            ("human", "{user_question}"),
            MessagesPlaceholder("messages")
        ])
        final_chain = prompt_final | llm.with_config({'temperature': 0})
        output = final_chain.invoke({"user_question": state['messages'][-1].content, "messages": state["messages"]})
        return {
            "messages": AIMessage(content=output.content.split('</think>\n\n')[-1]),
            "final_answer": output.content.split('</think>\n\n')[-1]
        }

class RAGManager:
    def __init__(self):
        # chromadb 상수 정의
        self.CHROMA_DIR = "./chroma_db"
        self.EMBEDDING_MODEL = "BAAI/bge-m3"  # 한국어 특화 임베딩 모델
        self.RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Rerank 모델
        self.COLLECTION = "approval_guide" 
        self.top_k = 3 # 프롬프트에 제공할 문서 갯수
        self.embeddings = None
        self.reranker = None

    def rag_init_node(self, state: AppState) -> Command[Literal["rag_execute_node", END]]:
        try:
            if self.embeddings and self.reranker:
                pass
            else:
                self.embeddings = HuggingFaceEmbeddings(
                        model_name=self.EMBEDDING_MODEL,
                        model_kwargs={'device': 'cuda:0'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                self.reranker = CrossEncoder(self.RERANK_MODEL)

            # 기존 DB가 존재하는지 확인
            if os.path.exists(self.CHROMA_DIR):
                self.vectorstore = Chroma(
                    collection_name=self.COLLECTION,
                    persist_directory=self.CHROMA_DIR,
                    embedding_function=self.embeddings
                )

                return Command(
                    goto="rag_execute_node",
                    update={
                        "rag_error": None,
                    }
                )
            else:
                return Command(
                    goto=END,
                    update={
                        "rag_error": "vectorestore initialize 실패",
                    }
                )
        except Exception as e:
            print(f"error in initialize_RAG_comp: {e}")
            return Command(
                    goto=END,
                    update={
                        "rag_error": "initialize_RAG_component 실패",
                    }
                )
            

    def rag_execute_node(self, state: AppState) -> AppState:
        # retrive
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        documents = self.retriever.get_relevant_documents(state['messages'][-1].content)
        # rerank
        query_doc_pairs = [(state['messages'][-1].content, doc.page_content) for doc in documents]
        scores = self.reranker.predict(query_doc_pairs)
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        # 상위 k개 문서 반환
        reranked_docs = ['<Document>' + doc.page_content[:] + '</Document>' for score, doc in scored_docs[:self.top_k]]

        return {
            "rag_retrieved_docs": documents,
            "rag_reranked_docs": reranked_docs
        }

    def rag_final_answer_gen(self, state: AppState) -> AppState:
        rag_reranked_docs = state['rag_reranked_docs']

        prompt_final = ChatPromptTemplate.from_messages(
        [
            ("system", f"""너는 사용자 질문에 대한 답변을 생성하는 역할이야. 
                        \n[조건]\n 아래 [정보]는 사용자의 질문을 기반으로 필요한 vectorDB에서 조회한 결과 문서야. 모르는 부분은 모른다고 대답하고, 정보에 없는 내용을 덧붙이거나 변형하여 환각을 일으키지 마.
                        \n[정보]\n {rag_reranked_docs}"""),
            ("human", "{user_question}"),
            MessagesPlaceholder("messages")
        ])
        final_chain = prompt_final | llm.with_config({'temperature': 0})
        output = final_chain.invoke({"user_question": state['messages'][-1].content, "messages": state["messages"]})
        return {
            "messages": AIMessage(content=output.content.split('</think>\n\n')[-1]),
            "final_answer": output.content.split('</think>\n\n')[-1]
        }

    def hallucination_check(self, state: AppState) -> AppState:
        rag_reranked_docs = state['rag_reranked_docs']
        final_answer = state['final_answer']

        prompt_check = ChatPromptTemplate.from_messages(
        [
            ("system", f"""너는 RAG(Retrieval-Augmented Generation)결과물인 [문서 정보]과 LLM 모델이 생성한 [생성 답변]을 비교하여, [생성 답변]에 [문서 정보]에 없는 내용이 포함되어있는지 hallucination 여부를 판단하는 역할이야.
                        \n[조건]\n : hallucination 발생 시 True, 없을 시 False를 반환하시오. 다른 문장 없이 무조건 True 또는 False로만 답하세요.
                        \n[문서 정보]\n {rag_reranked_docs}
                        \n[생성 답변]\n {final_answer}
                        """),
            ("human", "{user_question}")
        ])
        final_chain = prompt_check | llm.with_config({'temperature': 0}).with_structured_output(HallucinationState)
        output = final_chain.invoke({"user_question": state['messages'][-1].content})
        check_result = output['result']

        # hallucination 발생
        if check_result:
            return {
                        "hallucination_check": check_result,
                        "final_answer": state["final_answer"] + "\n* 이 답변은 정확하지 않은 정보를 포함하고 있는 점 참고바랍니다.\n"
                    }
        # hallucination 발생하지 않아서 종료
        else:
            return {
                        "hallucination_check": check_result,
                    }

rag_manager = RAGManager()
sql_manager = SQLManager()

builder = StateGraph(AppState)
# node
builder.add_node("router", router_node)
builder.add_node("general", general)
builder.add_node("get_schema", sql_manager.get_schema)
builder.add_node("sql_gen_node", sql_manager.sql_gen_node)
builder.add_node("sql_execute_node", sql_manager.sql_execute_node)
builder.add_node("sql_final_answer_gen", sql_manager.sql_final_answer_gen)
builder.add_node("rag_init_node", rag_manager.rag_init_node)
builder.add_node("rag_execute_node", rag_manager.rag_execute_node)
builder.add_node("rag_final_answer_gen", rag_manager.rag_final_answer_gen)
builder.add_node("hallucination_check", rag_manager.hallucination_check)
# builder.add_conditional_edges(
#     "router",
#     router_node,
#     {
#         "policy": "rag_node",
#         "employee": "sql_gen_node",
#         "unclear": "general"
#     },
# )
builder.add_edge(START, "router")
builder.add_edge("get_schema", "sql_gen_node")
builder.add_edge("sql_final_answer_gen", END)
builder.add_edge("general", END)
builder.add_edge("rag_execute_node", "rag_final_answer_gen")
builder.add_edge("rag_final_answer_gen", "hallucination_check")
builder.add_edge("hallucination_check", END)

checkpointer = MemorySaver()

#compile
graph = builder.compile(checkpointer=checkpointer)

"""
######## stream으로 실행 ########
# config 설정(재귀 최대 횟수, thread_id)
config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})
# 질문 입력
inputs = {"messages": [{"role":"user","content":"외근 교통비 청구방법 알려줘"}], "sql_error": None, "rag_error": None, "sql_error_cnt": 0, "rag_check_cnt":0, "sql_error_node": "none"}
# 그래프 실행
#print(graph.get_graph().nodes.keys())
prev_node = ""
node_names=['rag_final_answer_gen','hallucination_check']
for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
    curr_node = metadata["langgraph_node"]
    if not node_names or curr_node in node_names:
        # 노드가 변경된 경우에만 구분선 출력
        if curr_node != prev_node:
            print("\n" + "=" * 50)
            print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
            print("- " * 25)
        print(chunk_msg.content, end="", flush=True)

        prev_node = curr_node
"""

# 실행 예시
# out_1 = graph.invoke(
#     {"messages": [{"role":"user","content":"김다연은 어느 부서, 어느 팀 사원이니?"}], "sql_error": None, "sql_error_cnt": 0, "sql_error_node": "none"},
#     {"configurable": {"thread_id": "t1"}}
# )
# print(out_1)
# print(out_1['sql_draft'])
# print(out_1['sql_result'])
# print(out_1['final_answer'])

# # 실행 예시
out = graph.invoke(
    {"messages": [{"role":"user","content":"QX 개발실에서 일하는 사람 이름을 다 알려줘"}], "sql_error": None, "sql_error_cnt": 0, "sql_error_node": "none"},
    {"configurable": {"thread_id": "t1"}}
)
print(out['final_answer'])

# 실행 예시 RAG
out_2 = graph.invoke(
    {"messages": [{"role":"user","content":"외근 교통비 청구방법 알려줘"}], "sql_error": None, "rag_error": None, "sql_error_cnt": 0, "rag_check_cnt":0, "sql_error_node": "none"},
    {"configurable": {"thread_id": {"thread_id": "t1"}}}
)
#print(out_2)
print(out_2['final_answer'])


# 실행 예시 RAG
out_3 = graph.invoke(
    {"messages": [{"role":"user","content":"나는 쏘카 안썼는데, 외근 교통비 청구 어떻게 하니?"}], "sql_error": None, "rag_error": None, "sql_error_cnt": 0, "rag_check_cnt":0, "sql_error_node": "none"},
    {"configurable": {"thread_id": "t1"}}
)
print(out_3['final_answer'])


#graph visualization
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e :
    # This requires some extra dependencies and is optional
    print(e)
    pass

# from langgraph.graph import draw_mermaid_png, MermaidDrawMethod

# # 로컬 브라우저로 렌더링
# draw_mermaid_png(
#     graph, 
#     draw_method=MermaidDrawMethod.PYPPETEER  # 로컬 렌더링
# )


# from langgraph.graph import MermaidDrawMethod

# png = graph.get_graph().draw_mermaid_png(
#     draw_method=MermaidDrawMethod.PYPPETEER,  # 원격(API) 안 쓰고 로컬에서 렌더
#     max_retries=5, retry_delay=2.0
# )


# from IPython.display import Image, display
# from langchain_core.runnables.graph import  MermaidDrawMethod
# display(Image(graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER,)))


#################333 teddynote stream ###############
# from langchain_teddynote.messages import stream_graph, random_uuid
# from langchain_core.runnables import RunnableConfig

# # config 설정(재귀 최대 횟수, thread_id)
# config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

# # 질문 입력
# inputs = {"messages": [{"role":"user","content":"외근 교통비 청구방법 알려줘"}], "sql_error": None, "rag_error": None, "sql_error_cnt": 0, "rag_check_cnt":0, "sql_error_node": "none"}


# # 그래프 실행
# stream_graph(graph, inputs, config,["rag_final_answer_gen", "hallucination_check"])

