import time
import streamlit as st
import uuid
import os
import sys
import traceback
## Langchain libraries
#from langchain.llms import Ollama
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
#from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_ollama import ChatOllama
#from langmem.short_term import SummarizationNode
#from langchain_core.messages.utils import count_tokens_approximately
#from langgraph.prebuilt import create_react_agent

# from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, AnyMessage

## LangGraph libraries
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

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
# tools
sys.path.append("/home/sdt/Workspace/dykim/Langraph/AgenticDuck")
from toolings import taviliy_web_search_tool 

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
logger.handlers.clear()
logger.addHandler(log_fileHandler)
logger.propagate = False


# Streamlit 페이지 설정
st.set_page_config(
    page_title="HR 챗봇",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)
st.title("🤖 HR 챗봇")
st.markdown("사내 결재 규정과 직원 정보에 대해 궁금한 것을 물어보세요!")
# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# "qwen3:8b-fp16" / "qwen3:8b"
@st.cache_resource
def initialize_graph():
    logger.info("채팅 초기화")
    llm = ChatOllama(model="qwen3:8b-fp16", base_url="http://127.0.0.1:11434")

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
        messages: Annotated[list[dict], lambda left, right: add_messages_with_limit(left, right, max_messages=6)]
        query: str
        # router_node의 분기 결과
        path_results: str
        web_results : List[str]
        query_rewrite: Optional[bool]
        # for RAG
        rag_retrieved_docs: List[str]
        rag_reranked_docs: List[str]
        rag_check_cnt: int
        rag_error: str
        # for DB search
        sql_db_schema: str
        sql_draft: str
        sql_result: str
        sql_error_cnt: int
        sql_error_node: str
        sql_error: Optional[str] 
        # source를 사용한 답변 생성 후 hallucination 검토    
        hallucination_check: dict 
        # 검토 후 통과한 최종 답변
        final_answer: str 
        # 사용된 도구 이름들
        tools_used: Optional[List[str]]  
        tool_calls_made: Optional[bool]

    # 출력 스키마 사용
    class OutputState(TypedDict):
        # 검토 후 통과한 최종 답변
        final_answer: str # 외부로 반환하는 데이터


    class RouteOut(BaseModel):
        route: Literal["policy", "employee", "general"]  # 결재규정 / 직원정보 / 불명확
        confidence: float = Field(ge=0, le=1)

    class HallucinationState(TypedDict):
        result: bool
        reason: str
    class SearchState(TypedDict):
        result: bool


    # 라우터 체인
    # 온도 0으로 결정성 높이기
    prompt_router = ChatPromptTemplate.from_messages(
        [
            ("system", """너는 사용자 질문을 읽고 분석하여 목적에 맞게 분류하는 역할이야.
                        [중요 원칙]
                            - 질문자는 회사 직원이기 때문에 기본적으로 사내 정보에 대한 질문을 한다는 것을 명심해.
                            - 사내 정보에 대한 질문인데 일반적인 질문이라고 착각하지 않도록 충분히 생각해.
                            - 사용자가 질문을 통해 얻고싶은 정보가 무엇인지를 핵심으로 중요하게 생각해.
                        [조건]
                            - "policy" = 사내 인사 규정, 근무 규정, 업무 프로세스, 결재/기안 작성 규정, 출장비·경비 처리, 휴가·근태, 보고서 제출, 전결 규정 등 회사 규정이나 제도와 관련된 질문  
                            - "employee" = 사내 직원 개인의 이름, 연락처, 부서, 직급, 담당 업무 등 인사/조직 정보 관련 질문
                            - "general" = "policy", "employee" 범주에 해당되지 않고, 사내 문서를 참고하지 않고 답변할 수 있는 일반적인 질문
                            - "answer_unavailable" = 사용자의 질문이 1~4번에 해당되는 질문
                                1. 시스템 내부 정보: 서버 주소, 데이터베이스 접근 정보, API 키, 토큰, 비밀번호 등 시스템 보안과 관련된 정보. 내부 네트워크 구조, 로그, 소스코드, 모델 파라미터, 운영 인프라 세부사항
                                2. 개인정보 및 민감 데이터: 주민등록번호, 계좌번호, 급여 내역, 인사평가, 채용 심사 결과 등 민감한 개인 신상 정보. 특정 직원의 사적인 생활, 개인 기록, 비공개 건강·재무 정보
                                3. 보안/정책상 제공 불가한 요청: 회사의 보안 규정, 미공개 사업 전략, 계약 내용, 법적 분쟁 자료. 공개가 금지된 기밀문서나 내부 문건 요청
                                4. 기타: 외부 공개가 허용되지 않은 내부 시스템 데이터 및 로그.
                            """),
            ("human", "{user_question}"),
        ]
    )
    router_chain = prompt_router | llm.with_config({'temperature': 0.1}).with_structured_output(RouteOut)

    def router_node(state: AppState) -> Command[Literal["rag_init_node", "get_schema", "general", "security_filter"]]:
        '''
        사용자 질문을 보고 다음 스텝 분기를 정하는 노드
        '''
        logger.info(' == [router_node] node init == ')
        
        output = router_chain.invoke({
            "user_question": state['messages'][-1].content
            })
        # 분기
        #path = output.route if output.confidence >= 0.6 else "general"
        path = output.route
        logger.info(f"path : {path}")
        if path == "policy":
            return Command(
                goto="rag_init_node",
                update={
                    "path_results": path,
                    "query": state['messages'][-1].content
                }
            )
        elif path == "employee":
            return Command(
                goto="get_schema",
                update={
                    "path_results": path,
                    "query": state['messages'][-1].content
                }
            )
        elif path == "general":
            return Command(
                goto="general",
                update={
                    "path_results": path,
                    "query": state['messages'][-1].content
                }
            )
        elif path == "security_filter":
            return Command(
                goto="general",
                update={
                    "path_results": path,
                    "query": state['messages'][-1].content
                }
            )

    tools = [taviliy_web_search_tool]
    prompt_gen = ChatPromptTemplate.from_messages(
        [
            ("system", f"""너는 사용자의 일반적인 질문에 답변하는 Assistant야. 만약 웹 서치가 필요한 질문이라면 제공된 web search tool을 사용하고, 아닌 경우 바로 답변을 생성해.
                        [web search가 필요한 경우 예시]
                         - 실시간 데이터에 접근해야 하는 경우
                         - 최신 뉴스, 주식 가격, 날씨, 날짜
                         - LLM이 학습하지 않아 검색이 필요한 데이터
                        """),
            ("human", "{user_question}"),
            MessagesPlaceholder("messages"),
            ("placeholder", "{agent_scratchpad}"),
        ])
    agent = create_tool_calling_agent(llm.with_config({'temperature': 0.3}), tools, prompt_gen)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True) 
    
    def general(state: AppState) -> Command[Literal["query_rewrite", END]]:
        logger.info(' == [general] node init == ')
        #output = agent_executor.invoke({"user_question": "오늘 한국은 며칠이야?", "messages": ["messages"]})
        output = agent_executor.invoke({"user_question": state['query'], "messages": state["messages"]})
        tools_used = []
        search_contents = []
        if 'intermediate_steps' in output and output['intermediate_steps']:
            tool_calls_made = True
            for step in output['intermediate_steps']:
                if len(step) >= 2:
                    action, result = step[0], step[1]
                    if hasattr(action, 'tool'):
                        tools_used.append(action.tool)
                    if isinstance(result, dict) and 'results' in result:
                        for search_result in result['results']:
                            if 'content' in search_result and search_result['content']:
                                search_contents.append(search_result['content'].replace("{","(").replace("}",")"))
        else:
            tool_calls_made = False

        logger.info(f"tool_calls_made: {tool_calls_made}, tools_used: {tools_used}")

        # 웹 검색 tool을 사용한 경우 검색 결과 검증
        if tool_calls_made and  "taviliy_web_search_tool" in tools_used:
            prompt_search_check = ChatPromptTemplate.from_messages(
            [
                ("system", f"""너는 사용자 질문과 [웹 검색 결과]를 비교하여 검색 결과가 사용자 질문에 대한 답변을 생성하는데 적합한지 판단하는 역할이야.
                            [조건]
                            - 다른 문장 붙이지 말고 True 또는 False로만 대답해.
                            - 사용자 질문에 대한 답변으로 적합함 = True
                            - 사용자 질문에 대한 답변으로 부족합 = False 
                            [웹 검색 결과]
                            {search_contents[0]}
                            {search_contents[1]}
                            {search_contents[2]}
                            """),
                ("human", "사용자 질문: {user_question}, 웹에서 검색된 결과가 사용자 질문에대한 답변으로 적합한지 판단해."),
            ])
            # .with_structured_output(SearchState)를 사용하면 적합하지 않은 결과를 적합하다고 판단, 사용하지않으면 정확하게 잘 판단함
            check_chain = prompt_search_check | llm.with_config({'temperature': 0.2})
            output_check = check_chain.invoke({"user_question": state['messages'][-1].content})
            result = output_check.content.split('</think>\n\n')[-1]

            # 웹 검색 결과가 답변하기 충분하여 종료
            if result == 'True':
                return Command(
                            goto=END,
                            update={
                            "messages": AIMessage(content=output['output'].split('</think>\n\n')[-1]),
                            "final_answer": output['output'].split('</think>\n\n')[-1],
                            "tools_used": list(set(tools_used)),
                            "tool_calls_made": tool_calls_made,
                            "query_rewrite": False,
                            }
                    )
            # 웹 검색 결과가 적합하지 않아 쿼리를 재작성
            else:
                return Command(
                            goto="query_rewrite",
                            update={
                                "query_rewrite": True,
                                "web_results": search_contents
                            }
                        )
        else:
            return Command(
                            goto=END,
                            update={
                            "messages": AIMessage(content=output['output'].split('</think>\n\n')[-1]),
                            "final_answer": output['output'].split('</think>\n\n')[-1],
                            "tools_used": list(set(tools_used)),
                            "tool_calls_made": tool_calls_made
                            }
                    )

    def query_rewrite(state: AppState) -> AppState:
        logger.info(' == [query_rewrite] node init == ')
        web_results = state['web_results']
        prompt_rewrite = ChatPromptTemplate.from_messages(
        [
            ("system", f"""You a question re-writer that converts an input question to a better version that is optimized for web searches.
                        [조건]
                         - Look at the input and try to reason about the underlying semantic intent / meaning.
                         - 간결하게 답변하고 한국어로 답변해.
                         - [이전 검색 결과]를 참고해서, 이런 결과가 안나올 수 있는 질문으로 생성해.
                        [이전 검색 결과]
                         - {web_results[0]}
                         - {web_results[1]}
                         - {web_results[2]}
                        """),
            ("human", "원래 사용자 질문: {user_question}, 웹 검색을 위해 개선된 질문을 생성해."),
        ])
        rewrite_chain = prompt_rewrite | llm.with_config({'temperature': 0.5})
        output = rewrite_chain.invoke({"user_question": state['messages'][-1].content})
        return {
                "query": output.content.split('</think>\n\n')[-1],
                }
                    

    def security_filter(state: AppState) -> AppState:
        logger.info(' == [security_filter] node init == ')
        prompt_security = ChatPromptTemplate.from_messages(
        [
            ("system", f"""너는 사용자의 질문을 참고하여, 민감 정보 접근으로 인한 답변 불가능을 설명하는 역할이야.
                        [조건]
                         - 민감한 정보에 대한 사용자 질문을 참고하여, 그 질문에 대해 왜 답변할 수 없는지 설명해.
                         - 간결하게 답변하고 한국어로 답변해.
                        """),
            ("human", "{user_question}"),
        ])
        security_chain = prompt_gen | llm.with_config({'temperature': 0.5})
        output = security_chain.invoke({"user_question": state['messages'][-1].content})
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
            self.max_gen_sql = 2


        def get_schema(self, state: AppState) -> AppState:
                logger.info(' == [get_schema] node init == ')
                get_schema_result = None
                max_retries = 3
                retry_delay = 1
                for attempt in range(max_retries):
                    try:
                        db_structure= """"""
                        schema_found = False
                        # 스키마 순회
                        for schema_name in self.inspector.get_schema_names():
                            if schema_name == self.db_name:
                                schema_found = True
                                logger.info(f'스키마 "{schema_name}" 발견됨')
                                tables = self.inspector.get_table_names(schema=schema_name)
                                for idx, table_name in enumerate(tables):
                                    # 컬럼 이름과 타입 수집
                                    columns = self.inspector.get_columns(table_name, schema=schema_name)
                                    column_list = [f"{col['name']}" for col in columns]
                                    db_structure += f'\n[DB 스키마]\n{idx}.table_name: {table_name}\n{idx}.columns: {column_list}'
                                break
                        if not schema_found:
                            raise Exception(f'스키마 "{self.db_name}"를 찾을 수 없습니다')
                        logger.info('DB 스키마 가져오기 성공')
                        return {
                            "sql_db_schema": db_structure,
                            "sql_error_node": None,
                            "sql_error": None
                        }

                    except Exception as e:
                        logger.error(f'DB 스키마 가져오기 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}')
                        # 마지막 시도가 아니라면 잠시 대기 후 재시도
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            logger.info(f'{retry_delay}초 후 재시도합니다...')
                        else:
                            # 모든 시도 실패
                            logger.error('모든 재시도 실패. DB 스키마 가져오기를 포기합니다.')
                            return {
                                "sql_db_schema": None,
                                "sql_error_node": "get_schema",
                                "sql_error": f"get_schema 3회 시도 후 실패: {str(e)}"
                            }
                

        def sql_gen_node(self, state: AppState) -> Command[Literal["sql_execute_node", "sql_final_answer_gen"]]:
            logger.info(' == [sql_gen_node] node init == ')
            if state["sql_db_schema"] != None:
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
                                        [중요 원칙]
                                        - 무조건 [DB 스키마]에 존재하는 컬럼만 사용해야 함
                                        - [DB 스키마]에 없는 정보에 대한 질문을 한 경우, 그와 가장 유사한 데이터를 얻을 수 있는 SQL을 생성해.
                                        - 절대 다른 문장을 붙이지 말고, SQL문으로만 답변해야 함
                                        - 여러가지 후보를 생각해보고, 그 중 사용자의 질문에 가장 잘 맞는 문장을 선택해

                                        [DB 스키마]
                                        {state['sql_db_schema']}"""),
                            ("human", "{user_question}"),
                        ])
                        sql_chain = prompt_sql | llm.with_config({'temperature': 0, 'timeout': 10})
                        output = sql_chain.invoke({"user_question": state['messages'][-1].content})
                        return Command(
                            goto="sql_execute_node",
                            update={
                                "sql_draft": output.content.split('</think>\n\n')[-1]
                            }
                        )
                    # 에러 발생하여 최대 3번 다시 시도
                    elif sql_error != None and sql_error_cnt <= self.max_gen_sql:
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
                        sql_chain = prompt_sql | llm.with_config({'temperature': 0.2 * sql_error_cnt, "timeout": 10})
                        output = sql_chain.invoke({"user_question": state['messages'][-1].content})
                        return Command(
                            goto="sql_execute_node",
                            update={
                                "sql_draft": output.content.split('</think>\n\n')[-1]
                            }
                        )
                    else:
                        return Command(
                            goto="sql_final_answer_gen",
                            update={
                                "sql_draft": "none",
                                "sql_result": "결과가 없습니다."
                            }
                        )

                except Exception as e:
                    logger.info(traceback.format_exc())
                    return {
                        "sql_error_node": "sql_gen_node",
                        "sql_error": e
                    }
            else:
                return Command(
                            goto="sql_final_answer_gen",
                            update={
                                "sql_draft": "none",
                                "sql_result": "현재 연결이 불안정하여 데이터를 조회할 수 없습니다. 잠시 뒤에 다시 질문하라고 안내하세요."
                            }
                        )

        def sql_execute_node(self, state: AppState) -> Command[Literal["sql_gen_node", "sql_final_answer_gen"]]:
            logger.info(' == [sql_execute_node] node init == ')
            
            sql_draft = state["sql_draft"]
            try:
                with self.engine.connect() as self.connection:
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
            logger.info(' == [sql_execsql_final_answer_gen] node init == ')

            sql_result = state['sql_result']
            sql_draft = state['sql_draft']
            prompt_final = ChatPromptTemplate.from_messages(
            [
                ("system", f"""너는 질의응답 챗봇 시스템에서 사용자 질문에 대한 최종 답변을 생성하는 역할이야. 
                            [조건]
                             - 아래 [정보]는 사용자의 질문을 기반으로 [SQL문]으로 필요한 정보를 DB에서 조회한 결과야. 정보에 없는 내용을 덧붙이거나 변형하여 환각을 일으키지 마.
                             - DB 스키마, 테이블 등 내부 정보를 사용자에게 노출하지마.
                             - 한국어로 답변해.
                            [SQL문]\n {sql_draft}
                            [정보]\n {sql_result}"""),
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
            logger.info(' == [rag_init_node] node init == ')

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
            logger.info(' == [rag_execute_node] node init == ')

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
            logger.info(' == [rag_final_answer_gen] node init == ')

            rag_reranked_docs = state['rag_reranked_docs']

            prompt_final = ChatPromptTemplate.from_messages(
            [
                ("system", f"""너는 질의응답 챗봇 시스템에서 사용자 질문에 대한 최종 답변을 생성하는 역할이야. 
                            [조건]
                             - 아래 [정보]는 사용자의 질문을 기반으로 문서에서 조회한 결과야. 결과에 없는 부분은 모른다고 대답하고, 정보에 없는 내용을 덧붙이거나 변형하여 환각을 일으키지 마. 
                             - 한국어로 답변해.
                            [정보] \n {rag_reranked_docs}"""),
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
            logger.info(' == [hallucination_check] node init == ')

            rag_reranked_docs = state['rag_reranked_docs']
            final_answer = state['final_answer']

            prompt_check = ChatPromptTemplate.from_messages(
            [
                ("system", f"""너는 RAG(Retrieval-Augmented Generation)결과물인 [문서 정보]과 LLM 모델이 생성한 [생성 답변]을 비교하여, [생성 답변]에 [문서 정보]에 없는 내용이 포함되어있는지 hallucination 여부를 판단하는 역할이야.
                            \n[조건]\n : hallucination 발생 시 True, 없을 시 False를 'result' key에 반환하세요. 그리고 hallucination이 발생했다고 판단한 이유를 'reason' key에 반환하세요.
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
                            "hallucination_check": output,
                            "final_answer": state["final_answer"] + "\n* 이 답변은 정확하지 않은 정보를 포함하고 있는 점 참고바랍니다.\n"
                        }
            # hallucination 발생하지 않아서 종료
            else:
                return {
                            "hallucination_check": output,
                        }

    rag_manager = RAGManager()
    sql_manager = SQLManager()

    builder = StateGraph(AppState)
    # node
    builder.add_node("router", router_node)
    builder.add_node("general", general)
    builder.add_node("query_rewrite", query_rewrite)
    builder.add_node("security_filter", security_filter)
    builder.add_node("get_schema", sql_manager.get_schema)
    builder.add_node("sql_gen_node", sql_manager.sql_gen_node)
    builder.add_node("sql_execute_node", sql_manager.sql_execute_node)
    builder.add_node("sql_final_answer_gen", sql_manager.sql_final_answer_gen)
    builder.add_node("rag_init_node", rag_manager.rag_init_node)
    builder.add_node("rag_execute_node", rag_manager.rag_execute_node)
    builder.add_node("rag_final_answer_gen", rag_manager.rag_final_answer_gen)
    builder.add_node("hallucination_check", rag_manager.hallucination_check)

    # edge
    builder.add_edge(START, "router")
    builder.add_edge("get_schema", "sql_gen_node")
    builder.add_edge("sql_final_answer_gen", END)
    builder.add_edge("query_rewrite", "general")
    builder.add_edge("rag_execute_node", "rag_final_answer_gen")
    builder.add_edge("rag_final_answer_gen", "hallucination_check")
    builder.add_edge("hallucination_check", END)
    builder.add_edge("security_filter", END)

    checkpointer = MemorySaver()
    store = InMemoryStore()
    #compile
    graph = builder.compile(checkpointer=checkpointer)

    # # visualization
    # from IPython.display import Image
    # # PNG 바이트 생성
    # png_bytes = graph.get_graph().draw_mermaid_png()
    # # 파일로 저장
    # with open("graph.png", "wb") as f:
    #     f.write(png_bytes)

    return graph


# 그래프 초기화
try:
    graph = initialize_graph()
    st.success("✅ 챗봇이 성공적으로 초기화되었습니다!")
except Exception as e:
    st.error(f"❌ 챗봇 초기화 실패: {str(e)}")
    st.info("Ollama 서버가 실행 중인지 확인해주세요.")

# 채팅 히스토리 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if "execution_count" not in st.session_state:
    st.session_state.execution_count = 0

# 사용자 입력
if prompt := st.chat_input("궁금한 것을 물어보세요!"):
    #logger.info(f"사용자 입력 : {prompt}")
    timestamp = time.time()
    st.session_state.execution_count += 1
    logger.info(f"[세션ID: {st.session_state.thread_id}] - [{st.session_state.execution_count}번째 실행] 실행 시점: {timestamp} - 사용자 입력: {prompt}")
    
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # 챗봇 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            try:
                # LangGraph에 메시지 전송
                #human_message = HumanMessage(content=prompt)
                human_message = {
                    "messages": [{"role":"user","content":prompt}], 
                    "sql_draft": "",
                    "sql_result": "",
                    "sql_error": None,
                    "sql_error_cnt": 0,
                    "sql_error_node": "none",
                    "rag_check_cnt":0,
                    "rag_retrieved_docs": [],
                    "rag_reranked_docs": [],
                    "rag_error": None,
                    }
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # 그래프 실행
                result = graph.invoke(
                    human_message, 
                    config=config
                )
                logger.info(f"AppState : {result}")
                
                # 결과에서 답변 추출
                if "final_answer" in result and result["final_answer"]:
                    response = result["final_answer"]
                elif result["messages"]:
                    response = result["messages"][-1].content
                else:
                    response = "죄송합니다. 답변을 생성할 수 없습니다."
                
                st.write(response)
                
                # 응답을 채팅 히스토리에 추가
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                logger.info(f"AI 답변 : {response}")
                
            except Exception as e:
                error_message = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
                st.error(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})

# 사이드바에 추가 정보
with st.sidebar:
    st.header("ℹ️ 정보")
    st.markdown("""
    **사용 가능한 질문 유형:**
    - 📋 사내 결재 규정 관련 질문
    - 👥 사내 직원 정보 관련 질문
    - 💬 기타 일반적인 질문
    
    **사용법:**
    1. 아래 입력창에 질문을 입력하세요
    2. 챗봇이 자동으로 질문 유형을 분류합니다
    3. 적절한 정보를 검색하여 답변합니다
    """)
    
    if st.button("🔄 채팅 초기화"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**연결 상태:**")
    if "graph" in locals():
        st.success("✅ 연결됨")
    else:
        st.error("❌ 연결 안됨")
