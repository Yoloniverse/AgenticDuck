import sqlite3
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph_supervisor import create_supervisor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, List, Any, Annotated, Literal, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from dotenv import load_dotenv
from prompts import planner_system_prompt_template, router_system_prompt_template, document_search_prompt, hallucination_check_prompt
import os
from typing import Any
import time
import os
import sys
import traceback
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langgraph.store.memory import InMemoryStore
from langgraph.graph.message import add_messages
from langgraph.types import Command
## other libraries
from pydantic import BaseModel, Field
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

# ## should change username, passcode, host, port, database names to real ones.
# DB_URI = "postgresql://user:password@localhost:5432/dbname" 
# checkpointer = PostgresSaver.from_conn_string(DB_URI)
# checkpoint_saver = PostgresSaver(db_uri=DB_URI, table_name="agent_checkpoints")

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
        filename=f"./agent.log",
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode='a')

log_fileHandler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(log_fileHandler)
logger.propagate = False

load_dotenv()
print("LANGGRAPH_AES_KEY =", os.getenv("LANGGRAPH_AES_KEY"))


# 메시지를 최대 3세트(6개)로 제한하는 함수
def add_messages_with_limit(left: list, right: list, max_messages: int = 6) -> list:
    """메시지를 추가하되, 최대 메시지 수를 제한"""
    # 기본 add_messages 동작 수행
    combined = add_messages(left, right)
    
    # 최근 max_messages개의 메시지만 유지 (Human-AI 번갈아가는 구조)
    if len(combined) > max_messages:
        return combined[-max_messages:]
    
    return combined
class RAGState(TypedDict, total=False):
    messages: Annotated[list[dict], lambda left, right: add_messages_with_limit(left, right, max_messages=6)]
    # 분기 결과
    path_doc: str
    # for RAG
    rag_docs: List[str]
    rag_error: str
    # source를 사용한 답변 생성 후 hallucination 검토    
    hallucination_check: dict 
    # 검토 후 통과한 최종 답변
    final_answer: str 
    # 사용된 도구 이름들
    tools_used: Optional[List[str]]  
    tool_calls_made: Optional[bool]

class HallucinationState(TypedDict):
    result: bool
    reason: str

class SQLState(TypedDict, total=False):
    messages: Annotated[list[dict], lambda left, right: add_messages_with_limit(left, right, max_messages=6)]
    # for DB search
    sql_db_schema: str
    sql_draft: str
    sql_result: str
    sql_error_cnt: int
    sql_error_node: str
    sql_error: Optional[str] 
    final_answer: str 
    # 사용된 도구 이름들
    tools_used: Optional[List[str]]  
    tool_calls_made: Optional[bool]

class SearchState(TypedDict):
    result: bool

# LLM 모델 load
llm = ChatOllama(model="qwen3:8b", base_url="http://127.0.0.1:11434")

##Sqilite 사용 할 수 있게 하는 코드 
serde = EncryptedSerializer.from_pycryptodome_aes()  # reads LANGGRAPH_AES_KEY
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)
# checkpointer = SqliteSaver.from_file("langgraph_checkpoints.sqlite")

class plannerInputState(TypedDict):  
    task_id: str
    task_description: str
    dependencies: List[str]
    priority: int
class PlannerTasks(TypedDict):
    tasks: List[plannerInputState]

planner_llm_chain = planner_system_prompt_template | llm.with_structured_output(PlannerTasks)

result = planner_llm_chain.invoke("I wanna go to Italy. tell me how to go to italy and what to eat. And also tell me when the best seasons to visit is")
result = planner_llm_chain.invoke("I dont know what to do to find a job in Singapore")
result = planner_llm_chain.invoke("Tell me how many people can get prizes by working over 2 years in my company")

class routerOutputState(TypedDict):  
    agent: str

router_llm_chain = router_system_prompt_template | llm.with_structured_output(routerOutputState)
router_llm_chain.invoke('I wanna make select query for sql for example')
router_llm_chain.invoke('I wanna travel to latin america')
router_llm_chain.invoke('I wanna check facts in my documents')

####################### document_query_agent setting by dykim #######################
# chromadb 상수 정의
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"  # 한국어 특화 임베딩 모델
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Rerank 모델
COLLECTION = "approval_guide" 
top_k = 3 # 프롬프트에 제공할 문서 갯수
embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda:0'},
        encode_kwargs={'normalize_embeddings': True}
    )
reranker = CrossEncoder(RERANK_MODEL)

@tool
def search_company_documents(category: str, query: str) -> Dict[str, Any]:
    """
    회사 문서를 검색합니다.
    Args:
        category: 문서 카테고리 ("인사규정", "전자결재규정", "회사소개" 중 하나)
        query: 검색할 질문이나 키워드
    Returns:
        검색 결과와 상태 정보
    """
    try:
        # 카테고리별 벡터스토어 경로 매핑
        category_mapping = {
            "인사규정": {
                "chroma_dir": "./chroma_db",
                "collection_name": "hr_guide"
            },
            "전자결재규정": {
                "chroma_dir": "./chroma_db", 
                "collection_name": "approval_guide"
            },
            "회사소개": {
                "chroma_dir": "./chroma_db",
                "collection_name": "company_info"
            }
        }
        if category not in category_mapping:
            return {
                "success": False,
                "error": f"지원하지 않는 카테고리입니다: {category}",
                "results": [],
                "category": category
            }
        config = category_mapping[category]
        chroma_dir = config["chroma_dir"]
        collection_name = config["collection_name"]
        
        # 벡터스토어 디렉토리 존재 확인
        if not os.path.exists(chroma_dir):
            return {
                "success": False,
                "error": f"{category} 문서 데이터베이스가 존재하지 않습니다: {chroma_dir}",
                "results": [],
                "category": category
            }
        
        # 벡터스토어 초기화
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=chroma_dir,
            embedding_function=embeddings
        )
        # 문서 검색 수행
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        documents = retriever.get_relevant_documents(query)
        
        # rerank
        query_doc_pairs = [(query, doc.page_content) for doc in documents]
        scores = reranker.predict(query_doc_pairs)
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        # 상위 k개 문서 반환, 결과 포맷팅
        formatted_results = []
        for score, doc in scored_docs[:top_k]:
            formatted_results.append({
                "content": '<Document>' + doc.page_content[:] + '</Document>',
                "metadata": doc.metadata,
                "similarity_score": float(score)
            })
        return {
            "success": True,
            "category": category,
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Error in search_company_documents: {e}")
        return {
            "success": False,
            "error": f"문서 검색 중 오류 발생: {str(e)}",
            "results": [],
            "category": category
        }
# document_query_agent 셋팅
tools = [search_company_documents]
agent = create_tool_calling_agent(llm.with_config({'temperature': 0.5, 'timeout': 15}), tools, document_search_prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=False, 
    return_intermediate_steps=True
)
# for hallucination check
hallucination_chain = hallucination_check_prompt | llm.with_config({'temperature': 0}).with_structured_output(HallucinationState)

def rag_exacute_node(state: RAGState) -> RAGState:
    logger.info(' == [rag_exacute_node] node init == ')
    try:
        user_query = state['messages'][-1].content
        result = agent_executor.invoke({"input": test_query, "messages": state["messages"]})
        path_doc = result['intermediate_steps'][0][0].tool_input['category']
        output_answer = result['output'].split('</think>\n\n')[-1]
        output_documents = [content['content'] for content in result['intermediate_steps'][-1][-1]['results']]
        # hallucination check
        hallucination_check = hallucination_chain.invoke({
            "user_question": user_query,
            "rag_docs": output_documents,
            "final_answer": output_answer
            })
        check_result = hallucination_check['result']
        if check_result:
            output_answer = output_answer + "\n* 이 답변은 정확하지 않은 정보를 포함하고 있는 점 참고바랍니다.\n"
        else:
            pass

        return {
            "path_doc": path_doc,
            "rag_docs": output_documents,
            "rag_error": "None",
            "final_answer": output_answer,
            "hallucination_check": hallucination_check,
            "messages": AIMessage(content=output_answer),

        }
    except Exception as e:
        return {
            "rag_error": "Error in [rag_exacute_node]" + f"{e}"
        }
    

############################## document_query_agent test by dykim ################################
document_query_build = StateGraph(RAGState)
# node
document_query_build.add_node("rag_exacute_node", rag_exacute_node)
# edge
document_query_build.add_edge(START, "rag_exacute_node")
document_query_build.add_edge("rag_exacute_node", END)
checkpointer = MemorySaver()
document_query_agent = document_query_build.compile(checkpointer=checkpointer)

config = {"configurable": {"threa구조d_id": "dayeon"}}
test_query = "국내 출장 시 일비 기준이 어떻게 되나요?"
human_message = {
                    "messages": [{"role":"user","content":test_query}], 
                    "rag_docs": [],
                    "rag_error": None,
                    }

result = document_query_agent.invoke(human_message, config=config)

# visualization
from IPython.display import Image
# PNG 바이트 생성
png_bytes = document_query_agent.get_graph().draw_mermaid_png()
# 파일로 저장
with open("document_query_agent.png", "wb") as f:
    f.write(png_bytes)

####################### db_query_agent setting by dykim #######################
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(connection_string)
connection = engine.connect()
inspector = inspect(engine)
max_gen_sql = 2


############################## db_query_agent test by dykim ################################
db_query_build = StateGraph(SQLState)
# node
db_query_build.add_node("", )
# edge
db_query_build.add_edge(START, "")
db_query_build.add_edge("", END)
checkpointer = MemorySaver()
db_query_agent = db_query_build.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "dayeon"}}
test_query = "김다연님의 근무위치는 어디인가요?"
human_message = {
                    "messages": [{"role":"user","content":test_query}], 
                    "rag_docs": [],
                    "rag_error": None,
                    }

result = db_query_agent.invoke(human_message, config=config)

################################### Architecture ############################################

policy_guidance_team = create_supervisor(
    agents=[document_query_agent, db_query_agent],
    model=llm,
    prompt=policy_guidance_supervisor_prompt
).compile()


research_team = create_supervisor(
    agents=[research_agent],
    model=llm,
    prompt=research_supervisor_prompt
).compile()

top_supervisor = create_supervisor(
    agents=[policy_guidance_team, research_team],
    model=llm,
    pre_model_hook=[planner_agent],
    prompt=(router_system_prompt_template)
).compile()


###########################################################################

##https://docs.langchain.com/oss/python/langchain/short-term-memory#pre-model-hook
##https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": ""
            }
        ]
    }
):
    print(chunk)
    print("\n")