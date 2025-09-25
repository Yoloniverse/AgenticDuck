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
from prompts import planner_system_prompt_template, router_system_prompt_template, document_search_prompt
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

load_dotenv()
print("LANGGRAPH_AES_KEY =", os.getenv("LANGGRAPH_AES_KEY"))

class RAGState(TypedDict, total=False):
    messages: Annotated[list[dict], lambda left, right: add_messages_with_limit(left, right, max_messages=6)]
    query: str
    # 분기 결과
    path_results: str
    # for RAG
    rag_retrieved_docs: List[str]
    rag_reranked_docs: List[str]
    rag_check_cnt: int
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
class SearchState(TypedDict):
    result: bool
##Sqilite 사용 할 수 있게 하는 코드 
serde = EncryptedSerializer.from_pycryptodome_aes()  # reads LANGGRAPH_AES_KEY
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)

llm = ChatOllama(model="qwen3:8b", base_url="http://127.0.0.1:11434")
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


####################### RAG agent test by dykim #######################

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
# final answer chain
prompt_final = ChatPromptTemplate.from_messages(
[
    ("system", """너는 질의응답 챗봇 시스템에서 사용자 질문에 대한 최종 답변을 생성하는 역할이야. 
                [조건]
                    - 아래 [정보]는 사용자의 질문을 기반으로 문서에서 조회한 결과야. 결과에 없는 부분은 모른다고 대답하고, 정보에 없는 내용을 덧붙이거나 변형하여 환각을 일으키지 마. 
                    - 사용자는 내부적으로 어떤 로직에 의해 답변을 생성하는지 알 필요 없어. 일반적인 질의응답의 답변처럼 작성해. '제공된 정보에는~' '제공된 문서에는~' 이런 말투 쓰지마.
                    - 한국어로 답변해.
                [정보] \n {rag_reranked_docs}"""),
    ("human", "{user_question}"),
    MessagesPlaceholder("messages")
])
final_chain = prompt_final | llm.with_config({'temperature': 0})
# hallucination chain
prompt_check = ChatPromptTemplate.from_messages(
[
    ("system", """너는 RAG(Retrieval-Augmented Generation)결과물인 [문서 정보]과 LLM 모델이 생성한 [생성 답변]을 비교하여, [생성 답변]에 [문서 정보]에 없는 내용이 포함되어있는지 hallucination 여부를 판단하는 역할이야.
                \n[조건]\n : hallucination 발생 시 True, 없을 시 False를 'result' key에 반환하세요. 그리고 hallucination이 발생했다고 판단한 이유를 'reason' key에 반환하세요.
                \n[문서 정보]\n {rag_reranked_docs}
                \n[생성 답변]\n {final_answer}
                """),
    ("human", "{user_question}")
])
hallucination_chain = prompt_check | llm.with_config({'temperature': 0}).with_structured_output(HallucinationState)



def rag_init_node(state: RAGState) -> Command[Literal["rag_execute_node", END]]:
    logger.info(' == [rag_init_node] node init == ')
    try:
        # 기존 DB가 존재하는지 확인
        if os.path.exists(CHROMA_DIR):
            vectorstore = Chroma(
                collection_name=COLLECTION,
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
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
        search_results = vectorstore.similarity_search_with_score(
            query=query,
            k=5  # 상위 5개 문서 검색
        )
        # 결과 포맷팅
        formatted_results = []
        for doc, score in search_results:
            formatted_results.append({
                "content": doc.page_content,
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

# 추후에 toolnode로 만들고 
def rag_execute_node(state: RAGState) -> RAGState:
    logger.info(' == [rag_execute_node] node init == ')
    # retrive
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    documents = retriever.get_relevant_documents(state['messages'][-1].content)
    # rerank
    query_doc_pairs = [(state['messages'][-1].content, doc.page_content) for doc in documents]
    scores = reranker.predict(query_doc_pairs)
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    # 상위 k개 문서 반환
    reranked_docs = ['<Document>' + doc.page_content[:] + '</Document>' for score, doc in scored_docs[:top_k]]

    return {
        "rag_retrieved_docs": documents,
        "rag_reranked_docs": reranked_docs
    }

def rag_final_answer_gen(state: RAGState) -> RAGState:
    logger.info(' == [rag_final_answer_gen] node init == ')

    rag_reranked_docs = state['rag_reranked_docs']

    output = final_chain.invoke({
        "user_question": state['messages'][-1].content, 
        "messages": state["messages"],
        "rag_reranked_docs": rag_reranked_docs
        })
    return {
        "messages": AIMessage(content=output.content.split('</think>\n\n')[-1]),
        "final_answer": output.content.split('</think>\n\n')[-1]
    }

def hallucination_check(state: RAGState) -> RAGState:
    logger.info(' == [hallucination_check] node init == ')

    rag_reranked_docs = state['rag_reranked_docs']
    final_answer = state['final_answer']        
    output = hallucination_chain.invoke({
        "user_question": state['messages'][-1].content,
        "rag_reranked_docs": rag_reranked_docs,
        "final_answer": final_answer
        })
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
############################## tool test by dykim ################################
tools = [search_company_documents]
agent = create_tool_calling_agent(llm, tools, document_search_prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=False, 
    return_intermediate_steps=True
)

test_query = "국내 출장 시 외근교통비 청구 기준이 어떻게 되나요?"

result = agent_executor.invoke({"input": test_query})
print(f"질문: {test_query}")
print(result['output'].split('</think>\n\n')[-1])
print(f"중간 단계 수: {len(result['intermediate_steps'])}")
##############################################################

agent = create_tool_calling_agent(llm.with_config({'temperature': 0.3}), tools, prompt_init)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True)

builder = StateGraph(RAGState)
# node
builder_rag.add_node("rag_init_node", rag_init_node)
builder_rag.add_node("rag_execute_node", rag_execute_node)
builder_rag.add_node("rag_final_answer_gen", rag_final_answer_gen)
builder_rag.add_node("hallucination_check", hallucination_check)
# edge
builder_rag.add_edge(START, "rag_init_node")
builder_rag.add_edge("rag_init_node", "rag_execute_node")
builder_rag.add_edge("rag_execute_node", "rag_final_answer_gen")
builder_rag.add_edge("rag_final_answer_gen", "hallucination_check")
builder_rag.add_edge("hallucination_check", END)
checkpointer = MemorySaver()
graph_rag = builder_rag.compile(checkpointer=checkpointer)

###########################################################################






# 이 ID는 체크포인트 파일 내에서 특정 대화 세션을 식별하는 데 사용됩니다.
config = {"configurable": {"thread_id": "dayeon"}}


sql_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt="",
    name="sql_agent"
)

rag_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt="",
    name="rag_agent"
)
research_agent = create_react_agent(
    model=llm,
    tools=[book_hotel],
    prompt="",
    name="research_agent"
)

##https://docs.langchain.com/oss/python/langchain/short-term-memory#pre-model-hook
##https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/
supervisor = create_supervisor(
    agents=[sql_agent, rag_agent, web_search_agent],
    model=llm,
    pre_model_hook=[planner_agent],
    prompt=(router_system_prompt_template)
).compile()


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