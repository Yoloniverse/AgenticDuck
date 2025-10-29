


import aiosqlite
from langgraph.checkpoint.postgres import PostgresSaver
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph_supervisor import create_supervisor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver ## This should be changed to PostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langgraph.types import RetryPolicy, CachePolicy
from langchain_core.runnables import RunnableLambda

from typing import TypedDict, List, Annotated, Any
import sqlite3
from dotenv import load_dotenv
from prompts import planner_system_prompt_template, router_system_prompt_template, repeat_refined_query_system_prompt_template
import os
import uuid

# ## should change username, passcode, host, port, database names to real ones.
# DB_URI = "postgresql://user:password@localhost:5432/dbname" 
# checkpointer = PostgresSaver.from_conn_string(DB_URI)
# checkpoint_saver = PostgresSaver(db_uri=DB_URI, table_name="agent_checkpoints")

load_dotenv()
print("LANGGRAPH_AES_KEY =", os.getenv("LANGGRAPH_AES_KEY"))



##Sqilite 사용 할 수 있게 하는 코드 (sync)
# serde = EncryptedSerializer.from_pycryptodome_aes()  # reads LANGGRAPH_AES_KEY
# checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db", check_same_thread=False), serde=serde)

##Sqilite 사용 할 수 있게 하는 코드 (async)
serde = EncryptedSerializer.from_pycryptodome_aes()
db_file = "checkpoint.db"
conn_coro = aiosqlite.connect(db_file)
checkpointer = AsyncSqliteSaver(
    conn=conn_coro,
    serde=serde
)


llm = ChatOllama(model="qwen3:8b", base_url="http://127.0.0.1:11434")
# checkpointer = SqliteSaver.from_file("langgraph_checkpoints.sqlite")
 

class UserInputState(TypedDict):  
    messages: Annotated[List[BaseMessage], add_messages]

class plannerOutputState(TypedDict):  
    task_id: str
    task_description: str
    dependencies: List[str]
    priority: int

class PlannerTasksState(TypedDict):
    tasks: List[plannerOutputState]

planner_llm_chain = planner_system_prompt_template | llm.with_structured_output(PlannerTasksState)
# decomposed_result = planner_llm_chain.invoke("I wanna go to Italy. tell me how to go to italy and what to eat. And also tell me when the best seasons to visit is")
# decomposed_result = planner_llm_chain.invoke({"messages": [{"type": "human", "content": "I dont know what to do to find a job in Singapore"}]} )
# type(decomposed_result)


class QueryRefineryTasks(TypedDict):
    user_question: str
# refinery_llm_chain = repeat_refined_query_system_prompt_template | llm.with_structured_output(QueryRefineryTasks)
refinery_llm_chain = repeat_refined_query_system_prompt_template | llm
# result = refinery_llm_chain.invoke({"query": [{"type": "human", "content": "I wanna know how to go to Singapore from KL in Malaysia"}]})
# result = refinery_llm_chain.invoke({"query": [{"type": "human", "content": "What the fuck is wrong with this world?"}]})


class zz(TypedDict):  
    agent: str

class routerOutputState(TypedDict):  
    agent: str
    task: plannerOutputState




router_llm_chain = router_system_prompt_template | llm.with_structured_output(routerOutputState)

class SupervisorOverallState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_question: str
    tasks: List[plannerOutputState]
    task_id: str
    task_description: str
    dependencies: List[str]
    priority: int
    routing_results: List[routerOutputState]




async def task_decompose_node(state: UserInputState) -> PlannerTasksState:
    print(state['messages'])
    decomposed_result = await planner_llm_chain.ainvoke({"messages": [{"type": "human", "content": state['messages'][-1].content}]})
    return decomposed_result


async def subtask_router_worker(state: plannerOutputState) -> routerOutputState:
    """
    Async worker that computes only SINGLE task
    """
    # desc = task.task_description
    # print(f"  > (Async) 태스크 시작: '{desc}'")
    # 비동기 작업 시뮬레이션
    task = state['task_description']
    result = await router_llm_chain.ainvoke({"messages": [{"type": "human", "content": task}]})
    # result = router_llm_chain.invoke({"messages": [{"type": "human", "content": task}]})
    # result = f"[Execution Result for: '{desc}']"
    # print(f"  < (Async) 태스크 완료: '{desc}'")
    return result

async def parallel_task_routing_node(state: PlannerTasksState) -> SupervisorOverallState:
    """
    asynce worker that computes PARALLEL multiple tasks by abatch
    """

    if not state['tasks']:
        print("????????????????????????????????????????????????????????")

        # return {"completed_results": []}
    print(f"\nDEBUG: 'task_routing_node runs' a tasks.")
    subtask_worker_runnable = RunnableLambda(subtask_router_worker)
    results = await subtask_worker_runnable.abatch(state['tasks'])
    # results = subtask_worker_runnable.batch(decomposed_result['tasks'])
    return {"routing_results": results}






# task_agents = []


# num = 0
# for task in decomposed_result['tasks']:
#     num+=1
#     print(num)
#     result = router_llm_chain.invoke({"query": [{"type": "human", "content": "{task}"}]})
#     task_agents.append(result)



# result = router_llm_chain.invoke({"query": [{"type": "human", "content": "{decomposed_result}"}]})

# router_llm_chain.invoke('I wanna make select query for sql for example')
# router_llm_chain.invoke('I wanna travel to latin america')
# router_llm_chain.invoke('I wanna check facts in my documents')






# agent = create_agent(
#     model="anthropic:claude-sonnet-4-5",
#     tools=[search_web, analyze_data, send_email],
#     system_prompt="You are a helpful research assistant."
# )


# sql_agent = create_agent(
#     model=llm,
#     tools=[],
#     system_prompt="",
#     name="sql_agent"
# )

# rag_agent = create_agent(
#     model=llm,
#     tools=[],
#     system_prompt="",
#     name="rag_agent"
# )

# research_agent = create_agent(
#     model=llm,
#     tools=[book_hotel],
#     system_prompt="",
#     name="research_agent"
# )




in_memory_store = InMemoryStore()
thread_id = str(uuid.uuid4())
user_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}


master_builder = StateGraph(SupervisorOverallState)
master_builder.add_node("decomposer", task_decompose_node, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
master_builder.add_node("parallel_router", parallel_task_routing_node, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))

master_builder.add_edge(START, "decomposer")
master_builder.add_edge("decomposer", "parallel_router")
master_builder.add_edge("parallel_router", END)
master_graph = master_builder.compile(checkpointer=checkpointer, store=in_memory_store)


# Show the agent
from IPython.display import Image, display
display(Image(master_graph.get_graph(xray=True).draw_mermaid_png()))





result = await master_graph.ainvoke({"messages": [{"type": "human", "content": "What the fuck is wrong with this world?"}]}, config)


master_graph.get_state(config).metadata ##'user_id': 'ab815cde-4970-4201-a897-d1da5ad8d3fa'
master_graph.get_state(config).parent_config ##'checkpoint_id': '1f0b488d-71d6-63b7-8003-17876c44f6a3'
master_graph.get_state(config).values
list(master_graph.get_state_history(config))
thread_config = {"configurable": {"thread_id": thread_id}}
# user_config = {"configurable": {"user_id": user_id}} ##thread_id 무조건 있어야 함 
# master_graph.get_state(thread_config)
