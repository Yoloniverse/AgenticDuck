import sqlite3
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph_supervisor import create_supervisor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from dotenv import load_dotenv
from prompts import planner_system_prompt_template, router_system_prompt_template, repeat_refined_query_system_prompt_template
import os
from typing import Any
# ## should change username, passcode, host, port, database names to real ones.
# DB_URI = "postgresql://user:password@localhost:5432/dbname" 
# checkpointer = PostgresSaver.from_conn_string(DB_URI)
# checkpoint_saver = PostgresSaver(db_uri=DB_URI, table_name="agent_checkpoints")

load_dotenv()
print("LANGGRAPH_AES_KEY =", os.getenv("LANGGRAPH_AES_KEY"))

##Sqilite 사용 할 수 있게 하는 코드 
serde = EncryptedSerializer.from_pycryptodome_aes()  # reads LANGGRAPH_AES_KEY
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)

llm = ChatOllama(model="qwen3:8b", base_url="http://127.0.0.1:11434")
# checkpointer = SqliteSaver.from_file("langgraph_checkpoints.sqlite")
 

class InputState(TypedDict):  
    messages: Annotated[List[BaseMessage], add_messages]

class plannerInputState(TypedDict):  
    task_id: str
    task_description: str
    dependencies: List[str]
    priority: int


class PlannerTasks(TypedDict):
    tasks: List[plannerInputState]

planner_llm_chain = planner_system_prompt_template | llm.with_structured_output(PlannerTasks)

def task_decompose_node()
decomposed_result = planner_llm_chain.invoke("I wanna go to Italy. tell me how to go to italy and what to eat. And also tell me when the best seasons to visit is")
decomposed_result = planner_llm_chain.invoke({"query": [{"type": "human", "content": "I dont know what to do to find a job in Singapore"}]} )
type(decomposed_result)

decomposed_result['tasks']


class QueryRefineryTasks(TypedDict):
    user_question: str

refinery_llm_chain = repeat_refined_query_system_prompt_template | llm.with_structured_output(QueryRefineryTasks)
refinery_llm_chain = repeat_refined_query_system_prompt_template | llm

result = refinery_llm_chain.invoke({"query": [{"type": "human", "content": "I wanna know how to go to Singapore from KL in Malaysia"}]})
result = refinery_llm_chain.invoke({"query": [{"type": "human", "content": "What the fuck is wrong with this world?"}]})







class routerOutputState(TypedDict):  
    agent: str

router_llm_chain = router_system_prompt_template | llm.with_structured_output(routerOutputState)

len(decomposed_result['tasks'])


task_agents = []


num = 0
for task in decomposed_result['tasks']:
    num+=1
    print(num)
    result = router_llm_chain.invoke({"query": [{"type": "human", "content": "{task}"}]})
    task_agents.append(result)



result = router_llm_chain.invoke({"query": [{"type": "human", "content": "{decomposed_result}"}]})

router_llm_chain.invoke('I wanna make select query for sql for example')
router_llm_chain.invoke('I wanna travel to latin america')
router_llm_chain.invoke('I wanna check facts in my documents')




# 이 ID는 체크포인트 파일 내에서 특정 대화 세션을 식별하는 데 사용됩니다.
config = {"configurable": {"thread_id": "dayeon"}}


# agent = create_agent(
#     model="anthropic:claude-sonnet-4-5",
#     tools=[search_web, analyze_data, send_email],
#     system_prompt="You are a helpful research assistant."
# )


sql_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="",
    name="sql_agent"
)

rag_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="",
    name="rag_agent"
)

research_agent = create_agent(
    model=llm,
    tools=[book_hotel],
    system_prompt="",
    name="research_agent"
)


class SupervisorOverallState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_question: str
    tasks: List[plannerInputState]
    task_id: str
    task_description: str
    dependencies: List[str]
    priority: int




master_graph = StateGraph(SupervisorOverallState)

master_graph.add_edge(START, "start_point")




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