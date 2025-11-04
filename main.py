


import aiosqlite
from langgraph.checkpoint.postgres import PostgresSaver
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph_supervisor import create_supervisor
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver ## This should be changed to PostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import RetryPolicy, CachePolicy

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool


from typing import TypedDict, List, Annotated, Any
import sqlite3
from dotenv import load_dotenv
from prompts import planner_system_prompt_template, router_system_prompt_template, repeat_refined_query_system_prompt_template
import os
import uuid
import json
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
    """
    Input: User's query text in list. UserInputState -> messages: Annotated[List[BaseMessage], add_messages]
    """
    refined_statement: str
# refinery_llm_chain = repeat_refined_query_system_prompt_template | llm.with_structured_output(QueryRefineryTasks)
refinery_llm_chain = repeat_refined_query_system_prompt_template | llm
# result = refinery_llm_chain.invoke({"query": [{"type": "human", "content": "I wanna know how to go to Singapore from KL in Malaysia"}]})
# result = refinery_llm_chain.invoke({"query": [{"type": "human", "content": "What the fuck is wrong with this world?"}]})




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
    refined_statement: str
    final_statement: str



async def task_decompose_node(state: UserInputState) -> PlannerTasksState: ##-> PlannerTasksState: 이렇게 output format을 지정하지 않으면, 그에 맞게 및의 return decomposed_result를 {"tasks": decomposed_result} 으로 지정한다
    # print(f"User input message: {state['messages']}")
    # print(f"User input message type: {type(state['messages'])}") ##User input message type: <class 'list'>
    decomposed_result = await planner_llm_chain.ainvoke({"messages": [{"type": "human", "content": state['messages'][-1].content}]})
    # print(f"decomposed_result: {decomposed_result}") ##decomposed_result: {'tasks': [{'task_id': 'task_1', 'task_description': "Research Germany's job market trends and in-demand industries", 'dependencies': [], 'priority': 1}, {'task_id': 'task_2', 'task_description': 'Tailor resume and cover letter to German job market standards', 'dependencies': ['task_1'], 'priority': 2}, {'task_id': 'task_3', 'task_description': 'Learn German language proficiency (B1/C1 level recommended)', 'dependencies': ['task_1'], 'priority': 3}, {'task_id': 'task_4', 'task_description': 'Create LinkedIn profile optimized for German job search', 'dependencies': ['task_2', 'task_3'], 'priority': 4}, {'task_id': 'task_5', 'task_description': 'Apply for jobs through German job portals (e.g., StepStone, Indeed, Monster)', 'dependencies': ['task_4'], 'priority': 5}, {'task_id': 'task_6', 'task_description': 'Prepare for job interviews with German cultural norms and language practice', 'dependencies': ['task_5'], 'priority': 6}, {'task_id': 'task_7', 'task_description': 'Research visa/work permit requirements for foreign nationals', 'dependencies': ['task_1'], 'priority': 7}, {'task_id': 'task_8', 'task_description': 'Plan relocation logistics (housing, banking, insurance)', 'dependencies': ['task_7'], 'priority': 8}, {'task_id': 'task_9', 'task_description': 'Consider hiring recruitment agency for specialized roles', 'dependencies': ['task_5'], 'priority': 9}]}
    # print(f"Individual decomposed_result type: {type(decomposed_result['tasks'][0])}") 
    # print(f"decomposed_result type: {type(decomposed_result)}") ##decomposed_result type: <class 'dict'>
    return decomposed_result


async def statement_refinery(state: UserInputState) -> SupervisorOverallState:
    refined_statement = await refinery_llm_chain.ainvoke({"messages": [{"type": "human", "content": state['messages'][-1].content}]})
    print(f"refined_statement: {refined_statement}")
    # return {"refined_statement": refined_statement}
    return {"refined_statement": refined_statement}


async def subtask_router_worker(state: plannerOutputState) -> routerOutputState:
    """
    Async worker that computes only SINGLE task
    """

    ##여기의 state는 리스트는 하나하나 개별 값들 
    # print(state)
    # print(f"plannerOutputState type: {type(state)}") ##plannerOutputState type: <class 'dict'>
    # print(f"plannerOutputState: {state}") ##{'task_id': 'task_2', 'task_description': 'Update and tailor resume/cv for German job applications', 'dependencies': ['task_1'], 'priority': 2}
    task = state['task_description']
    router_for_individual_task_result = await router_llm_chain.ainvoke({"messages": [{"type": "human", "content": task}]})
    # print(f"subtask_router_worker: {router_for_individual_task_result}") ## {'agent': 'research_supervisor', 'task': {'task_id': 'networking_professionals', 'task_description': 'Identify and connect with professionals in target industries via LinkedIn and local German professional groups', 'dependencies': [], 'priority': 1}}
    # print(f"subtask_router_worker: {type(router_for_individual_task_result)}") ## subtask_router_worker: <class 'dict'>
    # result = router_llm_chain.invoke({"messages": [{"type": "human", "content": task}]})
    # result = f"[Execution Result for: '{desc}']"
    # print(f"  < (Async) 태스크 완료: '{desc}'")
    # return {"routerOutputState": result}
    return router_for_individual_task_result

# subtask_router_worker = {'agent': 'research_supervisor', 'task': {'task_id': 'networking_professionals', 'task_description': 'Identify and connect with professionals in target industries via LinkedIn and local German professional groups', 'dependencies': [], 'priority': 1}}
# type(subtask_router_worker)


async def parallel_task_routing_node(state: PlannerTasksState) -> SupervisorOverallState:
    """
    asynce worker that computes PARALLEL multiple tasks by abatch
    """
    # print(f"state structure: {state}") ##{'tasks': [{'task_id': 'task_1', 'task_description': 'Research the German job market (industries, cities with job opportunities, salary trends)', 'dependencies': [], 'priority': 1}, {'task_id': 'task_2', 'task_description': 'Create a tailored resume and cover letter compliant with German standards', 'dependencies': ['task_1'], 'priority': 2}, {'task_id': 'task_3', 'task_description': 'Prepare for job interviews (research common German interview practices, practice answers)', 'dependencies': ['task_2'], 'priority': 3}, {'task_id': 'task_4', 'task_description': 'Network with professionals in target industries (LinkedIn, local German professional groups)', 'dependencies': ['task_1', 'task_2'], 'priority': 4}, {'task_id': 'task_5', 'task_description': 'Apply for work visa (research required documents, application process, processing times)', 'dependencies': ['task_1'], 'priority': 5}, {'task_id': 'task_6', 'task_description': 'Utilize German job portals (StepStone, Indeed, Xing, local company career pages)', 'dependencies': ['task_1', 'task_2'], 'priority': 6}, {'task_id': 'task_7', 'task_description': 'Develop German language skills (certifications like Goethe Institute, language practice)', 'dependencies': ['task_1'], 'priority': 7}]}
    # print(f"state structure: {state['tasks']}") ##[{'task_id': 'task_1', 'task_description': 'Research the German job market (industries, cities with job opportunities, salary trends)', 'dependencies': [], 'priority': 1}, {'task_id': 'task_2', 'task_description': 'Create a tailored resume and cover letter compliant with German standards', 'dependencies': ['task_1'], 'priority': 2}, {'task_id': 'task_3', 'task_description': 'Prepare for job interviews (research common German interview practices, practice answers)', 'dependencies': ['task_2'], 'priority': 3}, {'task_id': 'task_4', 'task_description': 'Network with professionals in target industries (LinkedIn, local German professional groups)', 'dependencies': ['task_1', 'task_2'], 'priority': 4}, {'task_id': 'task_5', 'task_description': 'Apply for work visa (research required documents, application process, processing times)', 'dependencies': ['task_1'], 'priority': 5}, {'task_id': 'task_6', 'task_description': 'Utilize German job portals (StepStone, Indeed, Xing, local company career pages)', 'dependencies': ['task_1', 'task_2'], 'priority': 6}, {'task_id': 'task_7', 'task_description': 'Develop German language skills (certifications like Goethe Institute, language practice)', 'dependencies': ['task_1'], 'priority': 7}]

    if not state['tasks']:
        print("No tasks given!!")

        # return {"completed_results": []}
    print(f"\nDEBUG: 'task_routing_node runs' tasks.")
    subtask_worker_runnable = RunnableLambda(subtask_router_worker)
    subtask_routing_results = await subtask_worker_runnable.abatch(state['tasks']) ##모든 태스크들이 list로 들어감 
    # print(f"subtask_routing_results type: {type(subtask_routing_results)}")  ##subtask_routing_results type: <class 'list'>
    # print(f"parallel_task_routing_node final result: {subtask_routing_results}") ##parallel_task_routing_node final result: [{'agent': 'research_supervisor', 'task': {'task_id': 'research_german_job_market', 'task_description': 'Research the German job market, including in-demand industries and required qualifications', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'resume_cover_letter_german_standard', 'task_description': 'Create a tailored resume and cover letter according to German standards', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'networking_strategy_research', 'task_description': 'Conduct research on effective networking strategies using LinkedIn and German job portals to connect with industry professionals', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'GERMAN_JOB_COMMUNICATION', 'task_description': 'Provide resources and guidance for learning basic German language skills focused on job applications and workplace communication scenarios.', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'job_application_guidance', 'task_description': 'Provide step-by-step guidance on applying for jobs through German job portals (StepStone, Indeed Germany) and company career pages', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'interview_preparation_german_employers', 'task_description': 'Research cultural norms, common interview questions, visa requirements, and company backgrounds for German employers to prepare effective virtual/in-person interviews', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Research visa/work permit requirements for foreign professionals in Germany', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Identify and research potential employers in target industries in Germany', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'evaluate_relocation_costs_housing_quality_of_life_germany', 'task_description': 'Evaluate relocation costs, housing options, and quality of life in Germany', 'dependencies': [], 'priority': 1}}]

    # results = subtask_worker_runnable.batch(decomposed_result['tasks'])
    return {"routing_results": subtask_routing_results}


# state_structure = {'tasks': [{'task_id': 'task_1', 'task_description': 'Research the German job market (industries, cities with job opportunities, salary trends)', 'dependencies': [], 'priority': 1}, {'task_id': 'task_2', 'task_description': 'Create a tailored resume and cover letter compliant with German standards', 'dependencies': ['task_1'], 'priority': 2}, {'task_id': 'task_3', 'task_description': 'Prepare for job interviews (research common German interview practices, practice answers)', 'dependencies': ['task_2'], 'priority': 3}, {'task_id': 'task_4', 'task_description': 'Network with professionals in target industries (LinkedIn, local German professional groups)', 'dependencies': ['task_1', 'task_2'], 'priority': 4}, {'task_id': 'task_5', 'task_description': 'Apply for work visa (research required documents, application process, processing times)', 'dependencies': ['task_1'], 'priority': 5}, {'task_id': 'task_6', 'task_description': 'Utilize German job portals (StepStone, Indeed, Xing, local company career pages)', 'dependencies': ['task_1', 'task_2'], 'priority': 6}, {'task_id': 'task_7', 'task_description': 'Develop German language skills (certifications like Goethe Institute, language practice)', 'dependencies': ['task_1'], 'priority': 7}]}
# state_structure['tasks']
# parallel_task_routing_node_final_result = [{'agent': 'research_supervisor', 'task': {'task_id': 'research_german_job_market', 'task_description': 'Research the German job market, including in-demand industries and required qualifications', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'resume_cover_letter_german_standard', 'task_description': 'Create a tailored resume and cover letter according to German standards', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'networking_strategy_research', 'task_description': 'Conduct research on effective networking strategies using LinkedIn and German job portals to connect with industry professionals', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'GERMAN_JOB_COMMUNICATION', 'task_description': 'Provide resources and guidance for learning basic German language skills focused on job applications and workplace communication scenarios.', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'job_application_guidance', 'task_description': 'Provide step-by-step guidance on applying for jobs through German job portals (StepStone, Indeed Germany) and company career pages', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'interview_preparation_german_employers', 'task_description': 'Research cultural norms, common interview questions, visa requirements, and company backgrounds for German employers to prepare effective virtual/in-person interviews', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Research visa/work permit requirements for foreign professionals in Germany', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Identify and research potential employers in target industries in Germany', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'evaluate_relocation_costs_housing_quality_of_life_germany', 'task_description': 'Evaluate relocation costs, housing options, and quality of life in Germany', 'dependencies': [], 'priority': 1}}]


# state_list = []
# state_list.append(refinery_state['refined_statement'])
# state_list.append('I am fine and you?')
# from pprint import pprint
# print("\n\n".join(state_list))


import pickle


def result_concatnater(state: SupervisorOverallState) -> SupervisorOverallState:
    state_list = []
    # state_list2 = []
    state_list.append(state['refined_statement'].content)
    state_list.append(str(state["routing_results"]))
    with open("/home/sdt/Workspace/mvai/AgenticRAG/test.pkl", "wb") as f:
        pickle.dump(state_list, f)
    print(f"state_list: {state_list}")
    # state_list2.append(state['refined_statement']['messages'])
    # state_list2.append(str(state["routing_results"]))
    final_statement = "\n\n".join(state_list)
    print(f"final_statement: {final_statement}")
    return {"final_statement": final_statement}


# final_statement = [AIMessage(content='Let me rephrase your question!  \nYour goal is to find a job in Germany. What specific steps or actions should you take to make this happen?  \nIs my understanding correct?', additional_kwargs={}, response_metadata={'model': 'qwen3:8b', 'created_at': '2025-11-04T02:21:25.093749871Z', 'done': True, 'done_reason': 'stop', 'total_duration': 51657378989, 'load_duration': 65300947, 'prompt_eval_count': 233, 'prompt_eval_duration': 4709188084, 'eval_count': 421, 'eval_duration': 46754690176, 'model_name': 'qwen3:8b', 'model_provider': 'ollama'}, id='lc_run--5d9344b5-490d-4f88-ac00-2ee0bea906a9-0', usage_metadata={'input_tokens': 233, 'output_tokens': 421, 'total_tokens': 654}), "[{'agent': 'research_supervisor', 'task': {'task_id': 'research_german_job_market', 'task_description': 'Research the German job market and industry demand', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Prepare a tailored resume and cover letter for German employers', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Build professional network through LinkedIn and industry events', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'JOB_PORTAL_APPLICATION_RESEARCH', 'task_description': 'Research and guide user on applying for jobs through German job portals like StepStone and Indeed Germany', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Understand visa/work permit requirements for foreign workers', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Prepare for job interviews with German cultural norms and language skills', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'RELOCATION_LOGISTICS_GERMANy', 'task_description': 'Explore relocation logistics and cost of living in Germany', 'dependencies': [], 'priority': 1}}]"]

# # 1. 'data.pkl' 파일을 'rb' 모드로 엽니다.
# with open("/home/sdt/Workspace/mvai/AgenticRAG/test.pkl", 'rb') as f:
#     # 2. 파일에서 데이터를 불러와(load) loaded_data 변수에 할당합니다.
#     loaded_data = pickle.load(f)
# loaded_data[0].content
# loaded_data[1]
# "\n\n".join(loaded_data)

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


## subgraph
parallel_builder = StateGraph(SupervisorOverallState)
parallel_builder.add_node("decomposer", task_decompose_node, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
parallel_builder.add_node("parallel_router", parallel_task_routing_node, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
parallel_builder.add_node("statement_refinery", statement_refinery, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))

## 서브그래프의 내부 흐름 정의
parallel_builder.add_edge(START, "decomposer")
parallel_builder.add_edge(START, "statement_refinery")
parallel_builder.add_edge("decomposer", "parallel_router")

## 두 병렬 브랜치가 모두 서브그래프의 END를 가리키도록 함
parallel_builder.add_edge("parallel_router", END)
parallel_builder.add_edge("statement_refinery", END)


parallel_graph = parallel_builder.compile()



## master graph
master_builder = StateGraph(SupervisorOverallState)

## 서브그래프 자체를 'parallel_step'이라는 이름의 단일 노드로 추가
master_builder.add_node("parallel_step", parallel_graph)
## 결과 취합 노드 추가
master_builder.add_node("result_concatnater", result_concatnater, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))

## 메인 그래프는 이제 단순한 순차 흐름이 됨
master_builder.set_entry_point("parallel_step")
master_builder.add_edge("parallel_step", "result_concatnater")
master_builder.add_edge("result_concatnater", END)


master_graph = master_builder.compile(checkpointer=checkpointer, store=in_memory_store)






# master_builder = StateGraph(SupervisorOverallState)
# master_builder.add_node("decomposer", task_decompose_node, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
# master_builder.add_node("parallel_router", parallel_task_routing_node, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
# master_builder.add_node("statement_refinery", statement_refinery, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
# master_builder.add_node("result_concatnater", result_concatnater, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))


# master_builder.add_edge(START, "decomposer")
# master_builder.add_edge(START, "statement_refinery")
# master_builder.add_edge("decomposer", "parallel_router")
# master_builder.add_edge("parallel_router", "result_concatnater")
# master_builder.add_edge("statement_refinery", "result_concatnater")
# master_builder.add_edge("result_concatnater", END)


# master_graph = master_builder.compile(checkpointer=checkpointer, store=in_memory_store)


# Show the agent
from IPython.display import Image, display
display(Image(master_graph.get_graph(xray=True).draw_mermaid_png()))




result = await master_graph.ainvoke({"messages": [{"type": "human", "content": "What should I do to find a job in Germany?"}]}, config)


type(result)
result['messages']
result['tasks']

master_graph.get_state(config).metadata ##'user_id': 'ab815cde-4970-4201-a897-d1da5ad8d3fa'
master_graph.get_state(config).parent_config ##'checkpoint_id': '1f0b488d-71d6-63b7-8003-17876c44f6a3'
master_graph.get_state(config).values
list(master_graph.get_state_history(config))
thread_config = {"configurable": {"thread_id": thread_id}}
# user_config = {"configurable": {"user_id": user_id}} ##thread_id 무조건 있어야 함 
# master_graph.get_state(thread_config)
