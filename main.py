


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
from langchain.agents import create_agent

from operator import add
from pydantic.v1 import BaseModel, Field
from typing import TypedDict, List, Annotated, Any, Literal, Dict
import sqlite3
from dotenv import load_dotenv
import os
import uuid
import json
import pickle
import logging
import operator

##custom
from prompts import planner_system_prompt_template, router_system_prompt_template, repeat_refined_query_system_prompt_template, intent_classification_prompt_template##, tool_calling_evaluator_prompt_template
from toolings import taviliy_web_search_tool
import mcp_server_git

# ## should change username, passcode, host, port, database names to real ones.
# DB_URI = "postgresql://user:password@localhost:5432/dbname" 
# checkpointer = PostgresSaver.from_conn_string(DB_URI)
# checkpoint_saver = PostgresSaver(db_uri=DB_URI, table_name="agent_checkpoints")

load_dotenv()
print("LANGGRAPH_AES_KEY =", os.getenv("LANGGRAPH_AES_KEY"))


######################################################################
#                             Save Log                               #
######################################################################
os.makedirs('./logs', exist_ok=True)
logger = logging.getLogger("MultiAgents")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_max_size = 1024000
log_file_count = 3
log_fileHandler = logging.handlers.RotatingFileHandler(
        filename=f"./logs/multi_agents_main.log",
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode='a')

log_fileHandler.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(log_fileHandler)
logger.propagate = False




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
    messages: Annotated[List[str], add_messages] ##유저 인풋

class plannerSingleState(TypedDict):  
    task_id: str
    task_description: str
    dependencies: List[str]
    priority: int

class PlannerTasksState(TypedDict):
    tasks: List[plannerSingleState]

planner_llm_chain = planner_system_prompt_template | llm.with_structured_output(PlannerTasksState)
# decomposed_result = planner_llm_chain.invoke("I wanna go to Italy. tell me how to go to italy and what to eat. And also tell me when the best seasons to visit is")
# decomposed_result = planner_llm_chain.invoke({"messages": [{"type": "human", "content": "I dont know what to do to find a job in Singapore"}]} )
# type(decomposed_result)

class intentClassifyingState(TypedDict):
    intent: str

intent_classifier_llm_chain = intent_classification_prompt_template | llm.with_structured_output(intentClassifyingState)
# intent_classifier_llm_chain.invoke("I wanna know information on how to move to Australia")

class QueryRefineryTasks(TypedDict):
    """
    Input: User's query text in list. UserInputState -> messages: Annotated[List[BaseMessage], add_messages]
    """
    refined_statement: str

# refinery_llm_chain = repeat_refined_query_system_prompt_template | llm.with_structured_output(QueryRefineryTasks)
refinery_llm_chain = repeat_refined_query_system_prompt_template | llm
# result = refinery_llm_chain.invoke({"query": [{"type": "human", "content": "I wanna know how to go to Singapore from KL in Malaysia"}]})
# result = refinery_llm_chain.invoke({"messages": [{"type": "human", "content": "What the fuck is wrong with this world?"}]})
# result.content

class routerState(TypedDict):  
    agent: str
    # task: plannerSingleState

router_llm_chain = router_system_prompt_template | llm.with_structured_output(routerState)


# class taskEvalState(TypedDict):  
#     each_task_evaluation: Literal["good", "bad"]
#     # task: plannerSingleState
    
# router_llm_chain = router_system_prompt_template | llm.with_structured_output(taskEvalState)



class SupervisorOverallState(TypedDict):
    messages: Annotated[List[str], add_messages]
    # user_question: str
    tasks: List[plannerSingleState]
    agent: str
    refined_statement: str
    routing_results: List[routerState]
    tool_callings_result: List[Dict] ##List[str]
    tool_calling_eval: Literal["good", "bad"]
    # task_id: str
    # task_description: str
    # dependencies: List[str]
    # priority: int
    
    
    # # tool_calling_result: Annotated[str, operator.add]
    
    # final_statement: str
    



# result['tasks']
# result['refined_statement']
# result['routing_results']


tool_calling_chain = create_agent(
    model=llm,
    tools=[taviliy_web_search_tool],
    system_prompt="You are a helpful assistant to choose a tool for a task.",
)

# subtask_tool_calling_worker_result = await tool_calling_chain.ainvoke({"messages": [{"type": "human", "content": "Please tell me the price of Seoul average apartment price"}]})
# subtask_tool_calling_worker_result['messages'][1].tool_calls
# subtask_tool_calling_worker_result['messages'][2]
# subtask_tool_calling_worker_result['messages'][3]






async def task_decompose_node(state: UserInputState) -> SupervisorOverallState: ##-> PlannerTasksState: 이렇게 output format을 지정하지 않으면, 그에 맞게 및의 return decomposed_result를 {"tasks": decomposed_result} 으로 지정한다
    print(f"task_decompose_node executed: {task_decompose_node}")
    # print(f"User input message: {state['messages']}")
    # print(f"User input message type: {type(state['messages'])}") ##User input message type: <class 'list'>
    decomposed_result = await planner_llm_chain.ainvoke({"messages": [{"type": "human", "content": state['messages'][-1].content}]})
    # print(f"decomposed_result: {decomposed_result}") ##decomposed_result: {'tasks': [{'task_id': 'task_1', 'task_description': "Research Germany's job market trends and in-demand industries", 'dependencies': [], 'priority': 1}, {'task_id': 'task_2', 'task_description': 'Tailor resume and cover letter to German job market standards', 'dependencies': ['task_1'], 'priority': 2}, {'task_id': 'task_3', 'task_description': 'Learn German language proficiency (B1/C1 level recommended)', 'dependencies': ['task_1'], 'priority': 3}, {'task_id': 'task_4', 'task_description': 'Create LinkedIn profile optimized for German job search', 'dependencies': ['task_2', 'task_3'], 'priority': 4}, {'task_id': 'task_5', 'task_description': 'Apply for jobs through German job portals (e.g., StepStone, Indeed, Monster)', 'dependencies': ['task_4'], 'priority': 5}, {'task_id': 'task_6', 'task_description': 'Prepare for job interviews with German cultural norms and language practice', 'dependencies': ['task_5'], 'priority': 6}, {'task_id': 'task_7', 'task_description': 'Research visa/work permit requirements for foreign nationals', 'dependencies': ['task_1'], 'priority': 7}, {'task_id': 'task_8', 'task_description': 'Plan relocation logistics (housing, banking, insurance)', 'dependencies': ['task_7'], 'priority': 8}, {'task_id': 'task_9', 'task_description': 'Consider hiring recruitment agency for specialized roles', 'dependencies': ['task_5'], 'priority': 9}]}
    # print(f"Individual decomposed_result type: {type(decomposed_result['tasks'][0])}") 
    # print(f"decomposed_result type: {type(decomposed_result)}") ##decomposed_result type: <class 'dict'>
    # print(f"decom: {decomposed_result}")
    return decomposed_result ##decomposed_result가 이미 tasks: List[plannerSingleState] 형태를 가지고 있기 때문에 {'tasks': decomposed_result} 로 쓰지 않는다.



async def statement_refinery(state: UserInputState) -> SupervisorOverallState:
    print(f"statement_refinery executed: {statement_refinery}")
    refined_statement = await refinery_llm_chain.ainvoke({"messages": [{"type": "human", "content": state['messages'][-1].content}]})
    print(f"refined_statement: {refined_statement}")
    # return {"refined_statement": refined_statement}
    return {"refined_statement": refined_statement.content}


async def subtask_router_worker(state: plannerSingleState) -> SupervisorOverallState:
    """
    Async worker that computes only SINGLE task
    """
    print(f"subtask_router_worker executed: {subtask_router_worker}")
    # print(f"hereeee: {state}")
    ##여기의 state는 리스트는 하나하나 개별 값들 
    # print(state)
    # print(f"plannerSingleState type: {type(state)}") ##plannerSingleState type: <class 'dict'>
    # print(f"plannerSingleState: {state}") ##{'task_id': 'task_2', 'task_description': 'Update and tailor resume/cv for German job applications', 'dependencies': ['task_1'], 'priority': 2}
    task = state['task_description'] ## state = tasks의 각각의 태스크
    # print(f"task hh: {task}")
    router_for_individual_task_result = await router_llm_chain.ainvoke({"messages": [{"type": "human", "content": task}]}) ##{'agent': 'research_supervisor'} 이런 형태
    # print(f"router hh: {router_for_individual_task_result}")
    # print(f"subtask_router_worker: {router_for_individual_task_result}") ## {'agent': 'research_supervisor', 'task': {'task_id': 'networking_professionals', 'task_description': 'Identify and connect with professionals in target industries via LinkedIn and local German professional groups', 'dependencies': [], 'priority': 1}}
    # print(f"subtask_router_worker: {type(router_for_individual_task_result)}") ## subtask_router_worker: <class 'dict'>
    # result = router_llm_chain.invoke({"messages": [{"type": "human", "content": task}]})
    # result = f"[Execution Result for: '{desc}']"
    # print(f"  < (Async) 태스크 완료: '{desc}'")
    # return {"routerState": result}
    # return router_for_individual_task_result ##router_for_individual_task_result가 이미 {'agent': 'research_supervisor'} 이런 형태이기 때문에 그냥 쓴다.
    
    router_task_result = {"agent": router_for_individual_task_result['agent'], "task": state}
    return router_task_result 



# subtask_router_worker = {'agent': 'research_supervisor', 'task': {'task_id': 'networking_professionals', 'task_description': 'Identify and connect with professionals in target industries via LinkedIn and local German professional groups', 'dependencies': [], 'priority': 1}}
# type(subtask_router_worker)


async def parallel_task_routing_node(state: PlannerTasksState) -> SupervisorOverallState:
    """
    asynce worker that computes PARALLEL multiple tasks by abatch
    """
    # print(f"state structure: {state}") ##{'tasks': [{'task_id': 'task_1', 'task_description': 'Research the German job market (industries, cities with job opportunities, salary trends)', 'dependencies': [], 'priority': 1}, {'task_id': 'task_2', 'task_description': 'Create a tailored resume and cover letter compliant with German standards', 'dependencies': ['task_1'], 'priority': 2}, {'task_id': 'task_3', 'task_description': 'Prepare for job interviews (research common German interview practices, practice answers)', 'dependencies': ['task_2'], 'priority': 3}, {'task_id': 'task_4', 'task_description': 'Network with professionals in target industries (LinkedIn, local German professional groups)', 'dependencies': ['task_1', 'task_2'], 'priority': 4}, {'task_id': 'task_5', 'task_description': 'Apply for work visa (research required documents, application process, processing times)', 'dependencies': ['task_1'], 'priority': 5}, {'task_id': 'task_6', 'task_description': 'Utilize German job portals (StepStone, Indeed, Xing, local company career pages)', 'dependencies': ['task_1', 'task_2'], 'priority': 6}, {'task_id': 'task_7', 'task_description': 'Develop German language skills (certifications like Goethe Institute, language practice)', 'dependencies': ['task_1'], 'priority': 7}]}
    # print(f"state structure: {state['tasks']}") ##[{'task_id': 'task_1', 'task_description': 'Research the German job market (industries, cities with job opportunities, salary trends)', 'dependencies': [], 'priority': 1}, {'task_id': 'task_2', 'task_description': 'Create a tailored resume and cover letter compliant with German standards', 'dependencies': ['task_1'], 'priority': 2}, {'task_id': 'task_3', 'task_description': 'Prepare for job interviews (research common German interview practices, practice answers)', 'dependencies': ['task_2'], 'priority': 3}, {'task_id': 'task_4', 'task_description': 'Network with professionals in target industries (LinkedIn, local German professional groups)', 'dependencies': ['task_1', 'task_2'], 'priority': 4}, {'task_id': 'task_5', 'task_description': 'Apply for work visa (research required documents, application process, processing times)', 'dependencies': ['task_1'], 'priority': 5}, {'task_id': 'task_6', 'task_description': 'Utilize German job portals (StepStone, Indeed, Xing, local company career pages)', 'dependencies': ['task_1', 'task_2'], 'priority': 6}, {'task_id': 'task_7', 'task_description': 'Develop German language skills (certifications like Goethe Institute, language practice)', 'dependencies': ['task_1'], 'priority': 7}]
    print(f"parallel_task_routing_node executed: {parallel_task_routing_node}")
    if not state['tasks']:
        print("No tasks given!!")

        # return {"completed_results": []}
    print(f"\nDEBUG: 'task_routing_node runs' tasks.")
    subtask_worker_runnable = RunnableLambda(subtask_router_worker)
    subtask_routing_results = await subtask_worker_runnable.abatch(state['tasks']) ##모든 태스크들이 list로 들어감 
    # print(f"subtask_routing_results type: {type(subtask_routing_results)}")  ##subtask_routing_results type: <class 'list'>
    # print(f"parallel_task_routing_node final result: {subtask_routing_results}") ##parallel_task_routing_node final result: [{'agent': 'research_supervisor', 'task': {'task_id': 'research_german_job_market', 'task_description': 'Research the German job market, including in-demand industries and required qualifications', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'resume_cover_letter_german_standard', 'task_description': 'Create a tailored resume and cover letter according to German standards', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'networking_strategy_research', 'task_description': 'Conduct research on effective networking strategies using LinkedIn and German job portals to connect with industry professionals', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'GERMAN_JOB_COMMUNICATION', 'task_description': 'Provide resources and guidance for learning basic German language skills focused on job applications and workplace communication scenarios.', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'job_application_guidance', 'task_description': 'Provide step-by-step guidance on applying for jobs through German job portals (StepStone, Indeed Germany) and company career pages', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'interview_preparation_german_employers', 'task_description': 'Research cultural norms, common interview questions, visa requirements, and company backgrounds for German employers to prepare effective virtual/in-person interviews', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Research visa/work permit requirements for foreign professionals in Germany', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Identify and research potential employers in target industries in Germany', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'evaluate_relocation_costs_housing_quality_of_life_germany', 'task_description': 'Evaluate relocation costs, housing options, and quality of life in Germany', 'dependencies': [], 'priority': 1}}]

    # results = subtask_worker_runnable.batch(decomposed_result['tasks'])
    return {"routing_results": subtask_routing_results} ## agent와 task 값이 같이 있는 List[Dict]
    """ 
    routing_results -> 
    [{'agent': 'research_supervisor'},
    {'agent': 'research_supervisor'},
    {'agent': 'research_supervisor'},
    {'agent': 'research_supervisor'},
    {'agent': 'research_supervisor'},
    {'agent': 'research_supervisor'},
    {'agent': 'research_supervisor'},
    {'agent': 'research_supervisor'}]
    """
    """
    [{'task_id': 'task_1',
    'task_description': 'Research the job market in Germany (industry trends, in-demand roles, salary ranges)',
    'dependencies': [],
    'priority': 1},
    {'task_id': 'task_2',
    'task_description': 'Tailor resume and cover letter to German job market standards',
    'dependencies': ['task_1'],
    'priority': 2},
    {'task_id': 'task_3',
    'task_description': 'Check visa/work permit requirements for foreign workers in Germany',
    'dependencies': ['task_1'],
    'priority': 3},
    {'task_id': 'task_4',
    'task_description': 'Network with professionals in target industries (LinkedIn, industry events, local communities)',
    'dependencies': ['task_1', 'task_2'],
    'priority': 4},
    {'task_id': 'task_5',
    'task_description': 'Apply for jobs via German job portals (Indeed, StepStone, LinkedIn, Xing)',
    'dependencies': ['task_2', 'task_3', 'task_4'],
    'priority': 5},
    {'task_id': 'task_6',
    'task_description': 'Prepare for job interviews (common questions, cultural norms, technical assessments)',
    'dependencies': ['task_5'],
    'priority': 6},
    {'task_id': 'task_7',
    'task_description': 'Evaluate language skills (German proficiency for specific roles)',
    'dependencies': ['task_1'],
    'priority': 7}]
    """



# state_structure = {'tasks': [{'task_id': 'task_1', 'task_description': 'Research the German job market (industries, cities with job opportunities, salary trends)', 'dependencies': [], 'priority': 1}, {'task_id': 'task_2', 'task_description': 'Create a tailored resume and cover letter compliant with German standards', 'dependencies': ['task_1'], 'priority': 2}, {'task_id': 'task_3', 'task_description': 'Prepare for job interviews (research common German interview practices, practice answers)', 'dependencies': ['task_2'], 'priority': 3}, {'task_id': 'task_4', 'task_description': 'Network with professionals in target industries (LinkedIn, local German professional groups)', 'dependencies': ['task_1', 'task_2'], 'priority': 4}, {'task_id': 'task_5', 'task_description': 'Apply for work visa (research required documents, application process, processing times)', 'dependencies': ['task_1'], 'priority': 5}, {'task_id': 'task_6', 'task_description': 'Utilize German job portals (StepStone, Indeed, Xing, local company career pages)', 'dependencies': ['task_1', 'task_2'], 'priority': 6}, {'task_id': 'task_7', 'task_description': 'Develop German language skills (certifications like Goethe Institute, language practice)', 'dependencies': ['task_1'], 'priority': 7}]}
# state_structure['tasks']
# parallel_task_routing_node_final_result = [{'agent': 'research_supervisor', 'task': {'task_id': 'research_german_job_market', 'task_description': 'Research the German job market, including in-demand industries and required qualifications', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'resume_cover_letter_german_standard', 'task_description': 'Create a tailored resume and cover letter according to German standards', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'networking_strategy_research', 'task_description': 'Conduct research on effective networking strategies using LinkedIn and German job portals to connect with industry professionals', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'GERMAN_JOB_COMMUNICATION', 'task_description': 'Provide resources and guidance for learning basic German language skills focused on job applications and workplace communication scenarios.', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'job_application_guidance', 'task_description': 'Provide step-by-step guidance on applying for jobs through German job portals (StepStone, Indeed Germany) and company career pages', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'interview_preparation_german_employers', 'task_description': 'Research cultural norms, common interview questions, visa requirements, and company backgrounds for German employers to prepare effective virtual/in-person interviews', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Research visa/work permit requirements for foreign professionals in Germany', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Identify and research potential employers in target industries in Germany', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'evaluate_relocation_costs_housing_quality_of_life_germany', 'task_description': 'Evaluate relocation costs, housing options, and quality of life in Germany', 'dependencies': [], 'priority': 1}}]


# state_list = []
# state_list.append(refinery_state['refined_statement'])
# state_list.append('I am fine and you?')
# from pprint import pprint
# print("\n\n".join(state_list))
# tool_calling_chain



async def subtask_tool_calling_worker(state: SupervisorOverallState) -> SupervisorOverallState:
    """
    Async worker that computes only SINGLE tool calling task
    """
    print(f"subtask_tool_calling_worker executed: {subtask_tool_calling_worker}")
    with open("/home/sdt/Workspace/mvai/AgenticRAG/subtask_tool_calling_worker_result.pkl", "wb") as f:
        pickle.dump(state, f)

    task_description = state['task']['task_description'] ##각각의 individual tasks
    subtask_tool_calling_worker_result = await tool_calling_chain.ainvoke({"messages": [{"type": "human", "content": task_description}]})
    tool_call_final_result = subtask_tool_calling_worker_result['messages'][-1].content
    # print(tool_call_final_result)
    return tool_call_final_result


async def tool_calling(state: SupervisorOverallState) -> SupervisorOverallState:
    print(f"tool_calling executed: {tool_calling}")
    with open("/home/sdt/Workspace/mvai/AgenticRAG/tool_calling_result.pkl", "wb") as f:
        pickle.dump(state, f)
    # print(f"tool_calling executed: {tool_calling}")
    subtask_tool_calling_worker_runnable = RunnableLambda(subtask_tool_calling_worker)
    subtask_tool_calling_worker_results = await subtask_tool_calling_worker_runnable.abatch(state['routing_results']) ##모든 태스크들이 list로 들어감 

    print(f"subtask_tool_calling_worker_results: {subtask_tool_calling_worker_results}")
    # return {"refined_statement": refined_statement}
    return {"tool_callings_result": subtask_tool_calling_worker_results}



# async def refine_results_node(state: SupervisorOverallState) -> SupervisorOverallState:
#     # 1. 이전 노드에서 생성된 '결과 리스트'를 가져옵니다.
#     tool_results_list = state['tool_callings_result'] 
    
#     # 2. 이 리스트의 '각 항목'을 입력으로 삼아 abatch를 호출합니다.
#     # (refine_chain이 개별 항목을 처리하는 Runnable이라고 가정)
#     refined_results = await refine_chain.abatch(tool_results_list)
    
#     return {"refined_results": refined_results}


class Evaluation(BaseModel):
    evaluation: Literal["good", "bad"] = Field(description="The evaluation result, either 'good' or 'bad'")

# tool_calling_evaluator_llm_chain = tool_calling_evaluator_prompt_template | llm.with_structured_output(SupervisorOverallState)

async def tool_calling_result_evaluator(state: SupervisorOverallState) -> SupervisorOverallState: ##Literal["good", "bad"]:
    print("tool_calling_result_evaluator executed")
    user_query = state['messages'][-1].content
    tool_call_results_list = state['tool_callings_result']
    tasks = state['tasks']
    # print(f"tasks: {tasks}")
    # tasks_list = tasks.get('tasks', [])



    tool_calling_evaluator_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
                    You are a precisely correct evaluator of tool calling result. You will check user's original query, 
                    task description of a task derived from the user's query, and tool calling result based on task description.
                    Your job is to check if tool calling result is appropriate to the user's query and the task description.
                    If the result is appropriate, you MUST say good. If the result is not relevant and good enough, you MUST say bad. So basically you can choose only one answer from the list below:
                    
                    ["good", "bad"]


                """),  
        ("user", """

        [User's original query]
        {user_query}

        [Task description for the tool calling]
        {task_description}

        [Tool calling result]
        {tool_calling_result}


        Tell me if the tool calling result is appropriate given user query and task description which is inferred from user query. 
        """

        
        
        
        )  

    ])
    tool_calling_evaluator_llm_chain = tool_calling_evaluator_prompt_template | llm.with_structured_output(Evaluation)

    list_len = len(tasks)
    inputs_for_batch = []
    for i in range(list_len):
        inputs_for_batch.append({
            "user_query": user_query,  # [상수] 모든 항목에 동일한 쿼리 삽입
            "task_description": tasks[i]['task_description'], # [변수]
            "tool_calling_result": tool_call_results_list[i] # [변수]
        })
        print(tool_call_results_list[i])

    evaluation_results = await tool_calling_evaluator_llm_chain.abatch(inputs_for_batch)

    # subtask_tool_calling_worker_result = await tool_calling_evaluator_llm_chain.ainvoke({"user_query": user_query,
    #                                                                                      "tool_calling_result": tool_calling_result,
    #                                                                                      "task_description": str(task_description)
    #                                                                                     })

    return {"tool_calling_eval": evaluation_results}






# def result_concatnater(state: SupervisorOverallState) -> SupervisorOverallState:
#     state_list = []
#     # state_list2 = []
#     state_list.append(state['refined_statement'].content)
#     state_list.append(str(state["routing_results"]))
#     with open("/home/sdt/Workspace/mvai/AgenticRAG/test.pkl", "wb") as f:
#         pickle.dump(state_list, f)
#     print(f"state_list: {state_list}")
#     # state_list2.append(state['refined_statement']['messages'])
#     # state_list2.append(str(state["routing_results"]))
#     final_statement = "\n\n".join(state_list)
#     print(f"final_statement: {final_statement}")
#     return {"final_statement": final_statement}



def result_concatnater(state: SupervisorOverallState) -> SupervisorOverallState:
    print(f"result_concatnater executed: {result_concatnater}")
    state_list = []
    # state_list2 = []
    state_list.append(state['refined_statement'].content)
    state_list.append(str(state["tool_callings"]))
    with open("/home/sdt/Workspace/mvai/AgenticRAG/result_concatnater.pkl", "wb") as f:
        pickle.dump(state_list, f)
    print(f"state_list: {state_list}")
    # state_list2.append(state['refined_statement']['messages'])
    # state_list2.append(str(state["routing_results"]))
    final_statement = "\n\n".join(state_list)
    print(f"final_statement: {final_statement}")
    return {"final_statement": final_statement}


# final_statement = [AIMessage(content='Let me rephrase your question!  \nYour goal is to find a job in Germany. What specific steps or actions should you take to make this happen?  \nIs my understanding correct?', additional_kwargs={}, response_metadata={'model': 'qwen3:8b', 'created_at': '2025-11-04T02:21:25.093749871Z', 'done': True, 'done_reason': 'stop', 'total_duration': 51657378989, 'load_duration': 65300947, 'prompt_eval_count': 233, 'prompt_eval_duration': 4709188084, 'eval_count': 421, 'eval_duration': 46754690176, 'model_name': 'qwen3:8b', 'model_provider': 'ollama'}, id='lc_run--5d9344b5-490d-4f88-ac00-2ee0bea906a9-0', usage_metadata={'input_tokens': 233, 'output_tokens': 421, 'total_tokens': 654}), "[{'agent': 'research_supervisor', 'task': {'task_id': 'research_german_job_market', 'task_description': 'Research the German job market and industry demand', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Prepare a tailored resume and cover letter for German employers', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Build professional network through LinkedIn and industry events', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'JOB_PORTAL_APPLICATION_RESEARCH', 'task_description': 'Research and guide user on applying for jobs through German job portals like StepStone and Indeed Germany', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Understand visa/work permit requirements for foreign workers', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': '1', 'task_description': 'Prepare for job interviews with German cultural norms and language skills', 'dependencies': [], 'priority': 1}}, {'agent': 'research_supervisor', 'task': {'task_id': 'RELOCATION_LOGISTICS_GERMANy', 'task_description': 'Explore relocation logistics and cost of living in Germany', 'dependencies': [], 'priority': 1}}]"]


#==========================================================================================================================================================
#==========================================================================================================================================================
#==========================================================================================================================================================

# # 1. 'data.pkl' 파일을 'rb' 모드로 엽니다.
# with open("/home/sdt/Workspace/mvai/AgenticRAG/test.pkl", 'rb') as f:
#     # 2. 파일에서 데이터를 불러와(load) loaded_data 변수에 할당합니다.
#     loaded_data = pickle.load(f)
# loaded_data[0].content
# loaded_data[1]
# "\n\n".join(loaded_data)


# with open("/home/sdt/Workspace/mvai/AgenticRAG/subtask_tool_calling_worker.pkl", 'rb') as f:
#     subtask_tool_calling_worker_data = pickle.load(f)
# subtask_tool_calling_worker_data['task']['task_description']
# with open("/home/sdt/Workspace/mvai/AgenticRAG/tool_calling.pkl", 'rb') as f:
#     tool_calling_data = pickle.load(f)


# with open("/home/sdt/Workspace/mvai/AgenticRAG/tool_calling_result.pkl", 'rb') as f:
#     tool_calling_data = pickle.load(f)
# tool_calling_data['tool_callings_result'][0]['tool_calling_result']['messages'][-1]
# for i in tool_calling_data['tool_callings_result']:
#     print(i)
#     print("----")
#     print("----")

# tool_calling_data['routing_results'][0]['agent']
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

#==========================================================================================================================================================
#==========================================================================================================================================================
#==========================================================================================================================================================





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
thread_id = str(2222333333)
user_id = str(33333322222)

# thread_id = str(uuid.uuid4())
# user_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}


## subgraph
parallel_builder = StateGraph(SupervisorOverallState)
parallel_builder.add_node("decomposer", task_decompose_node, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
parallel_builder.add_node("parallel_router", parallel_task_routing_node, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
parallel_builder.add_node("tool_calling", tool_calling, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
# parallel_builder.add_node("tool_calling_evaluator", tool_calling_result_evaluator, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
"""
평가자 노드 (evaluator_node): [핵심] 별도의 '평가용 LLM'을 사용하여, [사용자 원본 질문], [Tool 호출 내용], [Tool 실행 결과] 3가지를 보고 "이 결과가 유용한가?"를 판단하여 state를 업데이트합니다.
"""
# parallel_builder.add_node("statement_refinery", statement_refinery, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))


## 서브그래프의 내부 흐름 정의
parallel_builder.add_edge(START, "decomposer")
# parallel_builder.add_edge(START, "statement_refinery")
parallel_builder.add_edge("decomposer", "parallel_router")

## 두 병렬 브랜치가 모두 서브그래프의 END를 가리키도록 함
parallel_builder.add_edge("parallel_router", "tool_calling")
parallel_builder.add_edge("tool_calling", END)
# parallel_builder.add_edge("tool_calling", "tool_calling_evaluator")
# parallel_builder.add_edge("tool_calling_evaluator", END)
# parallel_builder.add_conditional_edges("tool_calling_evaluator", path=tool_calling_result_evaluator, path_map={"bad": "tool_calling","good": END})
# parallel_builder.add_edge("statement_refinery", END)


parallel_graph = parallel_builder.compile()



## master graph
master_builder = StateGraph(SupervisorOverallState)

## 서브그래프 자체를 'parallel_step'이라는 이름의 단일 노드로 추가
master_builder.add_node("parallel_step", parallel_graph)
## 결과 취합 노드 추가
# master_builder.add_node("result_concatnater", result_concatnater, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))

## 메인 그래프는 이제 단순한 순차 흐름이 됨
master_builder.set_entry_point("parallel_step")
master_builder.add_edge("parallel_step", END)
# master_builder.add_edge("parallel_step", "result_concatnater")
# master_builder.add_edge("result_concatnater", END)


master_graph = master_builder.compile(checkpointer=checkpointer, store=in_memory_store)


# in_memory_store
# dir(checkpointer)
# checkpointer.list(config=)



"""
########################섭그래프없이 테스트해보는 것 ########################
"""


master_builder = StateGraph(SupervisorOverallState)
master_builder.add_node("decomposer", task_decompose_node, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
master_builder.add_node("statement_refinery", statement_refinery, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
master_builder.add_node("parallel_router", parallel_task_routing_node, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
master_builder.add_node("tool_calling", tool_calling, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))
master_builder.add_node("tool_calling_evaluator", tool_calling_result_evaluator, retry_policy=RetryPolicy(), cache_policy=CachePolicy(ttl=120))



master_builder.add_edge(START, "decomposer")
master_builder.add_edge(START, "statement_refinery")
master_builder.add_edge("decomposer", "parallel_router")
master_builder.add_edge("parallel_router", "tool_calling")
master_builder.add_edge("tool_calling", "tool_calling_evaluator")
master_builder.add_edge("tool_calling_evaluator", END)
master_builder.add_edge("statement_refinery", END)



master_builder.add_edge("tool_calling", "tool_calling_evaluator")
master_builder.add_edge("tool_calling_evaluator", END)
# master_builder.add_edge("statement_refinery", "result_concatnater")
# master_builder.add_edge("result_concatnater", END)


master_graph = master_builder.compile(checkpointer=checkpointer, store=in_memory_store)


# Show the agent
from IPython.display import Image, display
display(Image(master_graph.get_graph(xray=True).draw_mermaid_png()))

from time import time
start_time = time()
result = await master_graph.ainvoke({"messages": [{"type": "human", "content": "What should I do to find a job in Germany?"}]}, config)
# result = await master_graph.ainvoke({"messages": [{"type": "human", "content": "What is the best visa for work in Germany?"}]}, config)
finishe_time = time()
time_taken = finishe_time - start_time
print(f"{time_taken} seconds") ##99.43021845817566 seconds


result.keys()

result['messages']
result['tasks']
result['refined_statement']
result['routing_results']

result['tasks'][-1]['task_description']
result['tool_callings_result'][-1]
result['tool_calling_eval']

result.keys()
result['messages']
result['refined_statement']
result['tasks'][0]['task_description']
result['routing_results']
result['tool_calling_result']
result['tool_callings_result']
result['tool_callings_result'][0]['tool_calling_result']['messages'][-1].content
result['tasks']
result.keys()
result['routing_results'][0]['agent']

result['messages']
result['tasks']
result['refined_statement']
result['tool_callings_result'][0]['tool_calling_result']['messages'][-1].tool_calls
result['tool_callings_result'][1]['tool_calling_result']['messages'][-1]
result.keys()
result['final_statement']
zz = result['routing_results']
result['tasks']
result['tool_calling_result']


type(result['tool_callings_result'][0])



with open("/home/sdt/Workspace/mvai/AgenticRAG/final_result.pkl", "wb") as f:
    pickle.dump(result, f)



with open("/home/sdt/Workspace/mvai/AgenticRAG/final_result.pkl", 'rb') as f:
    final_result = pickle.load(f)

result['messages']
result['tasks']
result['tool_callings']
result['tool_callings'][0]['tool_calling']['messages'][3]


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
