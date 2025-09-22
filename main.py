from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_supervisor import create_supervisor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ## should change username, passcode, host, port, database names to real ones.
# DB_URI = "postgresql://user:password@localhost:5432/dbname" 
# # checkpointer = PostgresSaver.from_conn_string(DB_URI)
# checkpoint_saver = PostgresSaver(db_uri=DB_URI, table_name="agent_checkpoints")


llm = ChatOllama(model="qwen3:8b", base_url="http://127.0.0.1:11434")
checkpointer = SqliteSaver.from_file("langgraph_checkpoints.sqlite")
 


planner_system_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a master planner. Given the user's request, create a concise, step-by-step plan.
        <<EXAMPLE>>
        """),
        ('human'), "{query}"
    ]
)

def create_planner_agent(llm, tools) -> Runnable:
    """Creating a planner agent that smartly breaks down a question of user into sub-questions"""
    tools = [planner_func]
    # This is a conceptual example. A real agent would be more complex.
    llm_with_tools = llm.bind_tools(tools) 
    llm_with_tools_chain = planner_system_prompt_template | llm_with_tools
    return llm_with_tools_chain

planner_agent = create_planner_agent(llm)







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
web_search_agent = create_react_agent(
    model=llm,
    tools=[book_hotel],
    prompt="",
    name="web_search_agent"
)

##https://docs.langchain.com/oss/python/langchain/short-term-memory#pre-model-hook
##https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/
supervisor = create_supervisor(
    agents=[sql_agent, rag_agent, web_search_agent],
    model=llm,
    pre_model_hook=[planner_agent],
    prompt=(
        """너는 사용자 질문을 읽고 분석하여 필요한 기능을 선택하는 역할이야.
            [중요 원칙]

            [조건]

                """
    )
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