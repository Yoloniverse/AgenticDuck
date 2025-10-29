from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph_supervisor import create_supervisor
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
 

dir(llm)
router_system_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a precise router which decides which agent to call based on input data.
                  You must assign an agent only in the list and only one agent for one task.
                  <<EXAMPLE>>
               """),  
    ("user", "{input}")  
])

router_llm_chain = router_system_prompt_template | llm


sql_agent_system_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an sql agent who is responsible for various sql tasks.
                  You must choose a tool only within tools that is bound to you in sql_llm_chain
                  <<EXAMPLE>>
               """),  
    ("user", "{input}")  
])

llm.bind_tools([sql_select_all, sql_groupby])
sql_llm_chain = sql_agent_system_prompt_template | llm




def sql_agent(state: sqlAgentState):
    """A specialized agent that performs sql query."""
    messages = state['messages']
    

    llm_with_tools = llm.bind_tools([sql_select_all, sql_groupby])
    
    response = llm_with_tools.invoke(messages)
    
    ## The response will contain a tool_call, which we can then execute.
    tool_calls = response.tool_calls
    tool_output = None
    if tool_calls:

        ## For simplicity, only handle the first tool call.
        tool_call = tool_calls[0]
        tool_output = search_the_web.invoke(tool_call['args'])
    
    # Update the state with the new messages and tool output.
    new_messages = [response, HumanMessage(content=tool_output)]
    return {"messages": new_messages, "next_node": "router"}



def router_node(state: routerAgentState):
    """Decides which agent to call next based on the plan and state."""
    print("---ROUTER: Deciding the next step...---")
    messages = state['messages']
    last_message = messages[-1].content
    
    if "final answer" in last_message.lower() or "hello" in last_message.lower():
        print("---ROUTER: Routing to finalizer---")
        return "finalizer"
    elif "code" in last_message.lower() or "generate code" in last_message.lower():
        print("---ROUTER: Routing to code agent---")
        return "code_agent"
    elif "search" in last_message.lower() or "research" in last_message.lower():
        print("---ROUTER: Routing to research agent---")
        return "research_agent"
    else:
        # A fallback to the research agent for unclassified tasks.
        print("---ROUTER: No clear path, routing to research agent as fallback---")
        return "research_agent"





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

supervisor = create_supervisor(
    agents=[sql_agent, rag_agent, web_search_agent],
    model=llm,
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