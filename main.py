from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_supervisor import create_supervisor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

# ## should change username, passcode, host, port, database names to real ones.
# DB_URI = "postgresql://user:password@localhost:5432/dbname" 
# # checkpointer = PostgresSaver.from_conn_string(DB_URI)
# checkpoint_saver = PostgresSaver(db_uri=DB_URI, table_name="agent_checkpoints")


llm = ChatOllama(model="qwen3:8b", base_url="http://127.0.0.1:11434")
checkpointer = SqliteSaver.from_file("langgraph_checkpoints.sqlite")
 
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