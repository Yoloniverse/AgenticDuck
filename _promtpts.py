

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama



llm = ChatOllama(model="qwen3:8b", base_url="http://127.0.0.1:11434")

planner_system_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a master planner. Given the user's request, create a concise, step-by-step plan.
        <<EXAMPLE>>
        """),
        ('human'), "{query}"
    ]
)
chain = prompt_router | llm
chain.invoke({'query','i wanna know weather in korea and also where to go'})


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