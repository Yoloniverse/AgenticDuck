
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate



# 1. ChatPromptTemplate 객체 생성
# from_messages를 사용하여 메시지 템플릿을 정의합니다.
prompt_router = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a master planner. Given the user's request, create a concise, step-by-step plan
        to achieve the goal. Do not execute the plan, just create it.
        """),
        ('human'), "{query}"
    ]
)




router_system_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a precise router which decides which agent to call based on input data.
                  You must assign an agent only in the list and only one agent for one task.
                  <<EXAMPLE>>
               """),  
    ("user", "{input}")  
])

