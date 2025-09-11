
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

prompt_template.invoke({"topic": "cats"})



from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

prompt_template.invoke({"topic": "cats"})



from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])
prompt_template.invoke({"msgs": [HumanMessage(content="what??!")]})



prompt_template = ChatPromptTemplate([
    ("system", "You are a jerk."),
    ("placeholder", "{msgs}") # <-- This is the changed part
])
prompt_template.invoke({"msgs": [HumanMessage(content="what??!")]})







## How to prompting in LangChain/LangGraph
"""
####################################################################################################################################
1. 공식 문서 확인
대부분의 LLM 제공자(OpenAI, Anthropic, Mistral 등)는 프롬프트 형식이나 API 사용 예시를 공식 문서에 명시하고 있음.

예시:
Anthropic (Claude)
→ <system>, Human:, Assistant: 등의 토큰으로 구분.
→ Messages API 또는 구식 completion 방식 있음.


####################################################################################################################################

2. LangChain/LangGraph 프롬프트 템플릿 활용
LangGraph는 LangChain을 기반으로 동작하기 때문에, LangChain에서 제공하는 PromptTemplate, ChatPromptTemplate, 
MessagesPlaceholder 등을 이용하면 LLM마다 다른 포맷을 자동으로 처리할 수 있습니다.

from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="{input}")
])

LangGraph에서 LLM 노드에 이 prompt를 넣으면, 사용하는 LLM 타입(OpenAI, Anthropic 등)에 따라 자동으로 올바른 형식으로 직렬화됩니다.


####################################################################################################################################
3. LangChain 모델 wrapper를 활용하면 형식이 자동 처리됨.
## LangGraph에서 LLM 노드를 정의할 때 LangChain의 ChatOpenAI, ChatAnthropic 등을 사용하면 내부적으로 프롬프트 형식을 자동으로 맞춰줌.

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")  # OpenAI의 형식을 자동 사용
LangGraph 노드에서는 llm.invoke(prompt) 형태로 사용하면 프롬프트 포맷 걱정 없이 사용 가능.

####################################################################################################################################
4. 직접 모델에 맞게 포맷팅 (Advanced)

Anthropic 예시:

prompt = f""
<system>
You are a helpful assistant.
</system>
Human: {user_input}
Assistant:
""


"""



from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Hello, {name}. What can I do for you today?",
    input_variables=["name"]
)

filled_prompt = prompt.format(name="Alice")
print(filled_prompt)