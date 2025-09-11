from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama


class MovieRecommendation(BaseModel):
    title: str = Field(..., description="The title of the movie")
    genre: str = Field(..., description="The genre of the movie")
    rating: float = Field(..., description="Rating from 0.0 to 10.0")


parser = PydanticOutputParser(pydantic_object=MovieRecommendation)


prompt = PromptTemplate(
    template="Give me a movie recommendation in this format:\n{format_instructions}\n\nQuestion: {question}",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = ChatOllama(model="llama3.1", temperature=0.1) ## qwen3:8b 다운받아놓음. 한국어 실력이 더 좋다고 함.
# llm = ChatOpenAI(temperature=0)  # or any LLM supporting LangChain interface

prompt = PromptTemplate(
    template=(
        "You are a helpful assistant.\n"
        "Please answer the following question by providing a JSON object with the following format:\n"
        "{format_instructions}\n\n"
        "Question: {question}"
    ),
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


_input = prompt.format(question="Recommend a sci-fi movie.")
output = llm.invoke(_input)

print("Raw LLM output:\n", output.content)

parsed_output = parser.parse(output.content)
print("\nParsed object:\n", parsed_output)



# ChatOllama에서도 PydanticOutputParser는 그대로 사용 가능함

# 단, 명확한 JSON 출력 지시가 프롬프트에 꼭 포함되어야 함

# 출력은 .content에서 받아 파싱