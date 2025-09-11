from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import langchain_text_splitters

"""
문서들을 vector데이터에 저장하는 코드 (아직 베이직한 코드)
할일: reranker 모델을 추가 
"""


ollama_embeddings = OllamaEmbeddings(
    model="llama3.1")

loader = PyPDFLoader("/home/sdt/Workspace/mvai/AgenticDuck/data/glass.pdf")

def PDFLoad_VectorStoreText(pdf_path: str) -> str:
    """
    Load PDF file, then extract text from it. 
    After that, split the texts and then embed the text using an embedding model then save it to vector database.
    """

    # Text splitter setting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                chunk_overlap=100, 
                                                separators=["\n\n", "\n", ".", " ", ""] +
                                                [RecursiveCharacterTextSplitter.get_separators_for_language(language.value) for language in Language if language.value != 'perl']
                                                    ) ## better in general

    # async for page in loader.alazy_load():
    #     pages.append(page)
    pages = loader.lazy_load()
    documents = text_splitter.split_documents(pages)
    vector_store = Chroma.from_documents(documents, 
                                         ollama_embeddings,
                                         persist_directory="./chromadb")
    vector_store.persist()
    return "PDF texts' embeddings are all saved in the vectorDB"

PDFLoad_VectorStoreText("/home/sdt/Workspace/mvai/AgenticDuck/data/glass.pdf")


##DB열어서 저장된 벡터들 확인
# 기존 디렉토리에서 벡터 DB 로드
vector_store = Chroma(
    persist_directory="./chromadb",
    embedding_function=ollama_embeddings
)

retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "k": 5})
docs = retriever.invoke("what did the president say about ketanji brown jackson?")


docs = vector_store.similarity_search("metrics of glass model", k=2)
docs[0].to_json()








### rerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 모델 초기화
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

# 상위 3개의 문서 선택
compressor = CrossEncoderReranker(model=model, top_n=3)

# 문서 압축 검색기 초기화
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# 압축된 문서 검색
compressed_docs = compression_retriever.invoke("Word2Vec 에 대해서 알려줄래?")

# 문서 출력
pretty_print_docs(compressed_docs)











# dir(Chroma)
# # ## tiral to query text 
# # docs = vector_store.similarity_search("Who is the author of glass paper?", k=2)
# docs = vector_store.similarity_search("metrics of glass model", k=2)
# docs[0].to_json()

# dir(vector_store)
# vector_store.get('41015c4e-60fa-49ac-bffd-60a692450869')





















for page in loader.lazy_load():
    pages.append(page)

dir(pages[0])
print(f"{pages[0].metadata}\n")
print(pages[0].page_content)

from pprint import pprint
pages[0].metadata
pages[0].page_content
pprint(pages[0].page_content)



# Load the document, split it into chunks, embed each chunk and load it into the vector store.
# raw_documents = TextLoader('state_of_the_union.txt').load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


pages[0].page_content

## PDF 전체 페이지를 한번에 넣어서 페이지별로가 아닌, splitter 기준으로 청킹한다. 









for doc in docs:
    pprint(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')


## 원한다면 ensemble 리트리버를 만들어서 여러 곳에서 데이터를 쿼리 할 수 있다.
chroma_retriever = vector_store.as_retriever()
result = chroma_retriever.invoke("what did the president say about ketanji brown jackson?")











# ollama_embeddings = OllamaEmbeddings(
#     model="llama3.1",
# )


# loader = PyPDFLoader("/home/sdt/Workspace/mvai/AgenticRAG/data/glass.pdf")
# pages = []

# # async for page in loader.alazy_load():
# #     pages.append(page)


# for page in loader.lazy_load():
#     pages.append(page)

# dir(pages[0])
# print(f"{pages[0].metadata}\n")
# print(pages[0].page_content)

# from pprint import pprint
# pages[0].metadata
# pages[0].page_content
# pprint(pages[0].page_content)


# vector_store = Chroma.from_documents(pages, ollama_embeddings)
# docs = vector_store.similarity_search("What is grass?", k=2)

# for doc in docs:
#     pprint(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')


