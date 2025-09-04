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
docs = vector_store.similarity_search("metrics of glass model", k=2)
docs[0].to_json()



#######################################
from langchain.vectorstores import Chroma
import re
from pathlib import Path
from typing import List
from langchain_core.documents import Document
import os
# 임베딩: HuggingFaceEmbeddings (SentenceTransformers)
from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder

# === 설정 ===
TXT_PATH = "/home/sdt/Workspace/dykim/Langraph/AgenticDuck/hr_files/appr_process.txt"         # 당신의 txt 경로
CHROMA_DIR = "./chroma_db"         # Chroma 영구 저장 경로
os.makedirs(CHROMA_DIR, exist_ok=True)
COLLECTION = "approval_guide" 
EMBEDDING_MODEL = "BAAI/bge-m3" # 질의 인스트럭션을 자동으로 붙여서 질문-문서 정합성이 좋아짐
#EMBEDDING_MODEL = "jhgan/ko-sbert-nli"  # 한국어 특화 임베딩 모델
DEVICE = "cuda"                         # 'cuda' 가능
NORMALIZE = True                       # 코사인 유사도에 적합

# === 로더 & 청크 분할 ===
def load_and_split_by_rule(path: str) -> List[Document]:
    text = Path(path).read_text(encoding="utf-8")
    # 줄 전체가 --- (3개 이상) 로만 이루어진 구분선 기준으로 split
    chunks = re.split(r"(?m)^\s*-{3,}\s*$", text)
    docs = []
    for i, raw in enumerate(chunks):
        chunk = raw.strip()
        if not chunk:
            continue
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "source_path": str(Path(path).resolve()),
                },
            )
        )
    return docs

docs = load_and_split_by_rule(TXT_PATH)
print(f"총 청크 수: {len(docs)}")


# === 임베딩 준비 ===
emb = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": NORMALIZE},
)

# === 3) ChromaDB에 적재(영구 저장) ===
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=emb,
    collection_name=COLLECTION,
    persist_directory=CHROMA_DIR,
)

vectordb.persist()
print(f"Chroma 인덱스 저장 완료: {CHROMA_DIR} / collection={COLLECTION}")


# ===== (다음 실행부터) 로드 & 검색 =====
vectordb = Chroma(
    collection_name=COLLECTION,
    persist_directory=CHROMA_DIR,  # ← 동일 경로
    embedding_function=emb,            # ← 동일 임베딩으로 로드
)
query = "이 문서의 목차을 알려줘"
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
hits = retriever.get_relevant_documents(query)
documents = []
for h in hits:
    print(h.page_content[:].replace("\n"," "))

################ 리랭커 테스트 ############
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Rerank 모델
reranker = CrossEncoder(RERANK_MODEL)
query_doc_pairs = [(query, doc.page_content) for doc in hits]
scores = reranker.predict(query_doc_pairs)

scored_docs = list(zip(scores, hits))
scored_docs.sort(key=lambda x: x[0], reverse=True)

# 상위 k개 문서 반환
top_k = 3
reranked_docs = [doc for score, doc in scored_docs[:top_k]]






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


