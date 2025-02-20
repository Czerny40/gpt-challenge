from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.title("🦜🔗 Streamlit is 🔥")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
st.sidebar.markdown("https://github.com/Czerny40/gpt-challenge")


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource(show_spinner="파일을 분석하고있어요...")
def embed_file(file):

    file_name = file.name
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = {}

    if file_name in st.session_state.embeddings:
        return st.session_state.embeddings[file_name]


    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n"],
        length_function=len,
    )

    file_loader = UnstructuredFileLoader(file_path)
    docs = file_loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=openai_api_key
    )

    # cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    # 세션에 임베딩 저장
    st.session_state.embeddings[file_name] = embeddings

    return retriever


# 메시지 표시 함수
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# 메시지 저장 함수
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# 채팅 기록 불러오기 함수
def load_chat_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_documents(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        당신은 문서 분석 전문가입니다. 다음 규칙을 따르세요:
        1. 주어진 문서의 내용을 기반으로 답변하세요
        2. 제공된 정보가 있다면 반드시 그 내용을 포함하여 답변하세요.
        3. 모르는 내용은 솔직히 모른다고 하세요

        문서 내용:
        {context}
        """,
        ),
        ("human", "{question}"),
    ]
)

st.markdown(
    """
안녕하세요!

DocumentGPT는 업로드한 문서를 기반으로 질문에 답해드리는 서비스입니다!

먼저 사이드바에서 파일을 업로드해주세요!
"""
)

st.sidebar.markdown(
    """
### 지원하는 파일 형식
- PDF (.pdf)
- 텍스트 파일 (.txt)
- Word 문서 (.docx)

"""
)

with st.sidebar:
    file = st.file_uploader(
        ".txt, .pdf, .docx 파일을 업로드해주세요", type=["txt", "pdf", "docx"]
    )

if file and openai_api_key:
    retriever = embed_file(file)

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.1,
        api_key=openai_api_key,
    )

    send_message("무엇이든 물어보세요!", "ai", save=False)
    load_chat_history()
    message = st.chat_input("업로드한 문서에 대해 무엇이든 물어보세요")

    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever
                | RunnableLambda(lambda docs: format_documents(docs)),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")
            

else:
    if not openai_api_key:
        st.warning("사이드바에 OpenAI API 키를 입력해주세요")
    st.session_state["messages"] = []
