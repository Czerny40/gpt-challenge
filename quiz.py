import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.callbacks import StreamingStdOutCallbackHandler


function = {
    "name": "get_questions",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        You are a helpful assistant that is role playing as a teacher.
            
        Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
        
        Each question should have 4 answers, three of them must be incorrect and one should be correct.

        You must use Korean.

        The difficulty of the problem is '{level}'.
            
        Context: {context}
    """,
        )
    ]
)


def format_documents(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_resource(show_spinner="파일을 분석하고 있어요...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )

    file_loader = UnstructuredFileLoader(file_path)
    docs = file_loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_resource(show_spinner="문제를 만들고 있어요...")
def run_quiz_chain(_docs, topic, level):
    chain = prompt | llm
    return chain.invoke({"context": _docs, "level": level})


@st.cache_resource(show_spinner="문제를 만들고 있어요...")
def wiki_search(keyword):
    retriever = WikipediaRetriever(top_k_results=3, lang="ko")
    return retriever.get_relevant_documents(keyword)


st.title("🔗 QuizGPT")


with st.sidebar:
    docs = None
    topic = None
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    choice = st.selectbox(
        "사용할 항목을 골라주세요.",
        ("File", "Wikipedia"),
    )
    if choice == "File":
        file = st.file_uploader(
            ".docx, .txt, .pdf 파일만 업로드해주세요.",
            type=["docx", "txt", "pdf"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("검색할 키워드를 입력해주세요.")
        if topic:
            docs = wiki_search(topic)
    level = st.selectbox("난이도", ("쉬움", "어려움"))
    st.write("https://github.com/Czerny40/gpt-challenge")


if not docs:
    st.markdown(
        """
    안녕하세요!

    QuizGPT는 사용자의 학습을 위해 업로드한 문서 혹은 위키피디아를 바탕으로 GPT가 문제를 만들어주는 서비스입니다!

    먼저 사이드바에서 파일을 업로드, 또는 검색할 단어를 입력해주세요!
    """
    )
else:
    if not openai_api_key:
        st.error("OpenAI API 키를 입력해주세요.")
    else:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.2,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            api_key=openai_api_key,
        ).bind(
            function_call={
                "name": "get_questions",
            },
            functions=[function],
        )

        response = run_quiz_chain(docs, topic if topic else file.name, level)
        response = response.additional_kwargs["function_call"]["arguments"]

        with st.form("questions_form"):
            questions = json.loads(response)["questions"]
            answers = []

            for idx, question in enumerate(questions):
                st.markdown(f'Q{idx+1}. {question["question"]}')
                answer = st.radio(
                    "답을 고르세요.",
                    [answer["answer"] for answer in question["answers"]],
                    key=f"question_{idx}",
                )
                answers.append(answer)

            submitted = st.form_submit_button("제출")

            if submitted:
                correct_count = 0
                for idx, (question, user_answer) in enumerate(zip(questions, answers)):
                    if {"answer": user_answer, "correct": True} in question["answers"]:
                        st.success(f"Q{idx+1} 정답입니다!")
                        correct_count += 1
                    else:
                        st.error(f"Q{idx+1} 틀렸습니다.")

                if correct_count == len(questions):
                    st.balloons()
