import os
import json
import streamlit as st
from openai import OpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.document_loaders import WebBaseLoader
import time
from duckduckgo_search import DDGS

st.title("🔍 Research Assistant")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
st.sidebar.markdown("https://github.com/Czerny40/gpt-challenge")

client = OpenAI(api_key=openai_api_key)


def create_assistant():
    if "assistant" in st.session_state:
        return st.session_state["assistant"]

    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_ddg_results",
                "description": "Use this tool to perform web searches using the DuckDuckGo search engine.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_wiki_results",
                "description": "Use this tool to perform searches on Wikipedia.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ddg_scraper",
                "description": "Use this to get the content of a link found in DuckDuckGo for research.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to scrape.",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
    ]

    assistant = client.beta.assistants.create(
        name="Research Assistant",
        instructions="""
        You are a research expert. Use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 
        When you find a relevant website through DuckDuckGo, scrape the content from that website. Use this scraped content to thoroughly research and formulate a detailed answer to the question. 
        Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized and detailed, and include citations with links (URLs) for all sources used.
        Your research should be saved to a .txt file, and the content should match the detailed findings provided. Make sure to include all sources and relevant information.
        The information from Wikipedia must be included.
        Ensure that the final .txt file contains detailed information, all relevant sources, and citations.
        """,
        model="gpt-4o-mini",
        tools=functions,
    )

    st.session_state["assistant"] = assistant
    return st.session_state["assistant"]


def get_ddg_results(inputs):
    query = inputs["query"]
    ddgs = DDGS()
    results = ddgs.text(query)
    time.sleep(0.5)
    return results


def get_wiki_results(inputs):
    query = inputs["query"]
    wikipedia_api = WikipediaAPIWrapper()
    wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_api)
    return wikipedia.run(query)


def ddg_scraper(inputs):
    url = inputs["url"]
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[0].page_content.replace("\n", "")


functions_map = {
    "get_ddg_results": get_ddg_results,
    "ddg_scraper": ddg_scraper,
    "get_wiki_results": get_wiki_results,
}


def get_thread_id():
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "assistant",
                    "content": "Hi, How can I help you?",
                }
            ]
        )
        st.session_state["thread_id"] = thread.id
    return st.session_state["thread_id"]


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def auto_save_results(content):
    filename = "research_result.txt"

    os.makedirs("results", exist_ok=True)

    with open(os.path.join("results", filename), "w", encoding="utf-8") as f:
        f.write(content)

    st.success(f"Results saved to {filename}")


def start_run(thread_id, assistant_id, content):
    if "run" not in st.session_state or get_run(
        st.session_state["run"].id, get_thread_id()
    ).status in (
        "expired",
        "completed",
    ):
        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )

        st.session_state["run"] = run
    else:
        print("already running")

    run = st.session_state["run"]

    with st.status("Processing..."):
        while get_run(run.id, get_thread_id()).status == "requires_action":
            submit_tool_outputs(run.id, get_thread_id())

    print(f"done, {get_run(run.id, get_thread_id()).status}")
    final_message = get_messages(get_thread_id())[-1]
    if get_run(run.id, get_thread_id()).status == "completed":
        with st.chat_message(final_message.role):
            st.markdown(final_message.content[0].text.value)

        auto_save_results(final_message.content[0].text.value)
        print(final_message)
    elif get_run(run.id, get_thread_id()).status == "failed":
        with st.chat_message("assistant"):
            st.markdown("Sorry. I failed researching. Try Again later :()")


def get_messages(thread_id):
    messages = list(
        client.beta.threads.messages.list(
            thread_id=thread_id,
        )
    )
    return list(reversed(messages))


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        st.write(f"Calling function: {function.name} with arg {function.arguments}")
        print(f"Calling function: {function.name} with arg {function.arguments}")
        output = functions_map[function.name](json.loads(function.arguments))
        outputs.append(
            {
                "output": output,
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs_and_poll(
        run_id=run_id, thread_id=thread_id, tool_outputs=outputs
    )


def send_message(_content: str):
    thread_id = get_thread_id()
    start_run(thread_id, assistant_id, _content)


if not openai_api_key:
    st.error("OpenAI API 키를 입력해주세요.")
else:
    query = st.chat_input("궁금한 점을 물어보세요.")
    client = OpenAI(api_key=openai_api_key)
    assistant_id = create_assistant().id

    for message in get_messages(get_thread_id()):
        with st.chat_message(message.role):
            st.markdown(message.content[0].text.value)

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        send_message(query)
