from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


DUMMY_TEXT = """LangChain is a framework for developing applications powered by large language models (LLMs).

LangChain simplifies every stage of the LLM application lifecycle:

Development: Build your applications using LangChain's open-source building blocks, components, and third-party integrations. Use LangGraph.js to build stateful agents with first-class streaming and human-in-the-loop support.
Productionization: Use LangSmith to inspect, monitor and evaluate your chains, so that you can continuously optimize and deploy with confidence.
Deployment: Turn your LangGraph applications into production-ready APIs and Assistants with LangGraph Cloud."""

CHUNK_SIZE = 35
CHUNK_OVERLAP = 5


def get_model():
    # llm = ChatOllama(model="gemma3", temperature=0)
    llm = ChatOllama(model="qwen3", temperature=0)

    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    return llm