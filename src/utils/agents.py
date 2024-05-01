from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.agents import create_openai_tools_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor

load_dotenv()

def get_executor(website):
    """Function to get the executor for the given website

    Args:
        website (str): The website to get the executor for

    Returns:
        AgentExecutor: The executor for the given website
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # returning the executors
    if website == "Wikipedia":
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
        wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
        wiki_tools = [wiki_tool]
        wiki_agent = create_openai_tools_agent(llm, wiki_tools, prompt)
        return AgentExecutor(agent=wiki_agent, tools=wiki_tools, verbose=True)
    elif website == "Langsmith":
        loader = WebBaseLoader("https://docs.smith.langchain.com/")
        docs = loader.load()
        documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
        retriever = vectordb.as_retriever()
        retriever_tool = create_retriever_tool(retriever, "langsmith_search", description="Search the Langsmith documents")
        langsmith_tools = [retriever_tool]
        langsmith_agent = create_openai_tools_agent(llm, langsmith_tools, prompt)
        return AgentExecutor(agent=langsmith_agent, tools=langsmith_tools, verbose=True)
    elif website == "Arxiv":
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        arxiv_tools = [arxiv_tool]
        arxiv_agent = create_openai_tools_agent(llm, arxiv_tools, prompt)
        return AgentExecutor(agent=arxiv_agent, tools=arxiv_tools, verbose=True)