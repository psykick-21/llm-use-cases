from langchain_openai import ChatOpenAI # openai chat models
from langchain_core.prompts import ChatPromptTemplate # to create a chat qa_prompt
from langchain_community.llms import Ollama, HuggingFaceEndpoint # to use the llama3 and gemma models
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser # to parse the output
import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from src.utils import Utils, agents
from fastapi import FastAPI
import uvicorn
from langserve import add_routes

load_dotenv()

# setting up langsmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true" # enable tracking in langsmith, the results can be seen on the dashboard
os.environ["LANGCHAIN_PROJECT"] = "multiutility-llm-app" # project name for tracking

# creating a qa_prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

output_parser = StrOutputParser() # initialize the output parser

# streamlit app
st.title("ChatBot")

with st.sidebar:
    service = option_menu("Select Service", ["Home","Multi LLM QnA","Chat With Website"], default_index=0)    

# >----------------- Home -----------------< #
if service == "Home":
    st.write("Welcome to the Chatbot App")
    st.write("Select a service from the sidebar")
    
# >----------------- Multi LLM QnA -----------------< #
elif service == "Multi LLM QnA":
    # selecting the LLM
    with st.container(border=True):
        selected_llm = Utils.render_llm_select_radio()
        
    # initializing the selected llm
    llm = Utils.initialize_qa_llm(selected_llm)
    
    # initializing the chain
    chain = qa_prompt|llm|output_parser # create a chain from qa_prompt to llm to output parser

    # taking the input from the user
    input_text = st.text_input("Enter your question")

    # invoke the chain with user input
    if (input_text) and input_text!="":
        st.write(chain.invoke({"question":input_text}))

# >----------------- Chat With Website -----------------< #
elif service == "Chat With Website":
    with st.container(border=True):
        selected_website = Utils.render_website_select_radio()
    input_text = st.text_input("Enter your question")

    if selected_website == "Wikipedia":
        executor = agents.get_executor(selected_website)
        if (input_text) and input_text!="":
            st.write(executor.invoke({"input":input_text})['output'])
    elif selected_website == "Langsmith":
        executor = agents.get_executor(selected_website)
        if (input_text) and input_text!="":
            st.write(executor.invoke({"input":input_text})['output'])
    elif selected_website == "Arxiv":
        executor = agents.get_executor(selected_website)
        if (input_text) and input_text!="":
            st.write(executor.invoke({"input":input_text})['output'])