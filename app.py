from langchain_openai import ChatOpenAI # openai chat models
from langchain_core.prompts import ChatPromptTemplate # to create a chat prompt
from langchain_community.llms import Ollama, HuggingFaceEndpoint # to use the llama3 and gemma models
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser # to parse the output
import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from src.utils import Utils

load_dotenv()

# setting up langsmith tracing
# os.environ["LANGCHAIN_TRACING_V2"] = "true" # enable tracking in langsmith, the results can be seen on the dashboard
# os.environ["LANGCHAIN_PROJECT"] = "multi-llm-chatbot" # project name for tracking

# creating a chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

output_parser = StrOutputParser() # initialize the output parser

# streamlit app
st.title("ChatBot")
# input_text = st.text_input("Enter your question") # input text by user

with st.sidebar:
    service = option_menu("Select Service", ["Home","Multi LLM Chatbot"], default_index=0)    

# >----------------- Home -----------------< #
if service == "Home":
    st.write("Welcome to the Chatbot App")
    st.write("Select a service from the sidebar")  
    
# >----------------- Multi LLM Chatbot -----------------< #
elif service == "Multi LLM Chatbot":
    # selecting the LLM
    with st.container(border=True):
        selected_llm = Utils.render_llm_select_radio()
        
    # initializing the selected llm
    if selected_llm == "OpenAI":
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    elif selected_llm == "Ollama":
        llm = Ollama(model="llama3:instruct")
    elif selected_llm == "Groq API":
        llm = ChatGroq(model="gemma-7b-it", api_key=os.getenv("GROQ_API_KEY"))
    elif selected_llm == "Hugging Face API":
        llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", max_length=128, temperature=0.5)
    
    # initializing the chain
    chain = prompt|llm|output_parser # create a chain from prompt to llm to output parser

    # taking the input from the user
    input_text = st.text_input("Enter your question")

    # invoke the chain with user input
    if (input_text) and input_text!="":
        st.write(chain.invoke({"question":input_text}))