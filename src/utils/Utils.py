import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import Ollama, HuggingFaceEndpoint
from langchain_groq import ChatGroq
import os


# >----------------- Multi LLM QnA -----------------< #
def render_llm_select_radio():
    """
    This function renders a radio button to select the LLM method to use.
    Input: None
    Output: selected_llm (str) - The selected LLM method
    """

    # center the radio buttons
    st.markdown(
        """
        <style>
            .stRadio [role="radiogroup"] {
                display: flex;
                justify-content: center;
                align-items: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # defining radio buttons for selecting the LLM method
    selected_llm = st.radio(
        label="Select the method to use LLM", 
        options=["OpenAI","Ollama","Groq","HuggingFace"], 
        captions=['gpt-3.5-turbo','llama3-8b','gemma-7b-it','Mistral-7B-Instruct-v0.2'], 
        horizontal=True
    )
    return selected_llm

def initialize_qa_llm(selected_llm):
    """
    This function initializes the selected LLM method.
    Input: selected_llm (str) - The selected LLM method
    Output: llm (object) - The initialized LLM method
    """

    if selected_llm == "OpenAI":
        llm = ChatOpenAI(model="gpt-3.5-turbo")
    elif selected_llm == "Ollama":
        llm = ChatOllama(model="llama3:instruct")
    elif selected_llm == "Groq":
        llm = ChatGroq(model="gemma-7b-it", api_key=os.getenv("GROQ_API_KEY"))
    elif selected_llm == "HuggingFace":
        base_llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", max_length=128, temperature=0.5)
        llm = ChatHuggingFace(llm=base_llm)
    return llm

# >----------------- Chat With Website -----------------< #
def render_website_select_radio():
    """
    This function renders a radio button to select the website to chat with.
    Input: None
    Output: selected_website (str) - The selected website
    """

    # center the radio buttons
    st.markdown(
        """
        <style>
            .stRadio [role="radiogroup"] {
                display: flex;
                justify-content: center;
                align-items: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # defining radio buttons for selecting the website
    selected_website = st.radio(
        label="Select the website to chat with", 
        options=["Wikipedia","Langsmith","Arxiv"], 
        horizontal=True
    )
    return selected_website