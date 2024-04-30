import streamlit as st

def render_llm_select_radio():
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