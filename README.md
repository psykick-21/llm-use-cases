# LLM apps

I have created multiple utility applications on **Streamlit** using LLMs with the help of frameworks like Langchain, HuggingFace, etc. A brief of each of the service along with the components used to build it are described below.

## Question-Answering
This application provides an interface for the user to ask a question and get back a response. An option is provided to choose the method of selecting the preferred LLM. The current options are:
1. **OpenAI**: *gpt-3.5-turbo* model by calling OpenAI API
2. **Ollama**: *llama3-8b-instruct* model running on the local system through Ollama
3. **Groq**: *gemma-7b-it model* by calling the Groq API
4. **HuggingFace**: *Mistral-7B-Instruct-v0.2* model by calling HuggingFace API.

Prompt templates, output parsers and chains from Langchain are used to create an end-to-end question answering pipeline.