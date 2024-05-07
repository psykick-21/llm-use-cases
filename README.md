# LLM apps

I have created multiple utility applications on **Streamlit** using **LLMs** with the help of **Langchain** framework. A brief of each of the service along with the components used to build it are described below.<br>
**Langsmith** has been used to track all the interactions with the app.

## Question-Answering
This application provides an interface for the user to ask a question and get back a response. An option is provided to choose the method of selecting the preferred LLM. The current options are:
1. **OpenAI**: *gpt-3.5-turbo* model by calling OpenAI API
2. **Ollama**: *llama3-8b-instruct* model running on the local system through Ollama
3. **Groq**: *gemma-7b-it model* by calling the Groq API
4. **HuggingFace**: *Mistral-7B-Instruct-v0.2* model by calling HuggingFace API.

Prompt templates, output parsers and chains from Langchain are used to create an end-to-end question-answering pipeline.

## Website Search
This application provides an interface for the user to essentially ask a question to a website and get a response. An option is provided to select which website to chat with. Currently, these websites are supported.
1. **Wikipedia**
2. **Langsmith**
3. **Arxiv**

For these websites, by default, the model is selected as **OpenAI (gpt-3.5-turbo)**.<br>
To develop a specialization for the model for a particular website, specialized Langchain **Tools** and **Agents** have been created.

When given a query, the agent uses the provided tool to look up for the information related to the query and generates the context without any human intervention. Based on the generated context, the agent uses the power of LLM and returns a response.<br>
This is the power of AI Agents. They do not require any human intervention, and can perform all the steps on their own if defined properly, and if the problem is within the scope of the tools.

---

This project is still in development. More functionalities will be added in the future.
