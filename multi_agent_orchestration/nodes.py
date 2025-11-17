from server import get_tools
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from datetime import datetime
import os

def get_system_prompt(docs_info=None):
    system_prompt = f"""
    Today is {datetime.now().strftime("%Y-%m-%d")}
    You are a helpful AI Assistant that can use these tools:
    - generate_image: Generate an image using DALL-E based on a prompt
    - data_visualization: Create charts with Python and matplotlib
    - python_repl: Execute Python code
    
    When you call image generation or data visualization tool, only answer the fact that you generated, not base64 code or url.
    Once you generated image by a tool, then do not call it again in one answer.
    """
    if docs_info:
        docs_context = "\n\nYou have access to these documents:\n"
        for doc in docs_info:
            docs_context += f"- {doc['name']}: {doc['type']}\n"
        system_prompt += docs_context
        
    system_prompt += "\nYou should always answer in same language as user's ask."
    return system_prompt 

def create_chatbot(docs_info=None):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(get_system_prompt(docs_info)),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    # Use the LLM without binding tools
    chain = prompt | llm
    
    def chatbot(state: MessagesState):
        # Ensure messages are in the right format
        if isinstance(state["messages"], str):
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=state["messages"])]
        else:
            messages = state["messages"]
            
        response = chain.invoke(messages)
        return {"messages": messages + [response]}
    
    return chatbot