import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate
from google.cloud import aiplatform
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent,AgentType
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
from typing import List
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
import os


os.environ["GOOGLE_API_KEY"] = ""

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

custom_prompt = PromptTemplate.from_template("""
You are an AI assistant. Always use the `Intermediate Answer` tool when answering real-time questions.

If the question does NOT require real-time data, answer directly. Otherwise, perform a web search.

Question: {query}
""")
# Define a simple search tool


os.environ["SERPER_API_KEY"] = ""

search = GoogleSerperAPIWrapper()


serper_api_tool = Tool(
        name="Intermediate Answer",
        func=search.run,
        description="Use this tool when you need real-time information."
            )


# Initialize Agent
agent = initialize_agent(
 tools=[serper_api_tool],
 llm=llm,
 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
 verbose=True,
 prompt=custom_prompt,
 handle_parsing_errors=True )

# Define Agent State (Must be a BaseModel for LangGraph)
class AgentState(BaseModel):
    history: List[str] = []

# Define Nodes
def start(state: AgentState) -> dict:
    """Start node, returns the last message in history or default text"""
    last_message = state.history[-1] if state.history else "How can I assist you?"
    return {"history": state.history + [last_message]}  # ✅ Explicitly returning a state

def run_agent(state: AgentState) -> dict:
    """Process message using agent and return updated history"""
    query = state.history[-1] 
    print(f"User Input: {query}")

    # Invoke agent
    response = agent.invoke(query)
    print(response)   
    updated_history = state.history + [response]  # ✅ Append only the extracted content
    return {"history": updated_history}  # ✅ Return updated history



# Build LangGraph
graph = StateGraph(AgentState)
graph.add_node("start", start)
graph.add_node("run_agent", run_agent)
graph.add_edge("start", "run_agent")
graph.add_edge("run_agent", END)
graph.set_entry_point("start")

# Compile the Agent
agent_executor = graph.compile()

# FastAPI Setup
app = FastAPI()

class Query(BaseModel):
    message: str
    
@app.get("/health")
def health_check():
    return {"status": "I am Healthy!!!"}


@app.post("/chat")
def chat(query: Query):
    """Handle chat requests"""
    answer = agent_executor.invoke({"history": [query.message]})  # Invoke agent with history
    latest_response = answer["history"][-1] if "history" in answer and answer["history"] else "Error: No response generated."
    return {"answer": latest_response}

# Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
