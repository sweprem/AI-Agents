import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
from typing import List
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType,AgentExecutor




os.environ["GOOGLE_API_KEY"] = ""

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


def getengine():
    connection_string = "mysql+pymysql://root:Chikki%40123@35.192.164.209:3306/flightdata"
    engine = create_engine(connection_string, echo=True)
    return engine
    

db = SQLDatabase(getengine()) # langchain expects a engine by sqlalcamy to work.
toolkit = SQLDatabaseToolkit(db=db, llm=llm)


agent= create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)


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
    if isinstance(response, dict):
        response_text = response.get("output", "")
    else:
        response_text = str(response)
    updated_history = state.history + [response_text]  # ✅ Append only the extracted content
    return {"history": updated_history}  # ✅ Return updated history



# Build LangGraph
graph = StateGraph(AgentState)
graph.add_node("start", start)
graph.add_node("run_agent", run_agent)
graph.add_edge("start", "run_agent")
graph.add_edge("run_agent", END)
graph.set_entry_point("start")

# Compile the Agent
compiled_graph = graph.compile()

# FastAPI Setup
app = FastAPI()

class Query(BaseModel):
    message: str
    
@app.get("/health")
def health_check():
    return {"status": "I am Healthy!!!"}


@app.post("/chat")
def chat(query: Query):
    answer = compiled_graph.invoke({"history": [query.message]})
    latest_response = answer["history"][-1] if "history" in answer and answer["history"] else "Error: No response generated."
    return {"answer": latest_response}
        

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)