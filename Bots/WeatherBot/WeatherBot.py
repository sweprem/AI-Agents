import os
os.environ["GOOGLE_API_KEY"] = ""
os.environ["OPENWEATHERMAP_API_KEY"] = ""

from fastapi import FastAPI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent,AgentType,load_tools
from langgraph.graph import StateGraph, END
from typing import List
from langchain_google_genai  import ChatGoogleGenerativeAI


weather = OpenWeatherMapAPIWrapper()



#LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


#TOOLS
tools = load_tools(["openweathermap-api"], llm)


#Agent
agent = initialize_agent(
 tools=tools,
 llm=llm,
 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
 verbose=True,
 handle_parsing_errors=True )
 
# Define Agent State 
class AgentState(BaseModel):
    history: List[str] = []

# Define Nodes
def start(state: AgentState) -> dict:
    """Start node, returns the last message in history or default text"""
    last_message = state.history[-1] if state.history else "How can I assist you?"
    return {"history": state.history + [last_message]}  

def run_agent(state: AgentState) -> dict:
    """Process message using agent and return updated history"""
    query = state.history[-1] 
    print(f"User Input: {query}")

    # Invoke agent
    response = agent.invoke(query)
    print(response)   
    updated_history = state.history + [response]  
    return {"history": updated_history}  



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

