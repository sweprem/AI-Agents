from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent,AgentType
import os, asyncio
import nest_asyncio
from langchain.agents import initialize_agent,AgentType
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import List
os.environ["GOOGLE_API_KEY"] = ""


nest_asyncio.apply()

#LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Set up the Playwright browser and tools
async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
    
# Initialize the agent with tools and LLM
agent_chain = initialize_agent(
    tools=tools,
    llm=llm, 
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

class AgentState(BaseModel):
    history: List[str] = []

# Define Nodes

async def start(state: AgentState) -> dict:
    """Start node, returns the last message in history or default text"""
    last_message = state.history[-1] if state.history else "How can I assist you?"
    return {"history": state.history + [last_message]}  

async def run_agent(state: AgentState) -> dict:
    """Process message using agent and return updated history"""
    query = state.history[-1] 
    print(f"User Input: {query}")
    
    # Invoke the agent
    response = await agent_chain.ainvoke(query)
    print(f"Agent Response: {response}")

    # Update the history with the agent's response
    updated_history = state.history + [response]
    return {"history": updated_history}

# Build LangGraph (State Graph)
graph = StateGraph(AgentState)
graph.add_node("start", start)
graph.add_node("run_agent", run_agent)
graph.add_edge("start", "run_agent")
graph.add_edge("run_agent", END)
graph.set_entry_point("start")

# Compile the Agent
agent_executor = graph.compile()

# Run the agent within an event loop
async def run():
    message="go to https://www..com/ and get career"
    answer = await agent_executor.ainvoke({"history":[message]})
    latest_response = answer["history"][-1] if "history" in answer and answer["history"] else "Error: No response generated."
    return {"answer": latest_response}

if __name__ == "__main__":
    # Run the agent using asyncio
    asyncio.run(run())






