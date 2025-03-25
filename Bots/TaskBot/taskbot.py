import os
import re
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from typing import List,TypedDict
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_core.output_parsers import StrOutputParser
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
    connection_string = "mysql+pymysql://root:3306/task"
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

progress_prompt = PromptTemplate(
    input_variables=["completion_percentage","current_task"],
    template="""You are an empathetic AI assistant dedicated to supporting users with ADHD. Your goal is to provide personalized motivational messages based on the user's task completion percentage.
    Based on a completion percentage of {completion_percentage}%, provide the appropriate motivational message tailored to the {current_task}"""
)

progress_chain = progress_prompt | llm | StrOutputParser()

appreciation_prompt = PromptTemplate(
    input_variables=["current_task","Tasks"],
    template="""You are an AI assistant helping a user with ADHD by guiding them through their tasks. You will provide a list of tasks and ask them to choose from it

    Task List:
    {Tasks}"""
)
appreciation_chain = appreciation_prompt | llm | StrOutputParser()

#class `GraphState` that represents the state of the graph with attributes
class GraphState(TypedDict):
    query: str
    Tasks: List[str]
    current_task :str
    response:str
    motivation_message:str
    completion_status:str
    completion_percentage:int
    appreciation_message:str
    
def display_tasks_and_get_selection(tasks):
    print("Available tasks:")
    for i, task in enumerate(tasks, start=1):
        print(f"{i}. {task}")

    # Prompt the user to select a task number
    while True:
        try:
            selection = int(input("Please choose the number of the task you want to work on: "))
            if 1 <= selection <= len(tasks):
                return tasks[selection - 1]
            else:
                print(f"Please enter a number between 1 and {len(tasks)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    
#core functions
def task_retrieval(state):
    query = state["query"]
    tasks = agent.invoke(query)
    
    # Process the result assuming tasks can be either a list or a string
    if not tasks:
        state["response"] = "No tasks available."
        state["Tasks"] = []
    else:
        if isinstance(tasks, dict) and "output" in tasks: #agent retuns a dict,input and output
            tasks_text = tasks["output"]
            tasks_text = re.sub(r"^[^:]+are\s+", "", tasks_text, flags=re.IGNORECASE)
            task_list = re.split(r",\s*|\s+and\s+", tasks_text)
            task_list = [t.strip() for t in task_list if t.strip()]
        elif isinstance(tasks, str):
            task_list = [task.strip() for task in tasks.split(",") if task.strip()]
        else:
            task_list = tasks
            
        state["Tasks"] = task_list
        
        current_task = display_tasks_and_get_selection(task_list)
        state["current_task"] = current_task    #update state
    return {"Tasks": tasks,"current_task":current_task}

def progress_track(state):
    query = state["query"]  #whatever state was updated , assign back to func variable
    Tasks = state["Tasks"]
    current_task = state["current_task"]
    completion_percentage = None
    # Prompt the user for their completion percentage
    while completion_percentage is None:
        try:
            user_input = input(f"On a scale of 0 to 100, how much of '{current_task}' have you completed? ")
            completion_percentage = float(user_input)
            motivation_message=progress_chain.invoke({"completion_percentage": completion_percentage, "current_task": current_task})
            print(motivation_message)
            if not (0 <= completion_percentage <= 100):
                print("Please enter a number between 0 and 100.")
                completion_percentage = None
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    
    state["motivation_message"]=motivation_message
    state["completion_percentage"]=completion_percentage
    return{"motivation_message":motivation_message,"completion_percentage":completion_percentage}
    
    
def task_completion_check(state):
    completion_status = state["completion_percentage"]
    if completion_status == 100:
        return "nexttask"
    else:
        return "tracktask" 


def task_flows(state):
    Tasks = state["Tasks"]
    current_task = state["current_task"]
    appreciation_message = appreciation_chain.invoke({"current_task": current_task, "Tasks": Tasks})
    return{"appreciation_message":appreciation_message,"Tasks": Tasks}
        



  
#Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("task_retrieval", task_retrieval) 
workflow.add_node("progress_track",progress_track)
workflow.add_node("task_flows",task_flows)

workflow.set_entry_point("task_retrieval")
workflow.add_edge("task_retrieval", "progress_track")
workflow.add_conditional_edges(
    "progress_track",
    task_completion_check,
    {"tracktask": "progress_track", "nexttask": "task_flows"},
)
workflow.add_edge("task_flows", END)


# Compile the Agent
compiled_graph = workflow.compile()



# Create an initial state with the expected keys.
initial_state = {
    "query": "what is bob's task",
    "Tasks": [],
    "current_task": ""
}

# Invoke the compiled graph with the state dictionary.
response = compiled_graph.invoke(initial_state)
print("Graph response:", response)



    
    