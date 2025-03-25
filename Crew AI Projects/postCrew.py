import os
import time
import yaml
from crewai import Agent, Crew, Process, Task,LLM
from pydantic import BaseModel, Field
from typing import List
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\kalya\\Documents\\NeuroNudge\\"

tool =SerperDevTool()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Configure the LLM to use Cerebras
llm = LLM(
    provider="google",
    model="gemini-1.5-flash",
    api_key=GOOGLE_API_KEY
)


  
# Define file paths for YAML configurations
files = {
    'agents': 'C:/Users/kalya/Documents/NeuroNudge/AgentsPlayground/Crew AI Projects/config/agents.yaml',
    'tasks': 'C:/Users/kalya/Documents/NeuroNudge/AgentsPlayground/Crew AI Projects/config/tasks.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    try:
        with open(file_path, 'r') as file:
            configs[config_type] = yaml.safe_load(file)
            print(f"{config_type} loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        exit(1)

# Assign loaded configurations to specific variables
agents_config = configs['agents']
tasks_config = configs['tasks']
   
#Create Crew, Agents,Tasks
 
print(agents_config['search_agent'])


 
search_agent = Agent(
    role=agents_config['search_agent']['role'],
    goal=agents_config['search_agent']['goal'],
    backstory=agents_config['search_agent']['backstory'],
    tools=[tool],
    verbose=True,
    llm=llm,
    cache=True,
    respect_context_window=True,
    max_iter=20,
    max_rpm=1
)
 


content_creation_agent = Agent(
    role=agents_config['content_creation_agent']['role'],
    goal=agents_config['content_creation_agent']['goal'],
    backstory=agents_config['content_creation_agent']['backstory'],
    verbose=True,
    llm=llm,
    cache=True,
    respect_context_window=True,
    max_iter=20,
    max_rpm=1
)

 

quality_assurance_agent = Agent(
    role=agents_config['quality_assurance_agent']['role'],
    goal=agents_config['quality_assurance_agent']['goal'],
    backstory=agents_config['quality_assurance_agent']['backstory'],
    verbose=True,
    llm=llm,
    respect_context_window=True,
    max_iter=20,
    max_rpm=1
)
 


search_task = Task(
    description=tasks_config['search_task']['description'],
    expected_output=tasks_config['search_task']['expected_output'],
    agent=search_agent,
    tools=[tool]  
)

create_content_task = Task(
    description=tasks_config['create_content']['description'],
    expected_output=tasks_config['create_content']['expected_output'],
    agent=content_creation_agent,
    context=[search_task]
)

quality_assurance_task = Task(
    description=tasks_config['quality_assurance']['description'],
    expected_output=tasks_config['quality_assurance']['expected_output'],
    agent=quality_assurance_agent
)



def main():
    # Forming the crew and kicking off the process
    crew = Crew(
        agents=[search_agent,content_creation_agent,quality_assurance_agent],
        tasks=[search_task,create_content_task,quality_assurance_task],
        process=Process.sequential,
        verbose=True
    )
    result = crew.kickoff(inputs={'topic': 'Difference between Langgraph and Crew AI framework for real world business implementation'})
    print(result)

if __name__ == "__main__":
    main()



