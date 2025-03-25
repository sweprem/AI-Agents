from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\kalya\\Documents\\NeuroNudge\\"

tool =SerperDevTool()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Configure the LLM to use Cerebras
gemini_llm = LLM(
    provider="google",
    model="gemini-1.5-pro",
    api_key=GOOGLE_API_KEY
)

# Agent definition
researcher = Agent(
    role='{topic} Senior Researcher',
    goal='Uncover groundbreaking technologies in {topic} for the year 2024',
    backstory='Driven by curiosity, you explore and share the latest innovations.',
    tools=[tool],
    llm=gemini_llm,
    max_iter=25,
    max_rpm=2,
    
)

# Define a research task for the Senior Researcher agent
research_task = Task(
    description='Identify the next big trend in {topic} with pros and cons.',
    expected_output='A 3-paragraph report on emerging {topic} technologies.',
    agent=researcher
)

def main():
    # Forming the crew and kicking off the process
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )
    result = crew.kickoff(inputs={'topic': 'AI Agents'})
    print(result)

if __name__ == "__main__":
    main()
