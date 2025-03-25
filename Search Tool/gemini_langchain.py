from langchain_google_genai  import ChatGoogleGenerativeAI
import os


# Set your Google API key

os.environ["GOOGLE_API_KEY"] = ""



# Create a Gemini model instance

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)



# Send a prompt and get the response

response = gemini_llm.invoke("What is the capital of France?")

print(response.content) 
