import os
from google.cloud import aiplatform

#EnvironmentVariables
os.environ["SERPER_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
os.environ["USER_AGENT"] = "myagent"
os.environ["PINECONE_API_KEY"]=""
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\kalya\Documents\NeuroNudge\eminent-tape-444016-j4-da31b58631e7.json"

project_id = '671081661958'
aiplatform.init(project=project_id, location="us-central1")

#embedding model to store documents in vector database

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_google_vertexai import VertexAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langchain_community.tools.google_serper.tool import GoogleSerperRun
from typing_extensions import List, TypedDict
from langchain_core.output_parsers import JsonOutputParser
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from bs4 import BeautifulSoup



# Initialize Pinecone

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
my_index = "zzzbbokuhblljfvghbsgfd"
if my_index not in pc.list_indexes().names():
    pc.create_index(
        name=my_index,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
else:
   print(f"Index '{my_index}' already exists.")
   
# Connect to the index
index = pc.Index(my_index)
    
# List of URLs to load documents from
urls=["https://cookdtv.com/categories/cuisines/Indian",
"https://cookdtv.com/categories/cuisines/Italian",
"https://cookdtv.com/categories/cuisines/Middle%20Eastern"
]

# Load documents from the URLs
docs=[WebBaseLoader(url).load() for url in urls]
    
docs_list=[item for sublist in docs for item in sublist]

# Initialize a text splitter
text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0)

#chunks
doc_splits = text_splitter.split_documents(docs_list)

# Add the chunks to the Pinecone using Vertex AI embedding model
vectorstore=PineconeVectorStore(    
    index = index,
    embedding = VertexAIEmbeddings(model_name="textembedding-gecko@003")
)

documents = [Document(page_content=chunk.page_content) for chunk in doc_splits]

# Add documents to the vector store
vectorstore.add_documents(documents)

#Retriever
retriever = vectorstore.as_retriever(   
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)

#Web Search tool
serper = GoogleSerperAPIWrapper()
web_search_tool = GoogleSerperRun(api_wrapper=serper)

#Prompt and LLM
chefprompt = PromptTemplate(
    template="""
    You are an AI chef who creates recipes based on given ingredients.

    {% if documents %}
    Based on the following sources:
    {% for doc in documents %}
    - {{ doc.metadata.title }}: {{ doc.metadata.url }}
    {% endfor %}

    Using the ingredients provided, find the recipe. Ensure the recipe includes a title, ingredients list, and step-by-step instructions. If an ingredient is missing, suggest a substitute. Keep the response concise.

    Ingredients: {{ ingredients }}
    {% else %}
    No relevant documents found. Please generate a recipe based solely on the provided ingredients.

    Ingredients: {{ ingredients }}
    {% endif %}
    Recipe:
    """,
    input_variables=["ingredients", "documents"],
    template_format="jinja2"
)


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

rag_chain = chefprompt | llm | StrOutputParser()

#Decision Box EITHER WEB or DOCUMENT
#prompt = PromptTemplate(
#    template="""You are a grader assessing relevance of a retrieved document to a user ingredients. \n 
#    Here is the retrieved document: \n\n {document} \n\n
#    Here is the user ingredients: {ingredients} \n
#    If the document contains keywords related to the user ingredients, grade it as relevant. \n
#    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
#    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the ingredients. \n
#    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
#    input_variables=["ingredients", "document"],
#)
#
#retrieval_grader = prompt | llm | JsonOutputParser()


#class `GraphState` that represents the state of the graph with attributes
class GraphState(TypedDict):
    ingredients: str
    recipe: str
    search: str
    documents: List[str]
    
#core functions for the document retrieval and recipe generation pipeline.
def retrieve(state):
    ingredients = ", ".join(state["ingredients"])
    documents = retriever.invoke(ingredients)
    if not documents:
        # No relevant documents found; perform web search
        search = "Yes"
    else:
        search = "No"
    return {"documents": documents, "ingredients": ingredients,"search": search}
    
def generate(state):
    ingredients = state["ingredients"]
    documents = state["documents"]
    recipe = rag_chain.invoke({"documents": documents, "ingredients": ingredients})
    if documents:
        sources = "\n\nSources:\n"
        for doc in documents:
            title = doc.metadata.get("title", "Untitled")
            url = doc.metadata.get("url", "No URL")
            sources += f"- {title}: {url}\n"
    else:
        sources = "\n\nNo sources available."
    
    # Append sources to the recipe
    recipe_with_sources = recipe + sources
    
    return {
        "documents": documents,
        "ingredients": ingredients,
        "recipe": recipe_with_sources,
    }
    
#def grade_documents(state):
#    ingredients = state["ingredients"]
#    documents = state["documents"]
#    filtered_docs = []
#    search = "No"
#    for d in documents:
#        score = retrieval_grader.invoke(
#            {"ingredients": ingredients, "document": d.page_content}
#       )
#        grade = score["score"]
#        if grade == "yes":
#            filtered_docs.append(d)
#        else:
#            search = "Yes"
#            continue
#    return {
#        "documents": filtered_docs,
#        "ingredients": ingredients,
#        "search": search,
#    }
    

 
def web_search(state):
    ingredients = state["ingredients"]
    documents = state.get("documents", [])
    web_results = web_search_tool.run({"query": ingredients})
    
    try:
        if isinstance(web_results, str):
            # If web_results is a JSON string
            import json
            web_results = json.loads(web_results)
        
        if isinstance(web_results, list):
            documents.extend(
                [
                    Document(page_content=d.get("content", ""), metadata={"url": d.get("url", "")})
                    for d in web_results if isinstance(d, dict)
                ]
            )
        else:
            print("Unexpected structure of web_results")
    except (TypeError, KeyError, json.JSONDecodeError) as e:
        print(f"Error processing web_results: {e}")
    
    return {"documents": documents, "ingredients": ingredients} 
 
    
def decide_to_generate(state):
    search = state["search"]
    if search == "Yes":
        return "search"
    else:
        return "generate"
        
#Graph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  
#workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate) 
workflow.add_node("web_search", web_search) 


# Build graph
workflow.set_entry_point("retrieve")
#workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "retrieve",
    decide_to_generate,
    {"search": "web_search", "generate": "generate"},
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

custom_graph = workflow.compile()

# FastAPI Setup
app = FastAPI()

class Query(BaseModel):
    ingredients: List[str]
    
@app.get("/health")
def health_check():
    return {"status": "I am Healthy!!!"}
    
@app.post("/recipe")
def generate_recipe(query: Query): 
    formatted_ingredients = ", ".join(query.ingredients)
    answer=custom_graph.invoke({"ingredients": formatted_ingredients})
    return {"answer": answer}    
    
    
# Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


    










