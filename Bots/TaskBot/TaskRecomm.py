import os
os.environ["GOOGLE_API_KEY"] = ""


from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, Tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

prompt = PromptTemplate(
    template="""
    You are an AI assistant with expertise in productivity and task management.
    Based on the provided user profile and retrieved document context, answer the following query concisely and accurately.
    
    User Profile Context:
    {retrieved_docs}
    
    Query:
    {query}
    
    The task order should ensure an optimal balance between preference, priority, and urgency without compromising on critical deadlines.
    """
    
)


# Load PDF document
loader = PyPDFLoader(r"C:\Users\kalya\Documents\NeuroNudge\FujiAIService\swetha_test.pdf")
document = loader.load()


# Initialize a text splitter
text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0)

#chunks
chunks = text_splitter.split_documents(document)


#Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the FAISS vector store with the embeddings
vector_store = FAISS.from_documents(chunks, embedding_model)
print(f"Number of documents stored in FAISS: {len(vector_store.docstore._dict)}")

# Define the retriever using FAISS
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top-3 results
)
    

def retrieve_relevant_docs(query: str) -> str:
    """Retrieve relevant documents from the vector store based on the query."""
    results = retriever.invoke(query)
    
    if results:
        print(f"Retrieved {len(results)} relevant document(s).")
        retrieved_docs = [doc.page_content for doc in results]
    else:
        print("No relevant documents found.")
        retrieved_docs = ["No relevant documents found."]
    
    return "\n".join(retrieved_docs)


# Define tools for the agent
tools = [
    Tool(
        name="DocumentRetriever",
        func=retrieve_relevant_docs,
        description="Retrieves relevant information from the document for task management."
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

# Execute agent
# Assuming the tools and agent are defined as in your original code

# Define the query
query = "What tasks should swetha profile focus now?"

# Retrieve relevant documents using the DocumentRetriever tool
retrieved_docs = retrieve_relevant_docs(query)

# Format the prompt with the retrieved documents and query
formatted_prompt = prompt.format(query=query, retrieved_docs=retrieved_docs)

# Use the agent to process the formatted prompt and get the response
response = agent.invoke(formatted_prompt)

# Print the response
print(response)

