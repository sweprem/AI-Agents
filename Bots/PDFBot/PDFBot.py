import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\kalya\\Documents\\NeuroNudge\\personal-453614-52b6bb337847.json"
os.environ["GOOGLE_API_KEY"] = ""
os.environ["PINECONE_API_KEY"]=""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_google_vertexai import VertexAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory,ChatMessageHistory
import streamlit as st

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

# Load PDF document
loader = PyPDFLoader(r"C:\Users\kalya\Documents\NeuroNudge\FujiAIService\swetha_test.pdf")
doc = loader.load()


# Initialize a text splitter
text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0)

#chunks
doc_splits = text_splitter.split_documents(doc)


documents = [Document(page_content=chunk.page_content) for chunk in doc_splits]

# Add the chunks to the Pinecone using Vertex AI embedding model
vectorstore=PineconeVectorStore(    
    index = index,
    embedding = VertexAIEmbeddings(model_name="textembedding-gecko@003")
)

# Add documents to the vector store
vectorstore.add_documents(documents)

#Retriever
retriever = vectorstore.as_retriever(   
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)


#Memory
#def get_session_history(session_id: str):
#    return(StreamlitChatMessageHistory(key=session_id))


#LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

prompt = PromptTemplate(
    input_variables=["question", "context","history"],
    template="""
You are an AI assistant answering questions based strictly on the provided document. 
Use only the retrieved document content: {context} to generate answers.

Include the following previous conversation history when answering:
{history}

If the answer is not found in the document, say: 'I could not find relevant information in the document.'

Do not make up any information. Keep your responses clear, concise, and factual.

User Query: {question}
Context: {context}
"""
)


        
#chain
chain = prompt | llm | StrOutputParser()


def chatbot(question: str, context: str, history: str):
    response = chain.invoke({
        "question": question,
        "context": context,
        "history": history  # Pass the formatted history here
    })
    return response
    


# Function to format the history for the prompt
def format_history():
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["history"]])
    return history_text


    
if "history" not in st.session_state:
    st.session_state["history"] = [
        {"role": "assistant", "content": "Hello there! How can I help you today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state["history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input field
user_prompt = st.chat_input("Type your question here...")

if user_prompt:
    
    context = retriever.invoke(user_prompt)  #1st generate context with user prompt

    st.session_state["history"].append({"role": "user", "content": user_prompt}) #2nd add it to the history
    
    with st.chat_message("user"):   #3rd write it to the chatbox
        st.write(user_prompt)
        
    history = format_history()  #4th generate complete history
    
    response = chatbot(user_prompt, context, history) #5th after having history and context invoke the chatbot
        
    # Store assistant response in session state
    st.session_state["history"].append({"role": "assistant", "content": response})   #6th again append the history and write it
    
    with st.chat_message("assistant"):
        st.write(response)
        
        
        
#chain_with_history = RunnableWithMessageHistory(
#    chain,
#    get_session_history,
#    input_messages_key="question",
#    history_messages_key="context",
#)
#
#print(chain_with_history.invoke(
#    {"question": "What is swetha's age?","context":context},
#    config={"configurable": {"session_id": session_id}}
#))



#def chatbot(question: str, session_id: str):
#    for response in runnable_with_history.stream(
#        {"question": question, "context": retriever},  # Ensure context is passed
#        config={"configurable": {"session_id": session_id}}
#    ):
#        yield response