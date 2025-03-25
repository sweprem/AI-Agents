import streamlit as st



# Assign a custom session ID (e.g., use the user's name or a string)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "Swetha"  

# Display the session ID
st.write(f"Session ID: {st.session_state['session_id']}")

# Access the session ID
session_id_str = str(st.session_state["session_id"])


# Initialize chatbot messages in session state
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
    # Add user message to session state
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    
    with st.chat_message("user"):
        st.write(user_prompt)
    
    response = chatbot_history(user_prompt, session_id_str)
        
    # Store assistant response in session state
    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
        
        
        
        
        
