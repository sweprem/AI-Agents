import os
os.environ["GOOGLE_API_KEY"] = ""

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googleapiclient.errors import HttpError
import base64
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END, START
from typing import List,TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import GmailToolkit 
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)


toolkit = GmailToolkit()
#Load Google Cloud Credentials


credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file=r"C:\Users\kalya\Documents\NeuroNudge\AgentsPlayground\Bots\OAuthgmail.json",
)
#pass the creds to the gmail tookit
service = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=service)



#LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

#prompt
filter_prompt = PromptTemplate(
    input_variables=["subject", "content"],
    template="""You are an intelligent email assistant. Your task is to analyze the following email and classify its type based on its subject and content. Consider the nature of the message, any action required, and whether it demands immediate attention. The classification options are as follows:

- **'no action'**: The email does not require any follow-up, response, or  action. 
- **'take action'**: The email requires some form of action such as responding, scheduling, reading, filing, or any task that needs to be completed.

### Email Information:
- **Subject**: {subject}
- **Content**: {content}

Please provide only the classification result: either 'no action' or 'take action'. Do not include any additional commentary or explanation.
"""
)


filter_chain = filter_prompt | llm | StrOutputParser()

summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="""Summarize the following email content in 2 to 3 sentences: {content}"""
)

summary_chain = summary_prompt | llm | StrOutputParser()

  
response_prompt = PromptTemplate(
    input_variables=["sender", "subject", "content", "recipient"],
    template=""" 
    You are an intelligent email assistant. Your task is to analyze the following email and generate a formal response.

    Email Details:
    From: {sender}
    To: {recipient}
    Subject: {subject}
    Content: {content}

    Instructions:
    - Understand the content of the email.
    - Respond in a formal tone, addressing the key points raised in the email.
    - Ensure the response is polite, professional, and appropriate based on the subject and content.
    - Do not include any greeting lines (e.g., "Dear", "Hello") or signature lines in your response.
    - Ensure the response is concise and relevant to the email content.
    - If the email asks a question, provide a clear, direct, and informative answer.

    Please provide only the response content, without any additional commentary or explanation.
    """
)

response_chain = response_prompt | llm | StrOutputParser()



#class `EmailState` that represents the state of the email with attributes
class EmailState(TypedDict):
    current_email: str
    content: str
    subject: str
    action : str
    sender: str
    recipient: str
    summary: str
    
#Function to filter emails 
def filter_email(state):
    sender = state["sender"]
    recipient = state["recipient"]
    subject = state["subject"]
    content = state["content"]
    classify = filter_chain.invoke({"subject": subject, "content": content})
    if classify == "take action":
        action = "Yes"
        print("Action needed for this email.")
    else:
        action = "No"
        print("No action required.")
    return {"subject": subject, "content": content,"action": action,"sender":sender,"recipient":recipient}
    
 
def summary_email(state):
    sender = state["sender"]
    recipient = state["recipient"]
    subject = state["subject"]
    content = state["content"]
    summary=summary_chain.invoke({"content": content})
    return{"summary": summary,"sender":sender}
    
def response_email(state):
    sender = state["sender"]
    recipient = state["recipient"]
    subject = state["subject"]
    content = state["content"]
    reply_email=response_chain.invoke({"sender":sender,"recipient":recipient,"subject": subject, "content": content})
    body = reply_email 
    print(body)
    send_email(service, sender, recipient, subject, body)
    return {"sender":sender,"recipient":recipient,"subject": subject, "body": body}
    
    
def decide_to_respond(state):
    action = state["action"]
    if action == "Yes":
        return "response"
    else:
        return "summary"

# Function to send an email using Gmail API
def send_email(service, sender, recipient, subject, body):
    try:
        # Create a message
        message = MIMEMultipart()
        message['to'] =  sender
        message['from'] = recipient
        message['subject'] = subject
        
        # Attach the email body
        message.attach(MIMEText(body, 'plain'))
        
        # Encode the message as base64 and send
        raw_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
        
        # Send the email via Gmail API
        send_message = service.users().messages().send(userId="me", body=raw_message).execute()
        print(f"Message sent to {sender} with Message ID: {send_message['id']}")
        return send_message
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None
        
#Graph

workflow = StateGraph(EmailState)

# Define the nodes
workflow.add_node("filter_email", filter_email)  
workflow.add_node("summary_email", summary_email) 
workflow.add_node("response_email", response_email) 


# Build graph
workflow.set_entry_point("filter_email")
workflow.add_conditional_edges(
    "filter_email",
    decide_to_respond,
    {"response": "response_email", "summary": "summary_email"},
)
workflow.add_edge("response_email", END)

custom_graph = workflow.compile()

# Function to mark an email as read
def mark_email_as_read(service, msg_id):
    try:
        service.users().messages().modify(
            userId='me',
            id=msg_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        print(f"Email {msg_id} marked as read.")
    except Exception as error:
        print(f'An error occurred while marking email as read: {error}')

# Function to list unread emails
def list_unread_emails(service):
    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX','CATEGORY_PERSONAL'], q="is:unread").execute()
        messages = results.get('messages', [])
        print(f"Raw API response: {results}")
        print(f"Number of unread emails: {len(messages)}")
        return messages
    except Exception as error:
        print(f'An error occurred: {error}')
        return None
        


# Function to get details of a specific email
def get_email_details(service, msg_id):
    msg = service.users().messages().get(userId='me', id=msg_id).execute()
    headers = msg['payload']['headers']
    subject = next(header['value'] for header in headers if header['name'] == 'Subject')
    sender = next(header['value'] for header in headers if header['name'] == 'From')
    recipient = next(header['value'] for header in headers if header['name'] == 'To')
    content = msg['snippet']
    print(content)
    return subject, content,sender, recipient

unread_emails = list_unread_emails(service)


if unread_emails:
    for email in unread_emails:
        msg_id = email['id']
        subject, content, sender, recipient = get_email_details(service, msg_id)
        state = {
            "current_email": msg_id,
            "content": content,
            "subject": subject,
            "action": "",
            "sender": sender,  
            "recipient": recipient,  
            "summary": "",
        }
        answer=custom_graph.invoke(state)

        # Mark email as read only if LangFlow successfully processes it
        if answer:
            mark_email_as_read(service, msg_id)
