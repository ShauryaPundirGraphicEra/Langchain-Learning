import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="Chat with Mistral AI", layout="centered")

# Initialize Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model = ChatHuggingFace(llm=llm)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful and polite assistant agent.")
    ]

# Title
st.title("🤖 Chat with Mistral-7B (Hugging Face)")

# Chat display area
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User input box
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Display user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke model with full chat history (not just last user message)
    response = model.invoke(st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=response.content))

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(response.content)
