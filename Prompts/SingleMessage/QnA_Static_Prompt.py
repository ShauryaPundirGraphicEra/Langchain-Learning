from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables (for API key)
load_dotenv()


# Streamlit UI
st.header("🤖 QnA with Langchain and HuggingFace")
st.markdown("Ask questions or summarize text using Hugging Face's Mistral model (46.7B params) powered by LangChain!")


user_input = st.text_input("Ask a question:")  # static prompt for user input
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7, help="Controls response randomness.")
max_tokens = st.sidebar.slider("Max Tokens", 25, 500, 200, help="Maximum length of the response.")


# Initialize LLM with Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=temperature,
    max_new_tokens=max_tokens
)

model = ChatHuggingFace(llm=llm)



if st.button("Ask Question"): 
    if user_input:
        with st.spinner("Thinking..."):
            response = model.invoke(user_input)
            st.text(str(response.content))
    else:
        st.warning("Please enter a question before clicking Summarize.")
        
        
        
st.markdown(
    "<hr style='margin-top: 2rem;'><div style='text-align: center;'>Made with ❤️ by Shaurya Pundir, using Langchain and Open Source Models</div>",
    unsafe_allow_html=True
)
