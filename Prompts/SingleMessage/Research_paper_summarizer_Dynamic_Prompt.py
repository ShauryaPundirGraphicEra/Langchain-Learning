from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
import streamlit as st

# Load environment variables (for API key)
load_dotenv()


# Streamlit UI
st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )



# template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=['paper_input', 'style_input','length_input'],
validate_template=True
)

# if we want to use the template from json file, we can use the following code
# template = PromptTemplate.from_file("path_to_your_template.json")

prompt=template.invoke({
    "paper_input":paper_input,
    "style_input":style_input,
    "length_input":length_input
    })


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
    if prompt:
        with st.spinner("Thinking..."):
            response = model.invoke(prompt)
            st.text(str(response.content))
    else:
        st.warning("Please enter a question before clicking Summarize.")
        
# u can see in the above code we are using the invoke method two times, 
# first to create the prompt and then to get the response from the model.
#but we can also use the invoke method only once, using chains like this:

# if st.button('Summarize'):
#     chain = template | model
#     result = chain.invoke({
#         'paper_input':paper_input,
#         'style_input':style_input,
#         'length_input':length_input
#     })
#     st.text(result.content)
        
        
st.markdown(
    "<hr style='margin-top: 2rem;'><div style='text-align: center;'>Made with ❤️ by Shaurya Pundir, using Langchain and Open Source Models</div>",
    unsafe_allow_html=True
)
