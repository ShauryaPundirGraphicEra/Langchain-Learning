from langchain_community.llms import HuggingFaceHub
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate

prompt = input("Enter your prompt: ")

formatted_prompt = f"<s>[INST] {prompt} [/INST]"

llm = HuggingFaceHub(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
)

result = llm.invoke(formatted_prompt)
print(result)
