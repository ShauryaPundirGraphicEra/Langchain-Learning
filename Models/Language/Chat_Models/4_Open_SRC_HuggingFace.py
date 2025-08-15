# using opensource chat models from HuggingFace


from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint  # HuggingFaceEndpoint is used for custom endpoints(API)
from dotenv import load_dotenv

load_dotenv()

#  define the HuggingFace model and configure it, using the HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # TinyLlama/TinyLlama-1.1B-Chat-v1.0  or HuggingFaceH4/zephyr-7b-beta
    task="text-generation"
)

model= ChatHuggingFace(llm=llm)

prompt=input("Enter your prompt: ")

result= llm.invoke(prompt)

print(result.content)  # Output: New Delhi
