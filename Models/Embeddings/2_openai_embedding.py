from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

#
docs=[
    "This is shaurya",
    "I m currently in B.Tech CSE",
    "I want to serve my country through joining ARMY."
]

result = embedding.embed_query("Delhi is the capital of India") # embed_query is used to generate embeddings for a single query or text input.
result1=embedding.embed_documents(docs)


print(str(result))
print(str(result1))