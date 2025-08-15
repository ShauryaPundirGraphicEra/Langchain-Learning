# Embedding in models converts input text into numerical vectors that capture semantic meaning, enabling efficient similarity search and clustering.
# This is essential for tasks like information retrieval, recommendation systems, and natural language understanding.

from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize the HuggingFaceEmbeddings with a specific model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Example model, can be changed
)


docs=[
    "This is shaurya",
    "I m currently in B.Tech CSE",
    "I want to serve my country through joining ARMY."
]

# Example text to embed
text = "This is an example sentence for embedding."

# Generate embeddings for the text
embedding = embeddings.embed_query(text)  
embedding2=embeddings.embed_documents(docs)  

# Print the resulting embedding vector
print("Embedding vector 1:", embedding)

print("Embedding docs vector 2:",embedding2)