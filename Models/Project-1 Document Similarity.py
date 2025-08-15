# cosine similarity for the input query with the available 5-6 documnetns

from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

documents=[
    "The Batchelors of Technology is the degree in undergraduaget.In B.TEch we can do in differnet domains like computer science engineering,mechanical engineering,electronics engineering,e;electrical engineering,its an undergraduate course.",
    "US President Donald Trump — “I Stopped the War between India and Pakistan. I love Pakistan. PM Modi is a fantastic man —  I spoke to Modi, and now we can make a Trade deal”.",
    "Air India Crash: Air India says one engine on the crashed plane was new. 318 remains recovered in Gujarat",
    "The Indian Army is the land-based branch and the largest component of the Indian Armed Forces. The President of India is the Supreme Commander of the Indian Army, and it is commanded by the Chief of Army Staff (COAS), who is a four-star general.",     
]

query="Who stopped war between India and Pakistan?"

# Initialize the HuggingFaceEmbeddings with a specific model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
)

print("\n \n\n\n")
# Generate embeddings for the documents
document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)

