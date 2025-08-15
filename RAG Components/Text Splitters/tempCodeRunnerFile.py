# idea is to split text into chunks based on semantic meaning
# to compare the cosine similarity between the different lines and find the most similar ones using the embedding models
#currently in experimental phase

from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import Language

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


text = """
A short story is a piece of prose fiction.It can typically be read in a single sitting and focuses on a self-contained incident or series of linked incidents, with the intent of evoking a single effect or mood. The short story is one of the oldest types of literature and has existed in the form of legends, mythic tales, folk tales, fairy tales, tall tales, fables, and anecdotes in various ancient communities around the world. The modern short story developed in the early 19th century.[1] 
"""
splitter = SemanticChunker(embeddings=embeddings , language=Language.EN)
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])