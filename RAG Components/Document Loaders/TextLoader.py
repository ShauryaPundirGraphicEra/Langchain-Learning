# Simple and comonly used documentloader in langchain

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal



load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm1)

parser=StrOutputParser()

loader=TextLoader('./RAG Components/Document Loaders/doc1.txt',encoding='utf8')
data=loader.load()
# print(data)

    #template='Write a summary for the following text - \n{text}',
prompt=PromptTemplate(
    template='Check if the given user - {query} is correct according to the real facts using the article text - \n{text}',
    input_variables=['query','text']
)
chain=prompt | model |parser
print(chain.invoke({'text':data[0].page_content,'query':'Operation Sindoor was started by Pakistan and sot down 5 indian Rafaels'}))
