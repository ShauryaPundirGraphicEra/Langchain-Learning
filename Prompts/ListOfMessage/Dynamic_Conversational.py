from langchain_core.prompts import ChatPromptTemplate   #whereas in single message's dynamic prompt we used the PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

template=ChatPromptTemplate([
    AIMessage(content="U are a well learned and calm great {domain} teacher that explain concepts from scratch"),
    HumanMessage(content="Explain in simple terms the {concept} topic,")
])


prompt=chat_template.invoke({'domaon':'cricket','topic':'Dusra'})

print(prompt)