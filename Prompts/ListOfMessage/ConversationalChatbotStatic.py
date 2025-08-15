from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage # for the conversation history saving and context
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM with Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    # temperature=temperature,
    # max_new_tokens=max_tokens
)

model = ChatHuggingFace(llm=llm)

# now to create a chat history for keeping the AI chatbot in context,
#chat_history=[]  # commented for using the Message of langchain
# but using the above list we cant ,know which statement of chat_history is by whom AI or User
#so for dealing with that we use the " Message " class of Langchain

chat_history=[
    SystemMessage(content="You are a helpful and polite assistant agent.")
]


# result=model.invoke(messages)

# messages.append(AIMessage(content=result.content))

# print(messages)



#static prompt
while True:
    user_input=input("👨You: ")
    chat_history.append(HumanMessage(content=user_input))  # chat_history.append(user_input)
    if user_input=='exit':
        break
    result=model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content)) # chat_history.append(result.content)
    print(f"🤖AI: ${result.content}")
    