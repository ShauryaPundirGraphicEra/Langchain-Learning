# feedback -> Analyze -> positive -> give positive response
#                     -> negative -> negative response

# feedback -> model ->if  positve -> model ->response accordingly
#                   ->if  negative -> model -> response accordinggly 




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

model1 = ChatHuggingFace(llm=llm1)

parser=StrOutputParser()




class feedback(BaseModel):
    sentiment:Literal["positive","negative"] = Field(description="Sentiment of the feedback")

parser2=PydanticOutputParser(pydantic_object=feedback)



prompt1=PromptTemplate(
    template='Analyze the text and then classify the sentiment of the following feedback text into positive pr negative \n {feedback} {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

parser=StrOutputParser()


classifier_chain = prompt1 | model1 | parser2

# result=classifier_chain.invoke({'feedback':'This is a terrible smartphone. I want my money back!'})
# print(result)

# now the answer ^ could be anything including sentiment positive or negative and 
# many other lines or characters ; so to enforce the model to give output only 
# Positive or Negative  we will use Pydantic for structured output


prompt2=PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)




branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 | model1 | parser),
    (lambda x:x.sentiment=='negative',prompt3 | model1 | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain=classifier_chain | branch_chain


result= chain.invoke({'feedback':'This is a terrible smartphone. I want my money back!'})

print(result)

chain.get_graph().print_ascii()

