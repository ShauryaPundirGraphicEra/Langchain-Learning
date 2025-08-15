from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
# for those application wher ethe json output is required

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

parser=JsonOutputParser()  # creating object 


model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template="Give me name,age,city of a fictional person\n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
#partial variable gets filled up before the runtime without any need fo user to specify it 


# prompt=template.format()
# result=model.invoke(prompt)
# final_result=parser.parse(result.content)

chain=template | model |parser 
result=chain.invoke({})
print(result)


# print(final_result)


# flaw of json output parser is that it does not enforce a schema 
# example if we ask it to generate 5 fact about a topic ,then it will result into 
# "5 fact":["",""]
#but we want like this :- "5 facts"{
    # "fact 1": ""
    # "fact 2": ""
    # }
    
 # then we have to use the structured output for this   

