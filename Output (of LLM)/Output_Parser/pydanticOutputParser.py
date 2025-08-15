# its a structured output parser in langchain that uses the pydantic models to enforce schema validation whem 
# processing the llm responses

# enforces strict schema
#onverts llm outout into python object
# use pydantic built in validation to catvh incorrect ot missing dat
# easy validation 
# works well with langchain components

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name:str =Field(description='Name of person')
    age:int=Field(description='Age of person')
    city:str=Field(description='Name of the city the person belong to')

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template='Generate the name ,age,city of fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions() }
)


prompt=template.invoke({'place':'Indian'})

result=model.invoke(prompt)
final_result=parser.parse(result.content)
print(final_result)

