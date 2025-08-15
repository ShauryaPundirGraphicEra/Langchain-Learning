# it helps us to enforce response schema 
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name='fact_1',description="Fact 1 about topic"),
     ResponseSchema(name='fact_2',description="Fact 1 about topic"),
      ResponseSchema(name='fact_3',description="Fact 1 about topic"),
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template="Give 3 facts about {topic}\n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt=template.invoke({'topic':'black hole'})
# result= model.invoke(prompt)
# final_result=parser.parse(result.content)


# chain method which is simpler
result= template | model | parser

final_result=result.invoke({'topic':'Artificial Intelligence'})

print(final_result)


# disadvantage of it is :- cannot perform data validation
# (str) name
# (int )age   // so for it we use the pydantic output parser