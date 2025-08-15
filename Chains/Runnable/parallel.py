from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='Generate a well SEO and eyeball catchy twitter post on the following topic: {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Gnerate a well SEO professional techy linkedin post on following topic: {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()


parallel_chain=RunnableParallel({
    'twitter':RunnableSequence(prompt1 , model,parser),
    'linkedin':RunnableSequence(prompt2 , model , parser)
})

result=parallel_chain.invoke({'topic':'Artificial Intelligence'})

print(result)

parallel_chain.get_graph().print_ascii()


