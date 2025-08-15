# web based loader is a document loader in langchain used to 
# load and extract text from web pages. It uses BeautifulSoup under the hood to parse html and extract visibletest
from langchain_community.document_loaders import WebBaseLoader


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
 


load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm1)

parser=StrOutputParser()


url="https://www.amazon.in/2022-Apple-MacBook-Laptop-chip/dp/B0DLHWMPQL/ref=sr_1_16?adgrpid=133194557750&dib=eyJ2IjoiMSJ9.L2Hu7nsAw269fTBuFKBbdksdJpKjcs3xXIn3CIaav0rfbKGb21QeChMY3a5pO1DvFYRLO-4iqRUeuHrGMMwC7VtoX_EV62rTz_0slBf0eFpS-umispY0af0H13A6yCWnzu7cspBfJdPW-Ktw-f-hVjzh-SdXheUXTfg7UqNOT0KbMe74qQkHfYzLoxtYfI5Y-6Fdj3xXX5pjBFxLntqgcejvT5M-hDKykqjdE3nOJPs.vxazqWXizl_JfwJ94T5TgRP_mOYOA1HGgeyeGAHAq4k&dib_tag=se&hvadid=558702215645&hvdev=c&hvlocphy=1007819&hvnetw=g&hvqmt=b&hvrand=15564430522228148610&hvtargid=kwd-2207052127139&hydadcr=25623_1777031&keywords=macbook+air+13+inch+m2+512gb&mcid=3d5e33e124593967aae6c8bb193300a6&qid=1755021032&sr=8-16"

prompt=PromptTemplate(
    template='Answer the following question - \n{query} ,using the following text - \n{text}',
    input_variables=['query','text']
)



loader=WebBaseLoader(url)
data=loader.load()

# print(data[0].page_content)

chain=prompt | model |parser

print(chain.invoke({'query':'Is the laptop good choice for data science and machine learning related tasks?','text':data[0].page_content}))