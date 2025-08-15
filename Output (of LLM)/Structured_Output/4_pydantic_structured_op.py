from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
import os
from dotenv import load_dotenv
from typing import Literal, TypedDict,Annotated,Optional
from pydantic import BaseModel,Field

# Load environment variables (for API key)
load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)





class Review(BaseModel):
    key_points:list[str]=Field(...,description="Write down all the key themes discussed in the review")
    summary:str=Field(...,description="A short summary of the product review")
    sentiment:Literal["pos","neg"]=Field(...,description="The overall sentiment of review either positive ,negative or neutral")   
    pros:Optional[list[str]]=Field(default=None,description="A list of all positive aspect of product")
    cons:Optional[list[str]]=Field(default=None,description="A list of all negative aspect of product")

    
    


structured_model=model.with_structured_output(Review)



response = structured_model.invoke("""The hardware is great but software feels bloated.There are too many preinstalled apps in it that i cant remove .Also UI looks outdated compared to other brands .Hoping for a software update to fix this. """)

print(response.summary)