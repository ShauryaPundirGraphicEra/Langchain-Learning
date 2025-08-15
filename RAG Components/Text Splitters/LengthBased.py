# splitting text based on the length of characters
# we define a chunk size here
# but it does look for 


# from langchain.text_splitter import CharacterTextSplitter
# from langchain.community.document_loaders import PyPDFLoader

# text="Generative AI tools have become more common since the AI boom in the 2020s. This boom was made possible by improvements in transformer-based deep neural networks, particularly large language models (LLMs). Major tools include chatbots such as ChatGPT, Copilot, Gemini, Claude, Grok, and DeepSeek; text-to-image models such as Stable Diffusion, Midjourney, and DALL-E; and text-to-video models such as Veo and Sora.[9][10][11][12][13] Technology companies developing generative AI include OpenAI, xAI, Anthropic, Meta AI, Microsoft, Google, DeepSeek, and Baidu.[7][14][15]Generative AI is used across many industries, including software development,[16] healthcare,[17] finance,[18] entertainment,[19] customer service,[20] sales and marketing,[21] art, writing,[22] fashion,[23] and product design.[24] The production of Generative AI systems requires large scale data centers using specialized chips which require high levels of energy for processing and water for cooling.[25]nerative AI has raised many ethical questions and governance challenges as it can be used for cybercrime, or to deceive or manipulate people through fake news or deepfakes.[26][27] Even if used ethically, it may lead to mass replacement of human jobs.[28] The tools themselves have been criticized as violating intellectual property laws, since they are trained on copyrighted works.[29] The material and energy intensity of the AI systems has raised concerns about the environmental impact of AI, especially in light of the challenges created by the energy transition."

# splitter=CharacterTextSplitter(
#     chunk_size=50,
#     chunk_overlap=0,
#     separator=''
# )

# result =splitter.split_text(text)

# print(result)



from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./RAG Components/Text Splitters/sample.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

# split document  object
result = splitter.split_documents(docs)

print(result[3].page_content)