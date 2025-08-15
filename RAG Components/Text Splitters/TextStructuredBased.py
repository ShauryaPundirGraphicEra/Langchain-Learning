# recursive character text splitting technique

from langchain.text_splitter import RecursiveCharacterTextSplitter      

text="""
    The AI boom[1][2] is an ongoing period of progress in the field of artificial intelligence (AI) that started in the late 2010s before gaining international prominence in the 2020s. Examples include generative AI technologies, such as large language models and AI image generators by companies like OpenAI, as well as scientific advances, such as protein folding prediction led by Google DeepMind. This period is sometimes referred to as an AI spring, to contrast it with previous AI winters.[3][4]
History:
In 2012, a University of Toronto research team used artificial neural networks and deep learning techniques to lower the error rate below 25% for the first time during the ImageNet challenge for object recognition in computer vision. The event catalyzed the AI boom later that decade, when many alumni of the ImageNet challenge became leaders in the tech industry.[5][6] In March 2016, AlphaGo beat Lee Sedol in a five-game match, marking the first time a computer Go program had beaten a 9-dan professional without handicap. This match led to significant increase in public interest in AI.[7] The generative AI race began in earnest in 2016 or 2017 following the founding of OpenAI and earlier advances made in graphics processing units (GPUs), the amount and quality of training data, generative adversarial networks, diffusion models and transformer architectures.[8][9]

In 2018, the Artificial Intelligence Index, an initiative from Stanford University, reported a global explosion of commercial and research efforts in AI. Europe published the largest number of papers in the field that year, followed by China and North America.[10] Technologies such as AlphaFold led to more accurate predictions of protein folding and improved the process of drug development.[11] Economists and lawmakers began to discuss the potential impact of AI more frequently.[12][13]

The release of ChatGPT in November 2022, a chatbot based on a large language model created by OpenAI, accelerated the pace of AI boom.[14] ChatGPT had over 100 million users in two months, and according to investment bank UBS, was the fastest-growing consumer software application in history.[15][16] Several other companies have released competitors. At a similar time, text-to-image-models such as DALL-E and Midjourney become popular as a way to generate complicated photo-like illustrations.[17] Speech synthesis software also became able to replicate the voices and speech of specific people.[18]
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)

