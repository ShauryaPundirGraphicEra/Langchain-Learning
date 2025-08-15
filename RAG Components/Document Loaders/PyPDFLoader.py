from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface

loader=PyPDFLoader('./RAG Components/Document Loaders/ShauryaPundir Resume.pdf')
data=loader.load()

print(data)