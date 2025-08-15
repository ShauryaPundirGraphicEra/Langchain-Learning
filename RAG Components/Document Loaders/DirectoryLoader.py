from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader=DirectoryLoader(
    path='RAG Components/Document Loaders/',
    glob='*.pdf',
    loader_cls=PyPDFLoader,
    show_progress=True
)

docs=loader.load() # returns list of document

#to fasten up loading of the document loading into memoey we use 
#lazy loading:- document are loaded at one ata time,we get a generator object to load docs into m/m

#docs=loader.lazy_load()
# for d in docs:
#     print(d.metadata)



print(docs[0].page_content)
print(docs[0].metadata)





