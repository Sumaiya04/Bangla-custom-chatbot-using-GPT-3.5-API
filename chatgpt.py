import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader,  Docx2txtLoader, CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  txt_index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  #loader = DirectoryLoader("data/")

  
  # Define directories for each file type
  text_dir = "data/text_files/"
  doc_dir = "data/doc_files/"  # Assuming DOCs are in a separate folder
  csv_dir = "data/csv_files/"
  pdf_dir = "data/pdf_files/"

  # Create individual loaders for each type
  text_loader_kwargs={"encoding": "utf-8"}
  text_loader = DirectoryLoader(text_dir, glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
  csv_loader = DirectoryLoader(csv_dir, glob="./*.csv", loader_cls=CSVLoader)
  doc_loader = DirectoryLoader(doc_dir, glob="./*.docx", loader_cls=Docx2txtLoader)
  pdf_loader = DirectoryLoader(pdf_dir, glob="./*.pdf", loader_cls=PyPDFLoader)

  # Combine loaders into a list
  #loaders = [text_loader, pdf_loader, doc_loader, csv_loader]
  #loaders = [text_loader, csv_loader]


  if PERSIST:
    txt_index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([text_loader])
    #csv_index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([csv_loader])
    #doc_index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([doc_loader, pdf_loader, text_loader, csv_loader])
    #pdf_index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([pdf_loader])
    #doc_index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([doc_loader])
  else:
    txt_index = VectorstoreIndexCreator().from_loaders([text_loader])
    #csv_index = VectorstoreIndexCreator().from_loaders([csv_loader])
    #doc_index = VectorstoreIndexCreator().from_loaders([doc_loader, pdf_loader, text_loader, csv_loader])
    #pdf_index = VectorstoreIndexCreator().from_loaders([pdf_loader])
    #doc_index = VectorstoreIndexCreator().from_loaders([doc_loader])
    
  # Combine all indexes into a single list
  #all_indexes = [txt_index, csv_index, doc_index, pdf_index]

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=txt_index.vectorstore.as_retriever(search_kwargs={"k": 1}),
  #retriever=MultiVectorRetriever(all_indexes),
)

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print("Answer:")
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None