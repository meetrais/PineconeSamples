#Run this file One time to create and Save embeddings of your file to Pinecone Index.
#pip install pinecone-client openai tiktoken langchain
import getpass
import apikey
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
import pinecone

pinecone_api_key = apikey.PINECONE_API_KEY
pinecone_env = apikey.PINECONE_ENV
openai_api_key = apikey.OPENAI_API_KEY
pinecone_index_name = apikey.PINECONE_INDEX_NAME

loader = TextLoader("Files/Chess.txt", encoding="utf8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")

pinecone.init(
    api_key = pinecone_api_key,
    environment = pinecone_env
)

index_name = pinecone_index_name

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=1536  
    )

docsearch = Pinecone.from_documents(docs, embeddings,index_name=index_name)
query = "What is Castling?"

docs = docsearch.similarity_search(query)

print(docs[0].page_content)

print("Done !!!")



