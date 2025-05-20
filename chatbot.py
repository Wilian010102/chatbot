import sys
import os

import configuration

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

os.environ["OPENAI_API_KEY"]    = configuration.API_KEY

loader                          = TextLoader("data.txt")
documents                       = loader.load()
text_splitter                   = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts                           = text_splitter.split_documents(documents)

embeddings                      = OpenAIEmbeddings()
docseacrh                       = Chroma.from_documents(texts, embeddings)

qa                              = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=docseacrh.as_retriever(), return_source_documents=True)

query = input('Hai Wili, ada yang bisa saya bantu? \n')
print(qa.run(query))