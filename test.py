import os
import langchain
from langchain_openai import ChatOpenAI
import pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.sambanova import SambaStudioEmbeddings
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv

load_dotenv()


def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents


doc = read_doc("documents/")


def chunk_data(docs, chunk_size=810, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc = text_splitter.split_documents(docs)
    return doc


documents = chunk_data(docs=doc)
print(documents)

embeddings = SambaStudioEmbeddings()
