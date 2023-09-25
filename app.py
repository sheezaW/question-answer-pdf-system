# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CYWtGsw4VDthXIz0LVWNjto6fqDXnc9e
"""
!pip install pypdf chromadb langchain openai tiktoken tiktoken

from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS

# List of file paths to your documents
document_paths = ["/content/30(1-2)+9-17.txt", "/content/Kronenfeld-Structuralism-1979.txt"]

# Initialize an empty list to store document objects
documents = []

# Load and split each document
for document_path in document_paths:
    document = TextLoader(document_path).load_and_split()
    documents.append(document)

# Create retrievers for each document
retrievers = []

for document in documents:
    retriever = FAISS.from_documents(document, OpenAIEmbeddings()).as_retriever()
    retrievers.append(retriever)

# Define the retriever information
retriever_infos = []

for i, retriever in enumerate(retrievers):
    retriever_info = {
        "name": f"document_{i}",
        "description": f"Good for answering questions about document {i}",
        "retriever": retriever
    }
    retriever_infos.append(retriever_info)

# Create the MultiRetrievalQAChain
chain = MultiRetrievalQAChain.from_retrievers(OpenAI(), retriever_infos)

# Run the question-answering system
question = "What is structuralism?"
answer = chain.run(question)
print(answer)
