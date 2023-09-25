import streamlit as st
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS

# Function to initialize the MultiRetrievalQAChain
def initialize_qa_chain(document_paths):
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

    return chain

# Streamlit app
def main():
    st.title("Question Answering with Langchain and Streamlit")
    st.sidebar.title("Settings")

    # File paths to your documents
    document_paths = st.sidebar.text_area("Document Paths (comma-separated)", "/content/30(1-2)+9-17.txt, /content/Kronenfeld-Structuralism-1979.txt").split(",")

    # Initialize the QA chain
    chain = initialize_qa_chain([path.strip() for path in document_paths])

    # Input question
    question = st.text_input("Ask a question", "")

    if st.button("Get Answer"):
        if question:
            try:
                answer = chain.run(question)
                st.success("Answer: " + answer)
            except Exception as e:
                st.error("An error occurred while processing the question.")
                st.error(str(e))

if __name__ == "__main__":
    main()
