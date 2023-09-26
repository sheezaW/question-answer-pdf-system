import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import os
with st.sidebar:
    st.title('Document Chat App')
    st.markdown('''
    ## About
    This app is an OpenAI-powered Chatbot built using:
    - streamlit
    - langchain
    - OpenAI
    '''    
    )
    add_vertical_space(5)
    st.write('Made by Solutyics')


# Function to initialize the MultiRetrievalQAChain
def initialize_qa_chain(document_paths):
    # Initialize an empty list to store document objects
    documents = []

    # Create retrievers for each document
    retrievers = []

    for document_path in document_paths:
        try:
            # Load the text file with UTF-8 encoding
            with open(document_path, 'r', encoding='utf-8') as file:
                document_content = file.read()

            # Split the document
            document = TextLoader(document_content).load_and_split()
            documents.append(document)

            # Create retriever
            retriever = FAISS.from_documents(document, OpenAIEmbeddings()).as_retriever()
            retrievers.append(retriever)
        except Exception as e:
            #st.error(f"Error loading document {document_path}: {str(e)}")
            pass

        # Define the retriever information
        retriever_infos = []

    for i, retriever in enumerate(retrievers):
        retriever_info = {
            "name": f"document_{i}",
            "description": f"Good for answering questions about document {i}",
            "retriever": retriever
        }
        retriever_infos.append(retriever_info)

    # Streamlit input field for API key
    openai_api_key = st.text_input("Enter your OpenAI API Key", "")

# Set the OpenAI API key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Create the MultiRetrievalQAChain if API key is provided
    chain = None
    if openai_api_key:
        try:
            # Set the OpenAI API key
            OpenAI(api_key=openai_api_key)
            chain = MultiRetrievalQAChain.from_retrievers(OpenAI(), retriever_infos)
        except Exception as e:
            st.error("An error occurred while initializing the chain.")
            st.error(str(e))

    return chain

# Streamlit app
def main():
    st.title("Question Answering with Documents")
    st.sidebar.title("Settings")

    # File paths to your documents
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['txt'])

    if uploaded_files:
        document_paths = [uploaded_file.name for uploaded_file in uploaded_files]
        chain = initialize_qa_chain(document_paths)

        # Display a successful document upload message
        st.success("Documents successfully uploaded!")

        # Input question
        question = st.text_input("Ask a question", "")

        if st.button("Get Answer"):
            if chain and question:
                try:
                    answer = chain.run(question)
                    st.success("Answer: " + answer)
                except Exception as e:
                    st.error("An error occurred while processing the question.")
                    st.error(str(e))

if __name__ == "__main__":
    main()
