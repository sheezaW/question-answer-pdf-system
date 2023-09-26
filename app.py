import streamlit as st
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import os
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize a dictionary to store document content
document_contents = {}

# Function to extract relevant content
def extract_relevant_content(document_contents, question):
    relevant_content = []

    # Process the user's question with spaCy
    question_doc = nlp(question)

    # Iterate through each document's content
    for document_name, document_content in document_contents.items():
        # Process the document with spaCy
        doc = nlp(document_content)

        # Iterate through sentences in the document
        for sentence in doc.sents:
            # Check if the sentence contains entities or keywords from the question
            if any(entity.text in sentence.text or entity.text.lower() in sentence.text.lower() for entity in question_doc.ents):
                relevant_content.append(f"From document '{document_name}': {sentence.text}")

    return relevant_content

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

            # Store the document content in the dictionary
            document_name = os.path.basename(document_path)
            document_contents[document_name] = document_content

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
            "description": f"Good for answering questions about document {i} and telling the reference from the document",
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

def main():
    st.title("Question Answering with Documents")
    st.sidebar.title("Settings")

    # File paths to your documents
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['txt'])

    # Streamlit input field for API key
    openai_api_key = st.text_input("Enter your OpenAI API Key", "")

    if uploaded_files:
        document_paths = [uploaded_file.name for uploaded_file in uploaded_files]
        chain = initialize_qa_chain(document_paths, openai_api_key)

        # Display a successful document upload message
        st.success("Documents successfully uploaded!")

        # Input question
        question = st.text_input("Ask a question", "")

        if st.button("Get Answer"):
            if chain and question:
                try:
                    answer = chain.run(question)
                    st.success("Answer: " + answer)
                    
                    # Extract relevant content based on question
                    relevant_content = extract_relevant_content(document_contents, question)
                    
                    # Display relevant content
                    if relevant_content:
                        st.subheader("Relevant Content:")
                        for content in relevant_content:
                            st.write(content)  # Display relevant content on the screen
                except Exception as e:
                    st.error("An error occurred while processing the question.")
                    st.error(str(e))

if __name__ == "__main__":
    main()
