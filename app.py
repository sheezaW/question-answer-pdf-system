import streamlit as st
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import openai  # Make sure you have the 'openai' library installed
import os

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Function to initialize the MultiRetrievalQAChain
def initialize_qa_chain(document_paths, openai_api_key):
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
            # st.error(f"Error loading document {document_path}: {str(e)}")
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

    # Create the MultiRetrievalQAChain
    chain = None
    if openai_api_key:
        try:
            # Set the OpenAI API key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            OpenAI(api_key=openai_api_key)
            chain = MultiRetrievalQAChain.from_retrievers(OpenAI(), retriever_infos)
        except Exception as e:
            st.error("An error occurred while initializing the chain.")
            st.error(str(e))

    return chain

# Function to perform similarity search and retrieve context
# Function to perform similarity search and retrieve context
def similarity_search(chain, question):
    retrieved_context = None
    if chain and question:
        try:
            # Create a list of candidate contexts from all documents
            candidate_contexts = [doc.text for doc in chain.documents]

            # Use the OpenAI Search API to find the most relevant context
            search_response = openai.Answer.create(
                search_model="davinci",
                question=question,
                documents=candidate_contexts,
                max_responses=1  # Adjust as needed
            )

            if search_response.data:
                retrieved_context = search_response.data[0].document
        except Exception as e:
            st.error("An error occurred while performing similarity search.")
            st.error(str(e))
    return retrieved_context

# Function to get an answer using OpenAI GPT-3.5 Turbo API
def get_gpt_answer(context, question, document_source):
    try:
        # Set your OpenAI GPT-3.5 Turbo API key here
        api_key = "YOUR_OPENAI_API_KEY"

        # Call OpenAI's GPT-3.5 Turbo API to get an answer
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use the "gpt-3.5-turbo" engine
            prompt=f"Context: {context}\nQuestion: {question}\nAnswer (from document {document_source}):",
            max_tokens=50,  # Adjust max tokens as needed
            api_key=api_key
        )

        answer = response.choices[0].text.strip()
        return answer
    except Exception as e:
        st.error("An error occurred while calling the GPT-3.5 Turbo API.")
        st.error(str(e))

# Streamlit app
def main():
    st.title("Question Answering with Documents")
    st.sidebar.title("Settings")

    # File paths to your documents
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['txt'])

    if uploaded_files:
        document_paths = [uploaded_file.name for uploaded_file in uploaded_files]

        # Input OpenAI API key
        openai_api_key = st.text_input("Enter your OpenAI API Key", "")

        chain = initialize_qa_chain(document_paths, openai_api_key)

        # Display a successful document upload message
        st.success("Documents successfully uploaded!")

        # Input question
        question = st.text_input("Ask a question", "")

        if st.button("Get Answer"):
            if chain and question:
                # Perform similarity search to retrieve context
                retrieved_context = similarity_search(chain, question)

                if retrieved_context:
                    # Now you can pass the retrieved context to GPT for answering
                    answer = get_gpt_answer(retrieved_context, question, "source_document")
                    st.success("Answer: " + answer)
                else:
                    st.error("No relevant context found.")

if __name__ == "__main__":
    main()
