import streamlit as st
from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import os

# Initialize conversation history as an empty list
conversation_history = []

# Initialize the MultiRetrievalQAChain
def initialize_qa_chain(document_paths):
    # Initialize an empty list to store document objects
    documents = []
    retriever_infos = []

    for document_path in document_paths:
        try:
            # Load the text file with UTF-8 encoding
            with open(document_path, 'r', encoding='utf-8') as file:
                document_content = file.read()

            # Create a TextLoader for the document content
            text_loader = TextLoader(document_content)

            # Split the document
            document = text_loader.load_and_split()
            documents.append(document)

            # Create retriever
            retriever = FAISS.from_documents(text_loader, OpenAIEmbeddings()).as_retriever()
            retriever_info = {
                "name": f"document_{len(documents) - 1}",
                "description": f"Good for answering questions about document {len(documents) - 1}",
                "retriever": retriever
            }
            retriever_infos.append(retriever_info)
        except Exception as e:
            pass

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
    st.title("Document Chat App")
    st.sidebar.title("Settings")

    # File paths to your documents
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['txt'])

    if uploaded_files:
        document_paths = [uploaded_file.name for uploaded_file in uploaded_files]
        chain = initialize_qa_chain(document_paths)

        # Display a successful document upload message
        st.success("Documents successfully uploaded!")

    # Input question
    user_input = st.text_input("You:", "")

    if st.button("Send"):
        if chain and user_input:
            try:
                # Append user's message to the conversation history
                conversation_history.append(f"You: {user_input}")
                
                # Get the model's response
                response = chain.run(" ".join(conversation_history))
                
                # Append model's response to the conversation history
                conversation_history.append(f"Model: {response}")
                
                # Display the conversation in a chat-like format
                st.text("Conversation:")
                for msg in conversation_history:
                    if msg.startswith("You:"):
                        st.text_input("", msg[5:], key=msg)
                    else:
                        st.text_input("", msg[7:], key=msg)
                
            except Exception as e:
                st.error("An error occurred while processing the question.")
                st.error(str(e))

if __name__ == "__main__":
    main()
