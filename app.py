import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
import os

# Define a function to load and process the document
def process_document(file_contents):
    # Decode the file contents from bytes to a string
    file_contents = file_contents.decode("utf-8")  # You may need to use a different encoding if your file is not UTF-8 encoded
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(file_contents, separator="\n")  # Use a string separator
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(
        texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))]
    )
    return docsearch, texts

# Define a Streamlit app
def main():
    st.title("Document Question Answering App")
    st.sidebar.header("Settings")

    # User input for OpenAI key
    openai_key = st.sidebar.text_input("Enter your OpenAI API key")

    # User input for uploading a text file
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        st.sidebar.success("File uploaded successfully!")

        # Process the document and create a document search index
        docsearch, texts = process_document(file_contents)

        # User input for questions
        question = st.text_input("Ask a question about the document")

        if st.button("Ask Question"):
            if not openai_key:
                st.error("Please enter your OpenAI API key.")
            elif not question:
                st.error("Please enter a question.")
            else:
                # Load the question-answering chain
                template = """You are a chatbot having a conversation with a human.

                Given the following extracted parts of a long document and a question, create a final answer.

                {context}

                {chat_history}
                Human: {human_input}
                Chatbot:"""

                prompt = PromptTemplate(
                    input_variables=["chat_history", "human_input", "context"], template=template
                )
                memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
                chain = load_qa_chain(
                    OpenAI(temperature=0, api_key=openai_key), chain_type="stuff", memory=memory, prompt=prompt
                )

                # Perform similarity search and generate a response
                docs = docsearch.similarity_search(question)
                response = chain({"input_documents": docs, "human_input": question}, return_only_outputs=True)

                # Display the response
                st.subheader("Chatbot's Response:")
                st.write(response)

                # Store the conversation history
                chain.memory.buffer.append(f"Human: {question}")
                chain.memory.buffer.append(f"Chatbot: {response}")

    # Option to ask more questions about the same document
    if st.button("Ask More Questions"):
        question = st.text_input("Ask another question about the document")
        if question:
            # Perform similarity search and generate a response
            docs = docsearch.similarity_search(question)
            response = chain({"input_documents": docs, "human_input": question}, return_only_outputs=True)

            # Display the response
            st.subheader("Chatbot's Response:")
            st.write(response)

            # Store the conversation history
            chain.memory.buffer.append(f"Human: {question}")
            chain.memory.buffer.append(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
