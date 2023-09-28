import pickle
import streamlit as st
from PyPDF2 import PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss_vectorstore import FAISSVectorStore
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

def main():
    st.header("Chat with PDF 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # Input for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key:")

    # Initialize text embeddings model with the API key
    embeddings = None

    if openai_api_key:
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    if pdf is not None:
        pdf_reader = PdfFileReader(pdf)

        text = ""
        for page in range(pdf_reader.getNumPages()):
            text += pdf_reader.getPage(page).extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Get the file name without extension
        store_name = pdf.name[:-4]

        pickle_filename = f"{store_name}.pkl"

        if os.path.exists(pickle_filename) and os.path.getsize(pickle_filename) > 0:
            with open(pickle_filename, "rb") as f:
                VectorStore = pickle.load(f)
        else:
            if embeddings is not None:
                VectorStore = FAISSVectorStore.from_texts(texts=chunks, embeddings=embeddings)
                with open(pickle_filename, "wb") as f:
                    pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            if embeddings is not None:  # Check if embeddings is defined
                # Retrieve similar documents based on user query
                query_vector = embeddings.encode_text(query)
                similar_docs = VectorStore.search(query_vector, k=3)

                # Initialize OpenAI chatbot
                llm = OpenAI()
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    responses = []

                    for doc in similar_docs:
                        response = chain.run(input_documents=[doc], question=query)
                        responses.append(response)

                    for i, response in enumerate(responses):
                        st.write(f"Answer from Document {i + 1}:")
                        st.write(response)

if __name__ == '__main__':
    main()
