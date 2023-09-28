import os
import streamlit as st
import pickle
from pdfplumber import PdfReader
from vectorhub.encoders.text.tfhub import USE2
from vectorhub.faiss import SimilarityEngine
from vectorhub.tools.vectors import Vector
from transformers import pipeline

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def main():
    st.header("Chat with PDF 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Get the file name without extension
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            # Initialize text embeddings model
            embeddings = USE2()
            VectorStore = SimilarityEngine()
            for chunk in chunks:
                vector = embeddings.encode(chunk)
                VectorStore.add(Vector(data=vector))

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            # Retrieve similar documents based on user query
            query_vector = embeddings.encode(query)
            similar_docs = VectorStore.search(query_vector, k=3)

            # Initialize OpenAI chatbot
            if OPENAI_API_KEY:
                chatbot = pipeline("question-answering", model="gpt-3.5-turbo", token=OPENAI_API_KEY)
                responses = []

                for doc in similar_docs:
                    response = chatbot(question=query, context=doc.data)
                    responses.append(response)

                for i, response in enumerate(responses):
                    st.write(f"Answer from Document {i + 1}:")
                    st.write(response['answer'])
            else:
                st.warning("OpenAI API key not found. Please set it in your environment variables.")

if __name__ == '__main__':
    main()
