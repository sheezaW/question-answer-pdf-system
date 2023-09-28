import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Initialize Streamlit app
st.title("Document Question Answering App")
st.sidebar.header("Settings")

# User input for URL
url_input = st.sidebar.text_input("Enter the URL of the document")

# User input for questions
question = st.sidebar.text_input("Ask a question about the document")

if st.sidebar.button("Load Document"):
    # Load the document from the provided URL
    loader = WebBaseLoader(url_input)
    data = loader.load()

    # Split the document into smaller chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # Create embeddings and a document search index
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

    # Perform similarity search and retrieve relevant documents
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),
                                                      llm=ChatOpenAI(temperature=0))
    unique_docs = retriever_from_llm.get_relevant_documents(query=question)

    # Initialize the question-answering chain
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Get the answer to the question
    result = qa_chain({"query": question})

    # Display the answer
    st.subheader("Chatbot's Response:")
    st.write(result["result"])
