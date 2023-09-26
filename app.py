import streamlit as st
import os
import openai  # Make sure to install the 'openai' library using 'pip install openai'
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Function to load the document and perform question-answering
def load_document_and_answer(api_key, query, uploaded_file):
    # Set the OpenAI API key
    openai.api_key = api_key

    state_of_the_union = uploaded_file.read().decode("utf-8")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(
        texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))]
    )

    docs = docsearch.similarity_search(query)

    # Format the user message as required by OpenAI API
    user_message = {
        "role": "user",
        "content": query
    }

    # Create a list of messages
    messages = [user_message]

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
        OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt
    )

    # Use the messages as input to the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the chat history from the response
    chat_history = response['choices'][0]['message']['content']

    return chat_history

# Streamlit UI
st.title("Chatbot with Document Search")
api_key = st.text_input("Enter your OpenAI API Key:")
query = st.text_input("Ask a question:", "What is the paradox of Levi-Strauss's myth?")

# Allow users to upload a file
uploaded_file = st.file_uploader("Upload a document (TXT file):")

if st.button("Get Answer") and uploaded_file and api_key:
    chat_history = load_document_and_answer(api_key, query, uploaded_file)
    st.write("Chatbot Response:")
    st.write(chat_history)

# Note: The user can input the OpenAI API key, query, and upload a file at the same time.
