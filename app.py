import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# Page title
st.title("Question Answering with Langchain Streamlit App")

# Prompt the user to enter their OpenAI API key
openai_api_key = st.text_input("Enter your OpenAI API key:")

# Prompt the user to enter the questions
questions = st.text_input("Enter one or more questions (separated by a comma):")

# Prompt the user to enter the document paths (comma-separated)
document_paths = st.text_input("Enter document paths (comma-separated):")

# Convert user input to a list of questions and document paths
questions = questions.split(',')
document_paths = document_paths.split(',')

# Initialize an empty list to store document pages
pages = []

# Read the content of each document and add it to the list of pages
for document_path in document_paths:
    with open(document_path.strip(), "r") as f:
        document_content = f.read()
        # Create Document objects for each document with the content
        pages.append(Document(page_content=document_content, metadata={"source": document_path.strip()}))

# Combine the content of all specified documents into one large document
combined_document = "\n".join([page.page_content for page in pages])

# Create a CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Split the combined document into smaller chunks
texts = text_splitter.split_text(combined_document)
embeddings = OpenAIEmbeddings()

# Create a Chroma vector store
docsearch = Chroma.from_texts(
    texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))]
)

# Initialize FAISS index from the document pages
faiss_index = FAISS.from_documents(pages, embeddings)

# Define a template for the conversation
template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

# Create a PromptTemplate
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=template
)

# Create a conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

# Load the question-answering chain
chain = load_qa_chain(
    OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt
)

# Perform the similarity search and question-answering for each question when a button is clicked
if st.button("Answer Questions"):
    for question in questions:
        docs = faiss_index.similarity_search(question, k=2)  # Use question for similarity search
        chain({"input_documents": docs, "human_input": question}, return_only_outputs=True)
        st.write(f"Question: {question}")
        st.write("Answer:", chain.memory.buffer)
        for doc in docs:
            st.write(str(doc.metadata["source"]) + ":", doc.page_content[:500])
