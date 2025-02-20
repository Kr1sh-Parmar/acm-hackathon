import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from flask import Flask, render_template

# Streamlit configuration
st.set_page_config(page_title="Chat with PDF", page_icon="📚")

# Set up the Gemini API key
GEMINI_API_KEY = "API 1."

# Function to process uploaded files and create a vector store
@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes):
    try:
        with st.spinner("Processing and indexing the document..."):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes.read())
                tmp_file_path = tmp_file.name

            # Load PDF document using PyMuPDFLoader
            loader = PyMuPDFLoader(tmp_file_path)
            documents = loader.load()

            if not documents:
                st.error("❌ Could not load the PDF document!")
                st.stop()

            st.success(f"✅ Successfully loaded PDF with {len(documents)} pages")

            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Adjust chunk size
                chunk_overlap=200  # Add overlap for context
            )
            split_documents = text_splitter.split_documents(documents)

            # Create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY,
                credentials=None
            )

            # Create vector store
            vector_store = FAISS.from_documents(split_documents, embeddings)
            st.success("✅ Successfully created vector store!")

            # Clean up the temporary file
            os.unlink(tmp_file_path)

            return vector_store

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a PDF and ask me questions about it!"}
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Streamlit user interface
st.title("Chat With PDF")
st.caption("Drag and drop your PDF file below, then start chatting!")

# Drag-and-drop file upload
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"], accept_multiple_files=False)

if uploaded_file:
    pdf_bytes = BytesIO(uploaded_file.read())
    st.session_state.vector_store = process_pdf(pdf_bytes)

# Chat interface
if st.session_state.vector_store:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

    chat_model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,  # Lower temperature for more focused answers
        credentials=None,
        convert_system_message_to_human=True
    )

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),  # Increase k
        memory=memory,
        return_source_documents=True,
        chain_type="stuff",
        verbose=True
    )

    # Input prompt
    prompt = st.chat_input("Your question")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input and generate responses
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    question = st.session_state.messages[-1]["content"]
                    response = retrieval_chain({
                        "question": f"Answer the following question based on the document: {question}",
                        "chat_history": [(msg["role"], msg["content"]) 
                                          for msg in st.session_state.messages 
                                          if msg["role"] != "assistant"]
                    })

                    # Display response
                    st.write(response['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})

                    # Display source documents
                    if 'source_documents' in response:
                        with st.expander("View Source Documents"):
                            for i, doc in enumerate(response['source_documents']):
                                st.write(f"Source {i+1}:")
                                st.write(doc.page_content)
                                st.write("---")

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
else:
    st.info("ℹ️ Upload a PDF to get started.")