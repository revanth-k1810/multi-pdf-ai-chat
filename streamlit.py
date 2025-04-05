import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "uploaded_files/"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def process_pdfs(uploaded_files):
    os.makedirs(DATA_PATH, exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    database = FAISS.from_documents(text_chunks, embedding_model)
    database.save_local(DB_FAISS_PATH)
    return database

def set_custom_prompt():
    template = """
    You are a helpful AI assistant. Use the following context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Chat History: {chat_history}
    
    Question: {question}
    
    Answer:
    """
    return PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    groq_chat = ChatGroq(
        groq_api_key="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",
        model_name="llama3-8b-8192",
        temperature=0.5
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=groq_chat,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": set_custom_prompt()},
        chain_type="stuff",
        verbose=True
    )
    return conversation_chain

def main():
    st.title("AI Chatbot with Document Upload (Powered by Groq)")
    st.subheader("Upload PDF files and have a conversation based on their content")
    
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info("Processing uploaded documents...")
        vectorstore = process_pdfs(uploaded_files)
        st.success("Documents processed and stored in the vector database!")
    else:
        vectorstore = get_vectorstore()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm your document assistant. Upload some PDFs or ask me questions about previously uploaded documents."
        })
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question based on the uploaded documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            # Initialize conversation chain if not already in session state
            if "conversation_chain" not in st.session_state:
                st.session_state.conversation_chain = get_conversation_chain(vectorstore)
            
            # Get response from the conversation chain
            response = st.session_state.conversation_chain({"question": prompt})
            answer = response["answer"]
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()