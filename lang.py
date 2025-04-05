from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"

# Function to load PDF files
def pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Function to split text into chunks
def chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Function to generate embeddings
def embeddings():
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return model

# Load the documents
documents = pdf_files(DATA_PATH)

# Split the text into chunks
text_chunks = chunks(documents)

# Get the embedding model
model = embeddings()

# Create FAISS vector database
DB_FAISS = "vectorstore/db_faiss"
database = FAISS.from_documents(text_chunks, model)

# Save FAISS database
database.save_local(DB_FAISS)

print("FAISS database saved successfully!")
