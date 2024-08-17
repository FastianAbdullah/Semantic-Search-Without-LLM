import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader  
import os
import pandas as pd
import docx2txt
from typing import List
from langchain.schema import Document as LangChainDocument
import numpy as np
from PIL import Image
import pytesseract

@st.cache_resource
def load_embeddings():
    return HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Add some custom CSS for animations and styling
st.markdown("""
    <style>
    .stButton>button {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(0, 123, 255, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(0, 123, 255, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(0, 123, 255, 0);
        }
    }
    .stTextInput>div>div>input {
        animation: glow 1.5s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from {
            box-shadow: 0 0 5px #e60073, 0 0 10px #e60073, 0 0 15px #e60073, 0 0 20px #e60073;
        }
        to {
            box-shadow: 0 0 10px #ff4da6, 0 0 20px #ff4da6, 0 0 30px #ff4da6, 0 0 40px #ff4da6;
        }
    }
    </style>
    """, unsafe_allow_html=True)


class DataLoader:
    def __init__(self, filepath_or_url, user_agent=None):
        self.filepath_or_url = filepath_or_url
        self.user_agent = user_agent
    
    def load_document(self):
        file_extension = os.path.splitext(self.filepath_or_url)[1].lower()
        
        if file_extension == '.pdf':
            loader = PDFPlumberLoader(self.filepath_or_url)
            documents = loader.load()
        elif file_extension == '.txt':
            with open(self.filepath_or_url, 'r', encoding='utf-8') as file:
                text = file.read()
            documents = [LangChainDocument(page_content=text, metadata={"source": self.filepath_or_url})]
        elif file_extension in ['.docx', '.doc']:
            text = docx2txt.process(self.filepath_or_url)
            documents = [LangChainDocument(page_content=text, metadata={"source": self.filepath_or_url})]
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(self.filepath_or_url)
            documents = [LangChainDocument(page_content=df.to_string(), metadata={"source": self.filepath_or_url})]
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return documents

    def chunk_document(self, documents, chunk_size=1024, chunk_overlap=80):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def process_document(self, chunk_size=1024, chunk_overlap=80):
        documents = self.load_document()
        if len(documents) == 1 and (self.filepath_or_url.endswith(('.xlsx', '.xls'))):
            chunks = documents  
        else:
            chunks = self.chunk_document(documents, chunk_size, chunk_overlap)
        st.write(f"Number of chunks: {len(chunks)}")
        return chunks

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def scan_document(image):
  
    text = pytesseract.image_to_string(image)
    return text

def main():
    st.title("🚀 Enhanced Document Semantic Search")
    st.markdown("### With Cosine Similarity (No LLM) and Document Scanning")

    # Add a sidebar for additional options
    st.sidebar.header("Options")
    chunk_size = st.sidebar.slider("Chunk Size", 256, 2048, 1024)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 80)

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["📄 Document Upload", "📷 Document Scan"])

    with tab1:
        uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT, XLS, XLSX)", type=["pdf", "docx", "txt", "xls", "xlsx"])
        
        if uploaded_file is not None:
            with st.spinner("🔍 Processing document..."):
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                data_loader = DataLoader(temp_file_path)
                chunks = data_loader.process_document(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                os.remove(temp_file_path)
            
            process_chunks(chunks)

    with tab2:
        st.write("📸 Scan a document using your camera or upload an image")
        scanned_image = st.camera_input("Take a picture of your document")
        
        if scanned_image:
            image = Image.open(scanned_image)
            with st.spinner("🔍 Scanning document..."):
                scanned_text = scan_document(image)
            
            st.success("Document scanned successfully!")
            st.text_area("Scanned Text", scanned_text, height=200)
            
            # Create a document from scanned text
            document = LangChainDocument(page_content=scanned_text, metadata={"source": "Scanned Document"})
            chunks = DataLoader("").chunk_document([document], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            process_chunks(chunks)

def process_chunks(chunks):
    if chunks:
        st.text_area("Document Preview", chunks[0].page_content[:1000] + "...", height=200)
        
        with st.spinner("🧠 Creating embeddings and FAISS index..."):
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        
        st.success("Ready for questions!")
        
        query = st.text_input("🔎 Ask a question about the document")
        if query:
            k = st.slider("Number of results", min_value=1, max_value=10, value=5)
            
            with st.spinner("🕵️ Searching..."):
                query_embedding = embeddings.embed_query(query)
                document_embeddings = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)
                
                similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]
                top_k_indices = np.argsort(similarities)[-k:][::-1]
                
                results = [(vectorstore.docstore.search(vectorstore.index_to_docstore_id[i]), similarities[i]) for i in top_k_indices]
            
            if results:
                for i, (doc, score) in enumerate(results, 1):
                    with st.expander(f"Match {i} - Similarity: {score:.4f}"):
                        st.write(f"**Text:** {doc.page_content}")
                        st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
            else:
                st.write("No matches found.")

if __name__ == "__main__":
    main()