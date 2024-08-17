import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PDFPlumberLoader  
import os
import pandas as pd
import docx2txt
from typing import List
# from langchain.schema import Document
from langchain.schema import Document as LangChainDocument

# Load and cache the embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

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

def main():
    st.title("Enhanced Document Semantic Search (No LLM)")
  
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT, XLS, XLSX)", type=["pdf", "docx", "txt", "xls", "xlsx"])
    url = st.text_input("Or enter a URL to crawl")

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getvalue())
            data_loader = DataLoader(temp_file_path)
            chunks = data_loader.process_document()
            os.remove(temp_file_path)
    elif url:
        with st.spinner("Crawling website..."):
            data_loader = DataLoader(url)
            chunks = data_loader.process_document()
    else:
        st.write("Please upload a file or enter a URL.")
        return

    if chunks:
        st.text_area("Document Preview", chunks[0].page_content[:1000] + "...", height=200)
        
        with st.spinner("Creating embeddings and vector store..."):
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        
        query = st.text_input("Ask a question")
        if query:
            k = st.slider("Number of results", min_value=1, max_value=10, value=5)
            
            with st.spinner("Searching..."):
                results = vectorstore.similarity_search_with_score(query, k=k)
            
            if results:
                for i, (doc, score) in enumerate(results, 1):
                    st.write(f"**Match {i}:**")
                    st.write(f"*Text:* {doc.page_content}")
                    st.write(f"*Source:* {doc.metadata.get('source', 'Unknown')}")
                    st.write(f"*Similarity Score:* {1 - score:.4f}")  # Convert distance to similarity
                    st.write("---")
            else:
                st.write("No matches found.")

if __name__ == "__main__":
    main()