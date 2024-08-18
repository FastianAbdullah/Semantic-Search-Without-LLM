# AI-Powered Document Semantic Search with Web Crawling

This project implements an AI-powered document semantic search system using Streamlit, LangChain, and various document processing techniques. It allows users to upload documents, scan images, crawl web pages, process the content, and perform semantic searches without relying on a large language model.

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Features](#features-1)
- [Contributing](#contributing)
- [License](#license)

## Features
- Document upload and processing (PDF, DOCX, TXT, XLS, XLSX)
- Image scanning and OCR for text extraction
- Web crawling for content retrieval
- Text chunking and embedding using Hugging Face's sentence transformers
- FAISS vector store for efficient similarity search
- Cosine similarity-based semantic search without LLM
- Customizable chunk size and overlap
- Streamlit-based user interface for easy interaction

## System Architecture
Here's a high-level overview of the system architecture:

```mermaid

graph TD
    A[User] -->|Upload Document/Scan Image/Enter URL| B[Streamlit UI]
    B -->|Process Document| C[DataLoader]
    B -->|Crawl Web| H[WebCrawler]
    C -->|Chunk Text| D[Text Splitter]
    H -->|Extract Content| D
    D -->|Generate Embeddings| E[HuggingFace Embeddings]
    E -->|Store Vectors| F[FAISS Vector Store]
    A -->|Enter Query| B
    B -->|Retrieve Similar Chunks| F
    F -->|Calculate Cosine Similarity| G[Semantic Search]
    G -->|Return Results| B
    B -->|Display Results| A
```

## Setup
1. Clone the repository:
   ```
   https://github.com/FastianAbdullah/Semantic-Search-Without-LLM.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
## Usage:
   To start the application, run:
   ```
   streamlit run doc_search_and_crawl.py
   ```
The application will open in your default web browser.

## API Endpoints

This project is primarily a Streamlit web application and does not expose traditional API endpoints. However, the main functionalities are accessible through the Streamlit UI:

<details>
<summary>1. Document Upload</summary>

* Upload documents (PDF, DOCX, TXT, XLS, XLSX) using the file uploader in the "Document Upload" tab.
</details>

<details>
<summary>2. Document Scanning</summary>

* Use your camera to scan documents in the "Document Scan" tab.
</details>

<details>
<summary>3. Semantic Search</summary>

* Enter your query in the search bar after processing a document or scanned image.
</details>

<details>
<summary>4. Web Crawling</summary>

* Enter a URL in the "Web Crawl" tab to fetch and process web content.
</details>


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

<details>
<summary>View Acknowledgements</summary>

* Streamlit for the web app framework
* LangChain for document processing utilities
* Hugging Face for sentence transformers
* FAISS for efficient similarity search
* BeautifulSoup for web scraping
</details>

## Dependencies

<details>
<summary>View Dependencies</summary>

* streamlit
* langchain-text-splitters
* langchain-community
* sentence-transformers
* faiss-cpu
* pdfplumber
* pandas
* docx2txt
* numpy
* pillow
* pytesseract
* PyPDF2
</details>

## Code Overview

The main script `doc_search_and_crawl.py` contains the following key components:

<details>
<summary>View Components</summary>

1. `DataLoader` class: Handles document loading and chunking for various file types.
2. `WebCrawler` class: Handles web content retrieval and parsing.
3. `cosine_similarity` function: Calculates the cosine similarity between two vectors.
4. `scan_document` function: Uses OCR to extract text from scanned images.
5. `main` function: Sets up the Streamlit interface and manages the overall flow of the application.
6. `process_chunks` function: Creates embeddings, builds the FAISS index, and performs the semantic search.
</details>

## Future Improvements

<details>
<summary>View Future Improvements</summary>

* Add support for more file types
* Implement multi-language support
* Optimize performance for larger documents
* Integrate with cloud storage services for document management
* Implement advanced web crawling features (depth control, multiple page crawling)
</details>

Feel free to contribute to these improvements or suggest new features!
