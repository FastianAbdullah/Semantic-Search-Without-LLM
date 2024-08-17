I apologize for the confusion. You're absolutely right, and I should have understood your request better. Let me provide you with a complete, interactive README.md file that you can directly use in your GitHub repository. Here it is:

```markdown
#  Enhanced Document Semantic Search

Welcome to the Enhanced Document Semantic Search project! This interactive tool allows you to upload documents, scan images, and perform semantic searches using cosine similarity without relying on a large language model.

<details>
<summary> Features</summary>

-  Document Upload: Support for PDF, DOCX, TXT, XLS, and XLSX files
-  Document Scanning: Use your camera to scan documents
-  Semantic Search: Utilizes cosine similarity for accurate results
-  Embeddings: Powered by Hugging Face's sentence transformers
-  Customizable: Adjust chunk size and overlap for optimal results
</details>

<details>
<summary>Quick Start</summary>

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/enhanced-document-search.git
   cd enhanced-document-search
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run faiss_cosine.py
   ```

4. Open your web browser and navigate to the provided local URL (usually http://localhost:8501)
</details>

<details>
<summary>How to Use</summary>

### Document Upload
1. Click on the "📄 Document Upload" tab
2. Upload your document using the file uploader
3. Wait for the document to be processed
4. Enter your question in the search bar
5. Adjust the number of results using the slider
6. Explore the semantic search results!

### Document Scanning
1. Click on the "📷 Document Scan" tab
2. Use your camera to take a picture of the document
3. Wait for the OCR process to complete
4. Review the scanned text
5. Proceed with semantic search as in the Document Upload section
</details>

<details>
<summary>⚙️ Customization</summary>

Use the sidebar to adjust:
- Chunk Size: Controls the size of text chunks for processing
- Chunk Overlap: Determines the overlap between chunks
</details>

<details>
<summary>FAQ</summary>

<details>
<summary>What types of documents can I upload?</summary>
The tool supports PDF, DOCX, TXT, XLS, and XLSX files.
</details>

<details>
<summary>How accurate is the document scanning feature?</summary>
The accuracy depends on the quality of the image and the clarity of the text. For best results, ensure good lighting and a clear focus on the document.
</details>

<details>
<summary>Can I use this tool offline?</summary>
Yes, once you've installed the dependencies, the tool can run offline on your local machine.
</details>

<details>
<summary>How does the semantic search work without a large language model?</summary>
The tool uses sentence transformers to create embeddings and then calculates cosine similarity between the query and document chunks.
</details>
</details>

<details>
<summary>Troubleshooting</summary>

- If you encounter issues with PDF processing, ensure you have the latest version of pdfplumber installed.
- For OCR problems, check that pytesseract is correctly installed and the Tesseract executable is in your system PATH.
</details>

<details>
<summary>Contributing</summary>

Contributions are welcome! Please feel free to submit a Pull Request.
</details>

<details>
<summary>License</summary>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
</details>

<details>
<summary>Acknowledgements</summary>

- [Streamlit](https://streamlit.io/) for the awesome web app framework
- [LangChain](https://github.com/hwchase17/langchain) for document processing utilities
- [Hugging Face](https://huggingface.co/) for the sentence transformers
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
</details>

<details>
<summary>Dependencies</summary>

This project requires the following Python packages:

- streamlit
- langchain-text-splitters
- langchain-community
- sentence-transformers
- faiss-cpu
- pdfplumber
- pandas
- docx2txt
- numpy
- pillow
- pytesseract

You can install these dependencies using the following command:

```
pip install -r requirements.txt
```
</details>

<details>
<summary>Code Overview</summary>

The main script `faiss_cosine.py` contains the following key components:

1. `DataLoader` class: Handles document loading and chunking for various file types.
2. `cosine_similarity` function: Calculates the cosine similarity between two vectors.
3. `scan_document` function: Uses OCR to extract text from scanned images.
4. `main` function: Sets up the Streamlit interface and manages the overall flow of the application.
5. `process_chunks` function: Creates embeddings, builds the FAISS index, and performs the semantic search.

The script uses Streamlit for the user interface, allowing for an interactive experience with file uploads, document scanning, and real-time search results.
</details>

<details>
<summary>Future Improvements</summary>

- Add support for more file types
- Implement multi-language support
- Optimize performance for larger documents
- Integrate with cloud storage services for document management

Feel free to contribute to these improvements or suggest new features!
</details>
```
