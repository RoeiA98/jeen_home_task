# Document Vectorization System

## About
This project is a Python-based document processing system designed to ingest PDF and DOCX files, split them into manageable text chunks, generate vector embeddings using Google's Gemini API, and store the results in a PostgreSQL database for future retrieval or semantic search tasks.

## Features
- **Multi-format Support**: Handles both `.pdf` and `.docx` files.
- **Smart Chunking**: Splits text into logical paragraphs based on double newlines.
- **Advanced Embeddings**: Utilizes Google Gemini's `text-embedding-004` model.
- **Persistent Storage**: Saves text chunks, embeddings, and metadata into a PostgreSQL database.
- **Automatic Schema Management**: Automatically creates the necessary database table if it doesn't exist.

## Requirements
- Python 3.8+
- PostgreSQL Database
- Google Gemini API Key

### Python Packages
Install the required dependencies:
```bash
pip install pypdf python-docx psycopg2-binary google-genai python-dotenv
```

### Environment Variables
Create a `.env` file in the root directory with the following keys:
```dotenv
GEMINI_API_KEY=your_api_key_here
DATABASE_URL=postgresql://username:password@host:port/database
```

## Usage
1. **Configure Environment**: Ensure your `.env` file is set up with valid credentials.
2. **Prepare Files**: Place your documents (e.g., `test2.docx`) in the target directory (`./samples/`).
3. **Run the Script**:
   ```bash
   python index_documents.py
   ```

To process different files, modify the `__main__` block in `index_documents.py`:
```python
vectorizer.process_file("./path/to/your/document.pdf")
```

## The Process
1. **Initialization**: Connects to PostgreSQL and initializes the Google GenAI client. Checks for the existence of the `document_embeddings` table and creates it if missing.
2. **Extraction**: Reads the raw text content from the provided PDF or DOCX file.
3. **Chunking**: Splits the raw text into smaller segments (paragraphs) using double newlines as delimiters.
4. **Embedding**: Sends each text chunk to the Gemini API (`text-embedding-004`) to generate a vector representation.
5. **Storage**: Inserts the text chunk, its vector embedding, filename, and metadata into the database.

## Example output
When running the script successfully, you will see output similar to:

```text
Vectorizer initialized
Processing file_name
Successfully processed X chunks for file_name.
File has been successfully proccessed and stored in the DB.
```

## Database Example
![Alt text](https://i.ibb.co/nMpFjx9D/Screenshot-2025-12-23-at-19-36-25.png)
