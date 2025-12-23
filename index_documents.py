import os
import datetime
import psycopg2
import google.genai as genai
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class DocumentVectorizer:
    def __init__(self, db_url):
        self.conn = psycopg2.connect(db_url)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self._init_db()

    def _init_db(self):
        table_query = """
        CREATE TABLE IF NOT EXISTS "HomeTask_beforebone".document_embeddings (
            id SERIAL PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            embedding FLOAT8[] NOT NULL,
            file_name TEXT NOT NULL,
            split_strategy TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(table_query)
        self.conn.commit()

    def extract_text(self, file_path):
        """Extracts text from PDF or DOCX files."""
        ext = os.path.splitext(file_path)[1].lower()
        text = ""
        
        if ext == '.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n\n"
        elif ext == '.docx':
            doc = Document(file_path)
            text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        return text.strip()

    def chunk_text(self, text):
        """Chunks text into paragraphs based on double newlines."""
        # Simple splitting strategy: split by double newlines to get paragraphs
        chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
        return chunks

    def get_embedding(self, text):
        """Generates embedding using Gemini API."""
        response = self.client.models.embed_content(model="text-embedding-004", contents=text)
        return response.embeddings[0].values

    def process_file(self, file_path):
        """Main pipeline: Extract -> Chunk -> Embed -> Save."""
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")

        raw_text = self.extract_text(file_path)
        chunks = self.chunk_text(raw_text)
        split_strategy = "paragraph_split_newline"

        # Embed & Save
        with self.conn.cursor() as cur:
            for chunk in chunks:
                embedding = self.get_embedding(chunk)
                insert_query = """
                INSERT INTO document_embeddings 
                (chunk_text, embedding, file_name, split_strategy, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """
                cur.execute(insert_query, (chunk, embedding, file_name, split_strategy, datetime.datetime.now()))
    
        self.conn.commit()
        print(f"Successfully processed {len(chunks)} chunks for {file_name}.")

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    try:
        vectorizer = DocumentVectorizer(DB_URL)
        print("Vectorizer initialized")
        vectorizer.process_file("./samples/test1.docx")
        print("File has been successfully proccessed and stored in the DB.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'vectorizer' in locals():
            vectorizer.close()