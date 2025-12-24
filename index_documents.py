import os
import datetime
import time
import psycopg2
import google.genai as genai
from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from docx import Document


load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SCHEMA_NAME = "HomeTask_beforebone"
TABLE_NAME = "document_embeddings"
FULL_TABLE_NAME = f'"{SCHEMA_NAME}".{TABLE_NAME}'


class DocumentVectorizer:
    def __init__(self, db_url):
        if not db_url:
            raise ValueError("DATABASE_URL is missing/empty.")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing/empty.")

        self.conn = psycopg2.connect(db_url)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self._init_db()

    def _init_db(self):
        table_query = f"""
        CREATE SCHEMA IF NOT EXISTS "{SCHEMA_NAME}";
        CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            embedding FLOAT8[] NOT NULL,
            file_name TEXT NOT NULL,
            split_strategy TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self.conn, self.conn.cursor() as cur:
            cur.execute(table_query)

    def extract_text(self, file_path):
        """Extracts text from PDF or DOCX files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        ext = os.path.splitext(file_path)[1].lower()
        text = ""

        if ext == ".pdf":
            try:
                reader = PdfReader(file_path)
            except PdfReadError as e:
                raise ValueError(f"Failed to read PDF: {file_path}") from e

            if getattr(reader, "is_encrypted", False):
                raise ValueError(f"PDF is encrypted and cannot be processed: {file_path}")

            for i, page in enumerate(reader.pages):
                try:
                    extracted = page.extract_text() or ""
                except Exception as e:
                    # Skip problematic pages instead of failing the whole file
                    print(f"Warning: failed extracting text from page {i} in {file_path}: {e}")
                    extracted = ""
                if extracted.strip():
                    text += extracted + "\n\n"

        elif ext == ".docx":
            try:
                doc = Document(file_path)
            except Exception as e:
                raise ValueError(f"Failed to read DOCX: {file_path}") from e

            text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return text.strip()

    def chunk_text(self, text):
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        return chunks

    def get_embedding(self, text, *, max_retries=3, base_delay_s=1.0):
        """Generates embedding using Gemini API with basic retry/backoff."""
        if not text or not text.strip():
            return []

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.embed_content(model="text-embedding-004", contents=text)
                return response.embeddings[0].values
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    sleep_s = base_delay_s * (2 ** (attempt - 1))
                    time.sleep(sleep_s)
                else:
                    raise RuntimeError(f"Embedding failed after {max_retries} attempts.") from last_err

    def process_file(self, file_path):
        """Main pipeline: Extract -> Chunk -> Embed -> Save."""
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")

        raw_text = self.extract_text(file_path)

        if not raw_text or not raw_text.strip():
            split_strategy = "empty_file"
            insert_query = f"""
            INSERT INTO {FULL_TABLE_NAME}
            (chunk_text, embedding, file_name, split_strategy, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """
            with self.conn, self.conn.cursor() as cur:
                cur.execute(insert_query, ("", [], file_name, split_strategy, datetime.datetime.now()))
            print(f"File {file_name} is empty; skipped embeddings.")
            return

        chunks = self.chunk_text(raw_text)
        split_strategy = "paragraph_split_newline"

        insert_query = f"""
        INSERT INTO {FULL_TABLE_NAME}
        (chunk_text, embedding, file_name, split_strategy, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """

        inserted = 0
        with self.conn, self.conn.cursor() as cur:
            for idx, chunk in enumerate(chunks):
                try:
                    embedding = self.get_embedding(chunk)
                    cur.execute(insert_query, (chunk, embedding, file_name, split_strategy, datetime.datetime.now()))
                    inserted += 1
                except Exception as e:
                    print(f"Warning: failed processing chunk {idx} for {file_name}: {e}")

        print(f"Successfully processed {inserted}/{len(chunks)} chunks for {file_name}.")

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

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