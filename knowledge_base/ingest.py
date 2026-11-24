
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import chromadb
import re


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DOCUMENT_PATH = (
    ROOT / "knowledge_base" / "investigation_guidelines.txt"
)

DB_PATH = ROOT / "chroma_db"

COLLECTION_NAME = "insurance_guidelines"


def ingest_documents():

    # Read the knowledge base document
    text = DOCUMENT_PATH.read_text(encoding="utf-8")

    # Split the document by section
    sections = re.split(
        r"(?=SECTION \d+:)",
        text
    )

    chunks = [
        section.strip()
        for section in sections
        if section.strip().startswith("SECTION")
    ]

    print(f"Total chunks created: {len(chunks)}")

    # Generate embeddings
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    embeddings = embedding_model.embed_documents(chunks)

    print(
        f"Embedding dimensions: {len(embeddings[0])}"
    )

    # Initialize persistent ChromaDB storage
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = []
    metadata = []

    for chunk in chunks:

        section_number = re.search(
            r"SECTION (\d+):",
            chunk
        ).group(1)

        ids.append(f"GUIDE-001-SECTION-{section_number}")

        metadata.append({
            "document_id": "GUIDE-001",
            "source": "investigation_guidelines.txt",
            "section": int(section_number),
            "title": chunk.splitlines()[0],
            "version": "1.0",
            "synthetic": True
        })

    # Store documents, embeddings and metadata
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata
    )

    print(f"Stored chunks: {collection.count()}")
    print("Knowledge base ingestion completed.")


if __name__ == "__main__":
    ingest_documents()
