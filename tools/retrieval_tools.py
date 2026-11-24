
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
import chromadb


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "insurance_guidelines"


@lru_cache(maxsize=1)
def load_collection():

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


@lru_cache(maxsize=1)
def load_embedding_model():

    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


@tool
def retrieve_guidelines(query: str, top_k: int = 3) -> dict:
    """Retrieve relevant insurance investigation guidelines.

    Args:
        query: Investigation question or risk indicators.
        top_k: Number of relevant sections to retrieve.
    """

    if not query.strip():
        return {"error": "Query cannot be empty."}

    if not 1 <= top_k <= 10:
        return {"error": "top_k must be between 1 and 10."}

    try:

        collection = load_collection()
        embedding_model = load_embedding_model()

        if collection.count() == 0:
            return {"error": "Knowledge base is empty."}

        # Convert the query into an embedding
        query_embedding = embedding_model.embed_query(query)

        # Search the vector database
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_guidelines = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            retrieved_guidelines.append({
                "citation": (
                    f"{metadata['document_id']}, "
                    f"Section {metadata['section']}"
                ),
                "title": metadata["title"],
                "source": metadata["source"],
                "version": metadata["version"],
                "content": document,
                "similarity_score": round(
                    1 - float(distance), 4
                )
            })

        return {
            "query": query,
            "retrieved_count": len(retrieved_guidelines),
            "guidelines": retrieved_guidelines,
            "disclaimer": (
                "These are fictional investigation guidelines "
                "for a synthetic insurance demonstration."
            )
        }

    except Exception as error:

        return {
            "error": (
                f"Knowledge base retrieval failed: "
                f"{type(error).__name__}"
            )
        }


if __name__ == "__main__":

    result = retrieve_guidelines.invoke({
        "query": (
            "Recently purchased insurance policy "
            "with multiple previous claims"
        ),
        "top_k": 3
    })

    if "error" in result:
        print(result["error"])

    else:

        print("\n--- RETRIEVED GUIDELINES ---")

        for guideline in result["guidelines"]:

            print("\nCitation:", guideline["citation"])
            print("Title:", guideline["title"])
            print("Similarity:", guideline["similarity_score"])
            print("Content:", guideline["content"])
