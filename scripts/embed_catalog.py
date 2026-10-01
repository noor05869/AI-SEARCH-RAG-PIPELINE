import os
import sys
import time
import pathlib
import logging
from dotenv import load_dotenv
from sqlmodel import Session, select
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Add src to python path for imports
sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))

from fastapi_blog.database import engine
from fastapi_blog.models import Listing

# Silence warnings
logging.getLogger("google.genai").setLevel(logging.ERROR)

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "techvault_products"

def get_existing_product_ids(client: QdrantClient, collection_name: str) -> set:
    """Fetch set of product_ids already indexed in Qdrant to enable resume/checkpointing."""
    indexed_ids = set()
    try:
        if not client.collection_exists(collection_name):
            return indexed_ids
        
        offset = None
        while True:
            scroll_res = client.scroll(
                collection_name=collection_name,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            points, next_offset = scroll_res
            for p in points:
                if p.payload and "metadata" in p.payload:
                    pid = p.payload["metadata"].get("product_id")
                    if pid:
                        indexed_ids.add(pid)
            if next_offset is None:
                break
            offset = next_offset
    except Exception as e:
        print(f"[!] Warning checking existing Qdrant IDs: {e}")
    return indexed_ids

def embed_catalog_from_db(limit_count: int | None = None, batch_size: int = 100):
    """Embed database listings using FREE local BAAI/bge-small-en-v1.5 model with ZERO API limits."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)

    print("[+] Checking existing indexed listings in Qdrant for checkpoint resume...")
    already_indexed_ids = get_existing_product_ids(client, COLLECTION_NAME)
    print(f"[*] Found {len(already_indexed_ids)} listings already indexed in Qdrant.")

    print("[+] Connecting to TechVault SQL database to fetch active listings...")
    try:
        with Session(engine) as session:
            statement = select(Listing).where(Listing.is_deleted == False)
            if limit_count:
                statement = statement.limit(limit_count)
            all_listings = session.exec(statement).all()
    except Exception as e:
        print(f"[!] Database query error: {e}")
        return

    # Filter out items already indexed in Qdrant
    unindexed_listings = [item for item in all_listings if item.listing_id not in already_indexed_ids]
    
    print(f"[+] Total SQL Listings Fetched: {len(all_listings)} | Remaining to Embed: {len(unindexed_listings)}")

    if not unindexed_listings:
        print("[SUCCESS] All database listings are already fully indexed in Qdrant!")
        return

    # 1. Initialize FREE Local Embeddings (BAAI/bge-small-en-v1.5)
    print("[+] Loading local BAAI/bge-small-en-v1.5 embeddings model (0 cost, 0 quota limits)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 2. Format remaining SQL Listing rows into LangChain Document objects
    documents = []
    for item in unindexed_listings:
        title = item.listing_title or "Product Listing"
        price = float(item.effective_price or 0.0)
        stock = int(item.listed_quantity or 0)
        status_val = item.status or "ACTIVE"

        page_content = f"Title: {title}\nPrice: {price} PKR\nStock Quantity: {stock}\nStatus: {status_val}"
        
        metadata = {
            "product_id": item.listing_id,
            "title": title,
            "price_pkr": price,
            "stock_quantity": stock,
            "in_stock": stock > 0
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"[+] Collection '{COLLECTION_NAME}' does not exist. Creating 384-dim COSINE collection...")
        from qdrant_client.http import models as qmodels
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE)
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    total_batches = (len(documents) + batch_size - 1) // batch_size

    for idx, i in enumerate(range(0, len(documents), batch_size)):
        batch = documents[i : i + batch_size]
        current_batch_num = idx + 1
        print(f"    -> [Batch {current_batch_num}/{total_batches}] Indexing items {i+1} to {min(i+batch_size, len(documents))}...", flush=True)
        vector_store.add_documents(batch)

    print(f"\n[SUCCESS] Catalog embedding complete! Total vectors in Qdrant: {len(get_existing_product_ids(client, COLLECTION_NAME))}")

if __name__ == "__main__":
    # Embeds ALL listings using free local embeddings
    embed_catalog_from_db(limit_count=None, batch_size=100)
