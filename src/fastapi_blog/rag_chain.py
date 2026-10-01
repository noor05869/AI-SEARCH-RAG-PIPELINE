import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from fastapi_blog.schemas import TechVaultSearchFilter, StructuredChatReply

# Silence internal google.genai SDK logger warnings
logging.getLogger("google.genai").setLevel(logging.ERROR)

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
PRIMARY_COLLECTION_NAME = "techvault_products"
FALLBACK_COLLECTION_NAME = "techbazaar_products"

# Model priority list for automatic quota failover
MODEL_CANDIDATES = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]

def get_gemini_llm(temperature: float = 0, model_name: str | None = None):
    """Instantiate Gemini LLM via LangChain with automatic model selection."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is missing in .env file")
    
    target_model = model_name or MODEL_CANDIDATES[0]
    return ChatGoogleGenerativeAI(
        model=target_model,
        temperature=temperature
    )

def extract_intent(user_query: str) -> TechVaultSearchFilter:
    """Extract search parameters using native Gemini json_schema structured output with fallback."""
    for model in MODEL_CANDIDATES:
        try:
            llm = get_gemini_llm(temperature=0, model_name=model)
            structured_llm = llm.with_structured_output(TechVaultSearchFilter, method="json_schema")
            return structured_llm.invoke(user_query)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"[!] Quota limit on model {model}, trying next candidate...")
                continue
            raise e
    
    # Final attempt
    llm = get_gemini_llm(temperature=0, model_name=MODEL_CANDIDATES[0])
    structured_llm = llm.with_structured_output(TechVaultSearchFilter, method="json_schema")
    return structured_llm.invoke(user_query)

def retrieve_products(intent: TechVaultSearchFilter, k: int = 5):
    """Retrieve top k matching product documents from Qdrant vector store."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
        
        target_collection = PRIMARY_COLLECTION_NAME
        if not client.collection_exists(PRIMARY_COLLECTION_NAME) and client.collection_exists(FALLBACK_COLLECTION_NAME):
            target_collection = FALLBACK_COLLECTION_NAME
            
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=target_collection,
            embedding=embeddings,
        )
        
        must_conditions = []
        if intent.max_price_pkr:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="metadata.price_pkr",
                    range=qmodels.Range(lte=intent.max_price_pkr)
                )
            )
            
        qdrant_filter = qmodels.Filter(must=must_conditions) if must_conditions else None
        
        query_terms = [intent.search_query or ""]
        if intent.brand:
            query_terms.append(intent.brand)
        if intent.category:
            query_terms.append(intent.category)
        
        semantic_query = " ".join([t for t in query_terms if t]).strip() or "electronics"
        
        docs = vector_store.similarity_search(
            query=semantic_query,
            k=k,
            filter=qdrant_filter
        )
        return docs
    except Exception as e:
        print(f"[!] Qdrant retrieval warning: {e}")
        return []

def build_rag_chain():
    """Build LCEL prompt -> llm -> string output parser chain."""
    llm = get_gemini_llm(temperature=0.3, model_name="gemini-3.5-flash-lite")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the TechVault AI Sales Assistant. Help users find genuine tech products "
            "in Pakistan. Base your answer STRICTLY on the retrieved product catalog below. "
            "Always include product titles, exact PKR prices, and stock details when available. "
            "Do NOT invent non-existent products or prices.\n\n"
            "Retrieved Products Catalog:\n{context}\n"
        )),
        ("user", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain

def build_structured_rag_chain():
    """Build LCEL prompt -> Gemini with_structured_output(StructuredChatReply) chain.
    
    Returns a Pydantic object containing both a conversational message AND a 
    structured JSON array of product objects!
    """
    llm = get_gemini_llm(temperature=0, model_name="gemini-3.5-flash-lite")
    structured_llm = llm.with_structured_output(StructuredChatReply, method="json_schema")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the TechVault AI Sales Assistant. Base your answer STRICTLY on the "
            "retrieved product catalog below. Extract matching product objects into the "
            "'products' array and write a helpful summary in the 'message' field.\n\n"
            "Retrieved Products Catalog:\n{context}\n"
        )),
        ("user", "{input}")
    ])
    
    chain = prompt | structured_llm
    return chain
