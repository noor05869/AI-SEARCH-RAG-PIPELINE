import os
import logging
from fastapi import FastAPI, HTTPException, status
from dotenv import load_dotenv

# Silence internal google.genai SDK logger warnings
logging.getLogger("google.genai").setLevel(logging.ERROR)

from fastapi_blog.schemas import (
    postBase, postCreate, postResponse,
    ChatRequest, ChatResponse, TechVaultSearchFilter,
    StructuredChatResponse, StructuredChatReply
)
from fastapi_blog.rag_chain import (
    extract_intent, retrieve_products, build_rag_chain, build_structured_rag_chain,
    parse_scratchpad
)

load_dotenv()

app = FastAPI(
    title="TechVault AI & FastAPI Portal",
    description="FastAPI application powering SQL DB, Qdrant RAG, and Gemini AI Search",
    version="0.3.0"
)

# Mock Blog Database
posts = {
    1: {
        "id": 1,
        "title": "Getting Started with FastAPI",
        "content": "FastAPI is a modern, fast web framework for building APIs with Python.",
        "author": "Admin",
        "published": True,
        "date_posted": "2026-09-22"
    }
}

@app.get('/')
def read_root():
    return {
        "message": "TechVault AI & FastAPI Portal is Running",
        "docs_url": "/docs",
        "status": "online"
    }

# --- Blog Endpoints ---
@app.get('/api/posts', response_model=dict)
def get_posts():
    return posts

@app.get('/api/posts/{id}')
def get_post(id: int):
    if id not in posts:
        raise HTTPException(status_code=404, detail="Post not found")
    return posts.get(id)

# --- AI Search & LangChain Endpoints ---
@app.post('/api/ai-search/extract-intent', response_model=TechVaultSearchFilter)
def extract_search_intent(query: str):
    """Extract structured Pydantic search parameters using Gemini."""
    try:
        filters = extract_intent(query)
        return filters
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intent extraction failed: {str(e)}"
        )

@app.post('/api/ai-search/chat', response_model=ChatResponse)
def ai_search_chat(req: ChatRequest):
    """Conversational text search endpoint using Qdrant Vector Retrieval + Gemini."""
    try:
        extracted_filters = extract_intent(req.message)
        retrieved_docs = retrieve_products(extracted_filters)
        
        if retrieved_docs:
            context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        else:
            context = "No direct matching products found in current inventory."
        
        chain = build_rag_chain()
        raw_reply = chain.invoke({
            "context": context,
            "input": req.message
        })
        thinking, clean_reply = parse_scratchpad(raw_reply)
        
        return ChatResponse(
            session_id=req.session_id,
            reply=clean_reply,
            thinking=thinking or None,
            extracted_filters=extracted_filters
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI Search error: {str(e)}"
        )

@app.post('/api/ai-search/chat-structured', response_model=StructuredChatResponse)
def ai_search_chat_structured(req: ChatRequest):
    """Structured JSON RAG endpoint returning typed product objects in response JSON."""
    try:
        extracted_filters = extract_intent(req.message)
        retrieved_docs = retrieve_products(extracted_filters)
        
        if retrieved_docs:
            context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        else:
            context = "No direct matching products found in current inventory."
        
        structured_chain = build_structured_rag_chain()
        reply_data: StructuredChatReply = structured_chain.invoke({
            "context": context,
            "input": req.message
        })
        
        return StructuredChatResponse(
            session_id=req.session_id,
            reply_data=reply_data,
            extracted_filters=extracted_filters
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Structured AI Search error: {str(e)}"
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
