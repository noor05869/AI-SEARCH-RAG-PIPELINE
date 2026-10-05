from pydantic import BaseModel, ConfigDict, Field

# --- Blog Schemas ---
class postBase(BaseModel):
    title: str = Field(..., min_length=2, max_length=100)
    content: str = Field(..., min_length=2, max_length=1000)
    author: str = Field(..., min_length=2, max_length=100)
    published: bool = Field(default=True)

class postCreate(postBase):
    pass

class postResponse(postBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    date_posted: str

# --- TechVault AI Search & RAG Schemas ---
class TechVaultSearchFilter(BaseModel):
    category: str | None = Field(
        default=None, 
        description="Product category e.g., mobile, laptop, earbuds, smartwatch"
    )
    brand: str | None = Field(
        default=None, 
        description="Brand name if specified e.g., Samsung, Apple, Dell, Redmi"
    )
    max_price_pkr: float | None = Field(
        default=None, 
        description="Maximum budget/price filter in PKR"
    )
    in_stock_only: bool = Field(
        default=True, 
        description="True unless user explicitly asks for out-of-stock items"
    )
    search_query: str = Field(
        description="Cleaned semantic query keyword to pass to vector search"
    )

# Alias
TechBazaarSearchFilter = TechVaultSearchFilter

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique user session identifier")
    message: str = Field(..., description="User query / prompt")

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    thinking: str | None = Field(
        default=None, 
        description="Test-time compute security & reasoning scratchpad"
    )
    extracted_filters: TechVaultSearchFilter | None = None

# --- Structured JSON RAG Schemas (For Structured API Responses) ---
class StructuredProductItem(BaseModel):
    title: str = Field(description="Product listing title")
    price_pkr: float = Field(description="Price in PKR")
    stock_status: str = Field(description="Stock status e.g. In Stock or Out of Stock")

class StructuredChatReply(BaseModel):
    thinking: str = Field(
        default="",
        description="Internal test-time compute scratchpad: security audit against jailbreaks, catalog fact verification, and response planning"
    )
    message: str = Field(description="Conversational summary reply from AI Assistant")
    products: list[StructuredProductItem] = Field(default_factory=list, description="Structured array of recommended product objects")

class StructuredChatResponse(BaseModel):
    session_id: str
    reply_data: StructuredChatReply
    extracted_filters: TechVaultSearchFilter | None = None