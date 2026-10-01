import os
import sys
import logging
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Silence internal google.genai SDK logger warnings
logging.getLogger("google.genai").setLevel(logging.ERROR)

# Ensure stdout uses UTF-8 encoding on Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load GOOGLE_API_KEY from .env file
load_dotenv()

# 1. Define the Structured Output Schema using Pydantic
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
        description="Cleaned semantic query keyword to pass to Qdrant vector search"
    )

# Alias
TechBazaarSearchFilter = TechVaultSearchFilter

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[!] GOOGLE_API_KEY is missing in your .env file!")
        print("    Please add GOOGLE_API_KEY=your_key in d:\\learning\\fastapi_blog\\.env\n")
        return

    # 2. Initialize active Gemini Flash Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0
    )

    # 3. Attach Pydantic Schema using native 'json_schema' method
    extractor_chain = llm.with_structured_output(TechVaultSearchFilter, method="json_schema")

    # 4. Run test queries
    test_queries = [
        "I want a Samsung phone under 80000 PKR",
        "Show me budget wireless earbuds from Soundpeats below 10k",
        "Looking for Dell core i7 laptops"
    ]

    print("[+] Running Gemini Intent Extractor with LangChain + Pydantic...\n")

    for query in test_queries:
        print(f"[*] Input Query: '{query}'")
        try:
            result: TechVaultSearchFilter = extractor_chain.invoke(query)
            print("[-] Extracted Pydantic Result:")
            print(f"   • Category:     {result.category}")
            print(f"   • Brand:        {result.brand}")
            print(f"   • Max Price:    {result.max_price_pkr} PKR")
            print(f"   • In Stock:     {result.in_stock_only}")
            print(f"   • Vector Query: '{result.search_query}'\n")
        except Exception as e:
            print(f"[ERROR] {e}\n")

if __name__ == "__main__":
    main()
