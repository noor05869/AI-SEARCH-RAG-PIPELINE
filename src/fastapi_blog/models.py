from datetime import datetime, timezone
from sqlmodel import SQLModel, Field

class Listing(SQLModel, table=True):
    """Mapped directly to TechVault's real 'listings' table in MySQL."""
    __tablename__ = "listings"

    listing_id: int = Field(primary_key=True)
    listing_title: str = Field(index=True)
    listed_quantity: int = Field(default=0)
    effective_price: float | None = Field(default=0.0)
    status: str = Field(default="ACTIVE")
    is_deleted: bool = Field(default=False)

# Backward-compatibility alias
Product = Listing
