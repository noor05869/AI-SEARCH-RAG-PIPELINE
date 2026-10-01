import sys
import pathlib
from sqlmodel import Session, select

# Add src to python path for imports
sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))

from fastapi_blog.database import engine, create_db_and_tables
from fastapi_blog.models import Product

SAMPLE_PRODUCTS = [
    {
        "title": "Samsung Galaxy S24 Ultra 5G",
        "description": "Flagship smartphone with 200MP camera, Snapdragon 8 Gen 3, 12GB RAM, 256GB storage, Titanium Gray finish, S-Pen included.",
        "category": "mobile",
        "brand": "Samsung",
        "price_pkr": 385000.0,
        "stock_quantity": 15,
        "is_active": True
    },
    {
        "title": "Samsung Galaxy A55 5G",
        "description": "Mid-range killer smartphone with 6.6-inch Super AMOLED 120Hz display, Exynos 1480, 8GB RAM, 256GB storage, 5000mAh battery.",
        "category": "mobile",
        "brand": "Samsung",
        "price_pkr": 134999.0,
        "stock_quantity": 25,
        "is_active": True
    },
    {
        "title": "Apple iPhone 15 Pro Max 256GB",
        "description": "Titanium design, A17 Pro chip, Action Button, 48MP main camera with 5x Telephoto optical zoom, USB-C port, Natural Titanium color.",
        "category": "mobile",
        "brand": "Apple",
        "price_pkr": 495000.0,
        "stock_quantity": 8,
        "is_active": True
    },
    {
        "title": "Soundpeats Engine 4 Wireless Earbuds",
        "description": "Coaxial dual dynamic drivers, LDAC audio codec support, 43 hours total playtime, dual device connection, low latency gaming mode.",
        "category": "earbuds",
        "brand": "Soundpeats",
        "price_pkr": 12500.0,
        "stock_quantity": 50,
        "is_active": True
    },
    {
        "title": "Soundpeats Air4 Pro Noise Cancelling Earbuds",
        "description": "Adaptive ANC noise cancellation, Snapdragon Sound with aptX Lossless audio, 6 microphones for clear calls, Bluetooth 5.3.",
        "category": "earbuds",
        "brand": "Soundpeats",
        "price_pkr": 16500.0,
        "stock_quantity": 30,
        "is_active": True
    },
    {
        "title": "Dell XPS 15 9530 Core i7 Laptop",
        "description": "13th Gen Intel Core i7-13700H, 16GB DDR5 RAM, 1TB NVMe SSD, NVIDIA GeForce RTX 4060 8GB VRAM, 15.6-inch FHD+ display, backlit keyboard.",
        "category": "laptop",
        "brand": "Dell",
        "price_pkr": 520000.0,
        "stock_quantity": 5,
        "is_active": True
    },
    {
        "title": "Dell Inspiron 15 3520 Core i5",
        "description": "12th Gen Intel Core i5-1235U, 8GB RAM, 512GB SSD, 15.6-inch 120Hz FHD display, Intel Iris Xe graphics, Windows 11 Home.",
        "category": "laptop",
        "brand": "Dell",
        "price_pkr": 155000.0,
        "stock_quantity": 12,
        "is_active": True
    },
    {
        "title": "Redmi Note 13 Pro+ 5G",
        "description": "200MP OIS camera, 120W HyperCharge fast charging, Dimensity 7200-Ultra, 12GB RAM, 512GB storage, IP68 water resistance.",
        "category": "mobile",
        "brand": "Redmi",
        "price_pkr": 139999.0,
        "stock_quantity": 20,
        "is_active": True
    },
    {
        "title": "Anker 737 Power Bank (PowerCore 24K)",
        "description": "24000mAh 3-port portable charger with 140W output, smart digital display, ultra-fast charging for laptops and smartphones.",
        "category": "accessories",
        "brand": "Anker",
        "price_pkr": 34500.0,
        "stock_quantity": 18,
        "is_active": True
    }
]

def seed_database():
    print("[+] Initializing database tables...")
    create_db_and_tables()

    with Session(engine) as session:
        # Check existing count
        existing_products = session.exec(select(Product)).all()
        if existing_products:
            print(f"[*] Database already contains {len(existing_products)} products. Skipping seed.")
            return

        print(f"[+] Seeding {len(SAMPLE_PRODUCTS)} products into PostgreSQL database...")
        for item in SAMPLE_PRODUCTS:
            product = Product(**item)
            session.add(product)
        
        session.commit()
        print("[SUCCESS] Database seeded successfully!")

if __name__ == "__main__":
    seed_database()
