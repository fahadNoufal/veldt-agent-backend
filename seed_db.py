"""
seed_db.py  —  one-time database seeder for Maison Elara.

Scans IMAGE_BASE_PATH for all img_XXXX.png files and inserts a Product
row for each one. Run this ONCE before starting the server.

Usage
-----
    python seed_db.py                          # uses defaults from .env
    python seed_db.py --images ./website/images --db ./shop.db
    python seed_db.py --csv products.csv       # if you have real product data

CSV format (optional, takes priority over auto-generation):
    id,name,brand,price,rating,image_path
    0082,Silk Wrap Dress,Zimmermann,189.99,5,images/img_0082.png
"""

import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


# ── DB setup (mirrors server.py exactly) ──────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./shop.db")
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class Product(Base):
    __tablename__ = "products"
    id         = Column(String,  primary_key=True)
    name       = Column(String,  nullable=False)
    brand      = Column(String,  nullable=False)
    price      = Column(Float,   nullable=False)
    rating     = Column(Integer, nullable=False)
    quantity   = Column(Integer, default=20)
    image_path = Column(String,  nullable=False)


Base.metadata.create_all(bind=engine)


# ── Realistic dress product data for auto-generation ──────────────────────────

ADJECTIVES = [
    "Silk", "Chiffon", "Linen", "Velvet", "Satin", "Floral", "Ruched",
    "Draped", "Pleated", "Broderie", "Georgette", "Jersey", "Crepe",
    "Metallic", "Lace", "Sequined", "Ribbed", "Tiered", "Smocked", "Wrap",
]

STYLES = [
    "Midi Dress", "Maxi Dress", "Mini Dress", "A-Line Dress", "Bodycon Dress",
    "Slip Dress", "Wrap Dress", "Shift Dress", "Shirt Dress", "Cami Dress",
    "Sundress", "Blazer Dress", "Corset Dress", "Halter Dress", "Off-Shoulder Dress",
    "One-Shoulder Dress", "Backless Dress", "Strapless Dress", "Column Dress",
    "Fit & Flare Dress",
]

BRANDS = [
    "Réalisation Par", "Reformation", "Zimmermann", "Ba&sh", "Rotate",
    "Self-Portrait", "Faithfull the Brand", "Aje", "Alice McCall",
    "Cult Gaia", "Staud", "Nanushka", "Ganni", "Sandro", "Maje",
    "Alexis", "Likely", "Shoshanna", "Tanya Taylor", "Veronica Beard",
]

PRICE_BANDS = [
    (999,  89.99),   # budget
    (99.99,  149.99),  # mid
    (159.99, 249.99),  # premium
    (259.99, 399.99),  # luxury
]


def generate_product(pid: str, seed: int) -> dict:
    """Deterministic pseudo-random product data for a given image ID."""
    rng = random.Random(seed)
    adj    = rng.choice(ADJECTIVES)
    style  = rng.choice(STYLES)
    brand  = rng.choice(BRANDS)
    lo, hi = rng.choice(PRICE_BANDS)
    price  = round(rng.uniform(lo, hi), 2)
    rating = rng.choices([3, 4, 5], weights=[10, 35, 55])[0]
    return {
        "id":         pid,
        "name":       f"{adj} {style}",
        "brand":      brand,
        "price":      price,
        "rating":     rating,
        "quantity":   rng.randint(10, 30),
        "image_path": f"images/img_{pid}.png",
    }


# ── CSV loader ────────────────────────────────────────────────────────────────

def load_from_csv(csv_path: str) -> list[dict]:
    products = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append({
                "id":         row["id"].strip().zfill(4),
                "name":       row["name"].strip(),
                "brand":      row["brand"].strip(),
                "price":      float(row["price"]),
                "rating":     int(row["rating"]),
                "quantity":   int(row.get("quantity", 20)),
                "image_path": row.get("image_path", f"images/img_{row['id'].strip().zfill(4)}.png"),
            })
    return products


# ── Image scanner ──────────────────────────────────────────────────────────────

def scan_images(images_dir: str) -> list[str]:
    """Return sorted list of zero-padded IDs found in images_dir."""
    pattern = re.compile(r"img_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
    ids = []
    for fname in os.listdir(images_dir):
        m = pattern.match(fname)
        if m:
            ids.append(f"{int(m.group(1)):04d}")
    return sorted(set(ids))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed Maison Elara product database")
    parser.add_argument("--images",  default=os.getenv("IMAGE_BASE_PATH", "./website/images"),
                        help="Path to product images folder")
    parser.add_argument("--db",      default=DATABASE_URL,
                        help="SQLAlchemy database URL")
    parser.add_argument("--csv",     default=None,
                        help="Optional CSV with real product data")
    parser.add_argument("--clear",   action="store_true",
                        help="Delete all existing products before seeding")
    args = parser.parse_args()

    db = SessionLocal()

    # ── Optional clear ─────────────────────────────────────────
    if args.clear:
        deleted = db.query(Product).delete()
        db.commit()
        print(f"Cleared {deleted} existing products.")

    # ── Count existing ─────────────────────────────────────────
    existing_ids = {p.id for p in db.query(Product.id).all()}
    print(f"Existing products in DB: {len(existing_ids)}")

    # ── Build product list ─────────────────────────────────────
    if args.csv:
        print(f"Loading from CSV: {args.csv}")
        products = load_from_csv(args.csv)
    else:
        if not os.path.isdir(args.images):
            print(f"ERROR: images directory not found: '{args.images}'")
            print("Set IMAGE_BASE_PATH in .env or pass --images <path>")
            sys.exit(1)
        image_ids = scan_images(args.images)
        if not image_ids:
            print(f"ERROR: no img_XXXX.png files found in '{args.images}'")
            sys.exit(1)
        print(f"Found {len(image_ids)} images in '{args.images}'")
        products = [generate_product(pid, seed=int(pid)) for pid in image_ids]

    # ── Insert only missing rows ───────────────────────────────
    inserted = 0
    skipped  = 0
    for p in products:
        if p["id"] in existing_ids:
            skipped += 1
            continue
        db.add(Product(**p))
        inserted += 1

    db.commit()
    db.close()

    total = len(existing_ids) + inserted
    print(f"\nDone.")
    print(f"  Inserted : {inserted}")
    print(f"  Skipped  : {skipped} (already existed)")
    print(f"  Total    : {total} products in DB")

    if inserted > 0:
        print(f"\nSample rows:")
        db2 = SessionLocal()
        for p in db2.query(Product).limit(4).all():
            print(f"  {p.id}  {p.name:<32}  {p.brand:<22}  £{p.price:.2f}  ★{p.rating}")
        db2.close()


if __name__ == "__main__":
    main()