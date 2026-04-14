#!/bin/sh
set -e

# Define the expected path (matches your Dockerfile ENV)
DB_FILE="./data/shop.db"

echo "[entrypoint] Checking for database at $DB_FILE..."

if [ ! -f "$DB_FILE" ]; then
    echo "[entrypoint] DB not found — Initializing and Seeding..."
    # Ensure the directory exists so SQLite doesn't throw an 'Unable to open' error
    mkdir -p "$(dirname "$DB_FILE")"
    python seed_db.py
    echo "[entrypoint] Seeding complete."
else
    echo "[entrypoint] DB exists — Skipping seed to save time."
fi

echo "[entrypoint] Starting Maison Elara Server on Port ${PORT:-8080}..."

# We use 'exec' so the Python process receives termination signals from Azure
# We call 'python server.py' directly because your file already has the uvicorn.run() logic
exec python server.py