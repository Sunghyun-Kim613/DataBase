#!/usr/bin/env python
"""Load CSV data into the database."""

from backend.load_data import load_csv_to_db

if __name__ == "__main__":
    print("Starting data load...")
    load_csv_to_db()
    print("Data load complete!")
