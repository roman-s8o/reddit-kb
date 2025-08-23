#!/usr/bin/env python3
"""
Script to view documents from FAISS database
"""
import pickle
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append('backend')

from backend.utils.faiss_manager import faiss_manager

def format_timestamp(timestamp):
    """Convert Unix timestamp to readable date."""
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return "Unknown"

def clean_content(content):
    """Clean content by removing duplicate title."""
    # Remove "Title: ... Content:" pattern
    if content.startswith("Title:") and "Content:" in content:
        # Find where "Content:" starts
        content_start = content.find("Content:") + len("Content:")
        return content[content_start:].strip()
    return content

def view_documents():
    """View all documents in the database."""
    try:
        # Direct paths to files
        docs_path = "data/vector_db/documents.pkl"
        metadata_path = "data/vector_db/metadata.pkl"
        
        if not os.path.exists(docs_path):
            print("❌ Documents file not found!")
            return
            
        if not os.path.exists(metadata_path):
            print("❌ Metadata file not found!")
            return
        
        # Load documents and metadata
        with open(docs_path, 'rb') as f:
            documents = pickle.load(f)
            
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"📚 Found {len(documents)} documents in database")
        print("=" * 80)
        
        # Display documents
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            print(f"\n📄 Document {i+1}/{len(documents)}")
            print(f"Subreddit: r/{meta.get('subreddit', 'unknown')}")
            print(f"Type: {meta.get('type', 'unknown')}")
            print(f"Title: {meta.get('title', 'No title')}")
            print(f"Author: {meta.get('author', 'unknown')}")
            print(f"Score: {meta.get('score', 'N/A')}")
            print(f"Created: {format_timestamp(meta.get('created_utc', 'unknown'))}")
            print("-" * 40)
            print("Content:")
            
            # Clean and display content
            clean_doc = clean_content(doc)
            print(clean_doc)
            print("=" * 80)
            
            # Ask user if they want to continue
            if i > 0 and i % 5 == 0:
                response = input(f"\nПоказано {i+1} документов. Продолжить? (y/n): ")
                if response.lower() != 'y':
                    break
        
        print(f"\n✅ Просмотрено {min(i+1, len(documents))} документов из {len(documents)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    view_documents()
