"""
Script to build the ChromaDB vector database from JSON files.
Run this script to initialize or rebuild the vector database.

Usage:
    python build_vector_db.py
"""
import os
import json
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-ada-002"


def load_json_files(data_dir: str) -> List[Document]:
    """Load all JSON files from the data directory and convert to documents."""
    documents = []
    json_files = [
        "glhs_info.json",
        "glhs_course_catalog.json",
        "glhs_graduation_requirments.json",
        "glhs_clubs.json",
        "glhs_college_pathways.json",
        "glhs_college_filters.json",
        "academic_difficulty_profile.json",
        "honors_difficulty_profile.json",
        "ap_difficulty_profile.json",
        "majors.ljson",
        "oppurtunities_database.json",
        "class_requirements.json",
        "application_glossary.json",
        "system_metadata.json"
    ]
    
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Convert JSON to text representation
                # For structured data, create a readable text format
                if isinstance(data, dict):
                    text_content = json.dumps(data, indent=2, ensure_ascii=False)
                elif isinstance(data, list):
                    text_content = json.dumps(data, indent=2, ensure_ascii=False)
                else:
                    text_content = str(data)
                
                # Create document with metadata
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "source": json_file,
                        "type": "json",
                        "file": json_file
                    }
                )
                documents.append(doc)
                print(f"✓ Loaded JSON: {json_file}")
            except Exception as e:
                print(f"✗ Error loading {json_file}: {e}")
    
    return documents


def build_vector_database():
    """Build the ChromaDB vector database from JSON files."""
    # Get paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    persist_directory = os.path.join(base_dir, "chroma_db")
    
    print("=" * 60)
    print("Building ChromaDB Vector Database")
    print("=" * 60)
    print()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        return
    
    # Initialize embeddings
    print("Initializing embeddings...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Load documents
    print("\nLoading documents...")
    all_documents = []
    
    # Load JSON files
    json_docs = load_json_files(data_dir)
    all_documents.extend(json_docs)
    print(f"Loaded {len(json_docs)} JSON document(s)")
    
    if not all_documents:
        print("\nERROR: No documents found to load!")
        print("Please ensure JSON files exist in the data/ directory")
        return
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    
    # Split documents into chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create or update vector store
    print("\nCreating vector store...")
    
    # Remove existing database if it exists
    if os.path.exists(persist_directory):
        import shutil
        print(f"Removing existing database at {persist_directory}...")
        shutil.rmtree(persist_directory)
    
    # Create new vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"\n✓ Vector database created successfully!")
    print(f"  Location: {persist_directory}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print()
    print("=" * 60)
    print("Database build complete!")
    print("=" * 60)


if __name__ == "__main__":
    build_vector_database()

