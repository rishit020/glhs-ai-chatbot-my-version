"""
Script to build the ChromaDB vector database from JSON files and PDFs.
Run this script to initialize or rebuild the vector database.

Usage:
    python build_vector_db.py
"""
import os
import json
import re
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pypdf not installed. PDF processing disabled.")

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
        "clubs.json",
        "glhs_college_pathways.json",
        "glhs_college_filters.json",
        "academic_difficulty_profile.json",
        "honors_difficulty_profile.json",
        "ap_difficulty_profile.json",
        "majors.ljson",
        "oppurtunities_database.json",
        "class_requirements.json",
        "application_glossary.json",
        "system_metadata.json",
        "wcpss_planning_guide_glhs.json",
        "class_directory.json",
        "freshman_courses.json",
        "sophmore_courses.json",
        "junior_courses.json",
        "wake_tech.json"
    ]
    
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Special handling for clubs.json to create separate documents per club
                if json_file == "clubs.json" and isinstance(data, dict) and "clubs" in data:
                    # Create a separate document for each club to improve retrieval accuracy
                    for club in data.get("clubs", []):
                        # Format club information as readable text
                        club_text = f"Club Name: {club.get('name', 'N/A')}\n"
                        club_text += f"Category: {club.get('category', 'N/A')}\n"
                        if club.get('advisors'):
                            advisors = club['advisors'] if isinstance(club['advisors'], list) else [club['advisors']]
                            club_text += f"Advisors: {', '.join(advisors)}\n"
                        if club.get('student_contacts'):
                            contacts = club['student_contacts'] if isinstance(club['student_contacts'], list) else [club['student_contacts']]
                            club_text += f"Student Contacts: {', '.join(contacts)}\n"
                        if club.get('activities'):
                            club_text += f"Activities: {club.get('activities')}\n"
                        if club.get('meeting_day'):
                            club_text += f"Meeting Day: {club.get('meeting_day')}\n"
                        if club.get('location'):
                            club_text += f"Location: {club.get('location')}\n"
                        
                        doc = Document(
                            page_content=club_text,
                            metadata={
                                "source": json_file,
                                "type": "json",
                                "file": json_file,
                                "club_name": club.get('name', ''),
                                "category": club.get('category', '')
                            }
                        )
                        documents.append(doc)
                    print(f"✓ Loaded JSON: {json_file} ({len(data.get('clubs', []))} clubs as separate documents)")
                else:
                    # Convert JSON to text representation for other files
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


def is_glhs_relevant(text: str) -> bool:
    """
    Filter content to only include GLHS-relevant information.
    Excludes content specific to other schools or general WCPSS info not applicable to GLHS.
    """
    text_lower = text.lower()
    
    # GLHS-specific keywords (positive indicators)
    glhs_keywords = [
        "green level",
        "glhs",
        "green level high school",
        # Include general course information that applies to all schools
        "course",
        "graduation",
        "credit",
        "prerequisite",
        "honors",
        "ap ",
        "advanced placement",
        "gpa",
        "grade point average",
        "schedule",
        "registration",
        "elective",
        "required",
        "math",
        "science",
        "english",
        "social studies",
        "world language",
        "arts",
        "cte",
        "healthful living",
        "physical education"
    ]
    
    # School-specific keywords to exclude (negative indicators)
    exclude_keywords = [
        "apex high school",
        "cary high school",
        "garner high school",
        "holly springs high school",
        "leesville road high school",
        "middle creek high school",
        "panther creek high school",
        "wakefield high school",
        "wake forest high school",
        "enloe high school",
        "millbrook high school",
        "sanderson high school",
        "broughton high school",
        "athens drive high school",
        "fuquay-varina high school",
        "green hope high school",
        "heritage high school",
        "hillside high school",
        "knightdale high school",
        "rolesville high school",
        "southeast raleigh",
        "southeast raleigh high school",
        "wake early college",
        "wake stem",
        "wake young men's",
        "wake young women's"
    ]
    
    # Check for exclusion keywords first (higher priority)
    for exclude in exclude_keywords:
        if exclude in text_lower:
            return False
    
    # Check for GLHS or general keywords
    for keyword in glhs_keywords:
        if keyword in text_lower:
            return True
    
    # If no specific keywords found, include general educational content
    # that could be relevant (course descriptions, requirements, etc.)
    educational_keywords = [
        "requirement",
        "curriculum",
        "program",
        "pathway",
        "endorsement",
        "diploma"
    ]
    
    for keyword in educational_keywords:
        if keyword in text_lower:
            return True
    
    # Default: exclude if no clear relevance
    return False


def load_pdf_files(pdf_dir: str) -> List[Document]:
    """Load PDF files from the pdf_docs directory and filter for GLHS-relevant content."""
    if not PDF_SUPPORT:
        return []
    
    documents = []
    
    if not os.path.exists(pdf_dir):
        return documents
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_dir, pdf_file)
        try:
            print(f"Processing PDF: {pdf_file}...")
            reader = PdfReader(file_path)
            
            all_text = []
            total_pages = len(reader.pages)
            relevant_pages = 0
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        # Filter for GLHS-relevant content
                        if is_glhs_relevant(page_text):
                            all_text.append(page_text)
                            relevant_pages += 1
                        else:
                            # Still include if it's general course/graduation info
                            # but skip school-specific sections
                            if any(keyword in page_text.lower() for keyword in 
                                   ["course", "graduation", "credit", "requirement", 
                                    "curriculum", "program", "pathway"]):
                                all_text.append(page_text)
                                relevant_pages += 1
                except Exception as e:
                    print(f"  Warning: Could not extract text from page {page_num}: {e}")
                    continue
            
            if all_text:
                full_text = "\n\n".join(all_text)
                
                # Additional filtering: remove sections that are clearly not GLHS-specific
                # Split by common section headers and filter
                sections = re.split(r'\n{2,}', full_text)
                filtered_sections = []
                
                for section in sections:
                    if section.strip() and is_glhs_relevant(section):
                        filtered_sections.append(section)
                
                if filtered_sections:
                    final_text = "\n\n".join(filtered_sections)
                    
                    doc = Document(
                        page_content=final_text,
                        metadata={
                            "source": pdf_file,
                            "type": "pdf",
                            "file": pdf_file,
                            "total_pages": total_pages,
                            "relevant_pages": relevant_pages
                        }
                    )
                    documents.append(doc)
                    print(f"✓ Loaded PDF: {pdf_file} ({relevant_pages}/{total_pages} relevant pages)")
                else:
                    print(f"⚠ Skipped PDF: {pdf_file} (no GLHS-relevant content found)")
            else:
                print(f"⚠ Skipped PDF: {pdf_file} (no extractable text or no relevant content)")
                
        except Exception as e:
            print(f"✗ Error loading PDF {pdf_file}: {e}")
    
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
    
    # Load PDF files
    pdf_dir = os.path.join(data_dir, "pdf_docs")
    pdf_docs = load_pdf_files(pdf_dir)
    all_documents.extend(pdf_docs)
    if pdf_docs:
        print(f"Loaded {len(pdf_docs)} PDF document(s)")
    
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

