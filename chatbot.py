"""
RAG-powered chatbot for Green Level High School counseling.
Handles vector store initialization, document retrieval, and response generation.
"""
import os
import json
import logging
import pathlib
from typing import List, Tuple
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader

# System prompt template
SYSTEM_PROMPT = """
You are Green Level AI Counselor, a virtual academic advisor.
You help students plan courses, explore electives, and understand requirements.
Never provide personal or emotional advice—redirect to a school counselor if needed.
"""

# Mental health keywords that should trigger counselor referral
MENTAL_HEALTH_KEYWORDS = [
    "suicide", "self-harm", "depression", "anxiety crisis",
    "cutting", "ending my life", "want to die", "kill myself",
    "hopeless", "helpless", "can't go on"
]

logger = logging.getLogger(__name__)


class GLHSChatbot:
    """Main chatbot class with RAG capabilities."""
    
    def __init__(
        self,
        data_path: str = "./data/glhs_info.json",
        pdf_dir: str = "./data/pdf_docs",
        persist_dir: str = "./chroma_db",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_model: str = "text-embedding-ada-002",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
        self.data_path = data_path
        self.pdf_dir = pdf_dir
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        
        # Initialize or load vectorstore
        self.vectorstore = self._initialize_vectorstore()
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=self.llm_model, temperature=self.temperature)
    
    def _json_to_documents(self, data: dict) -> List[Document]:
        """Convert JSON data structure to LangChain Documents."""
        documents = []
        
        # Convert graduation requirements
        if "graduation_requirements" in data:
            req_text = "GRADUATION REQUIREMENTS:\n"
            req_text += f"Total credits needed: {data['graduation_requirements'].get('total_credits', 'N/A')}\n"
            req_text += f"English credits: {data['graduation_requirements'].get('english_credits', 'N/A')}\n"
            req_text += f"Math credits: {data['graduation_requirements'].get('math_credits', 'N/A')}\n"
            req_text += f"Science credits: {data['graduation_requirements'].get('science_credits', 'N/A')}\n"
            req_text += f"Social Studies credits: {data['graduation_requirements'].get('social_studies_credits', 'N/A')}\n"
            req_text += f"Foreign Language credits: {data['graduation_requirements'].get('foreign_language_credits', 'N/A')}\n"
            req_text += f"Physical Education credits: {data['graduation_requirements'].get('physical_education_credits', 'N/A')}\n"
            req_text += f"Health credits: {data['graduation_requirements'].get('health_credits', 'N/A')}\n"
            req_text += f"Elective credits: {data['graduation_requirements'].get('elective_credits', 'N/A')}\n"
            req_text += f"Minimum GPA: {data['graduation_requirements'].get('minimum_gpa', 'N/A')}\n"
            req_text += f"Community service hours: {data['graduation_requirements'].get('community_service_hours', 'N/A')}\n"
            req_text += f"\n{data['graduation_requirements'].get('description', '')}"
            documents.append(Document(page_content=req_text, metadata={"category": "graduation_requirements"}))
        
        # Convert courses
        if "courses" in data:
            for course in data["courses"]:
                course_text = f"COURSE: {course.get('name', 'Unknown')} ({course.get('code', 'N/A')})\n"
                course_text += f"Credits: {course.get('credits', 'N/A')}\n"
                course_text += f"Category: {course.get('category', 'N/A')}\n"
                course_text += f"Description: {course.get('description', 'No description available.')}\n"
                if course.get('prerequisites'):
                    course_text += f"Prerequisites: {', '.join(course['prerequisites'])}\n"
                documents.append(Document(page_content=course_text, metadata={"category": "courses", "code": course.get('code', '')}))
        
        # Convert counselors
        if "counselors" in data:
            counselors_text = "SCHOOL COUNSELORS:\n"
            for counselor in data["counselors"]:
                counselors_text += f"\n{counselor.get('name', 'Unknown')}\n"
                counselors_text += f"Email: {counselor.get('email', 'N/A')}\n"
                counselors_text += f"Phone: {counselor.get('phone', 'N/A')}\n"
                counselors_text += f"Specialization: {counselor.get('specialization', 'N/A')}\n"
                counselors_text += f"Office: {counselor.get('office', 'N/A')}\n"
            documents.append(Document(page_content=counselors_text, metadata={"category": "counselors"}))
        
        # Convert policies
        if "policies" in data:
            policies_text = "SCHOOL POLICIES:\n"
            for policy_type, policy_data in data["policies"].items():
                policies_text += f"\n{policy_type.upper()} POLICIES:\n"
                for key, value in policy_data.items():
                    policies_text += f"{key.replace('_', ' ').title()}: {value}\n"
            documents.append(Document(page_content=policies_text, metadata={"category": "policies"}))
        
        # Convert electives
        if "electives" in data:
            electives_text = "AVAILABLE ELECTIVES:\n"
            for category, elective_list in data["electives"].items():
                electives_text += f"\n{category.replace('_', ' ').title()}:\n"
                electives_text += ", ".join(elective_list) + "\n"
            documents.append(Document(page_content=electives_text, metadata={"category": "electives"}))
        
        # Convert college prep info
        if "college_prep" in data:
            college_text = "COLLEGE PREPARATION INFORMATION:\n"
            
            if "ap_vs_honors" in data["college_prep"]:
                ap_info = data["college_prep"]["ap_vs_honors"]
                college_text += "\nAP vs Honors Courses:\n"
                college_text += f"AP Courses: {ap_info.get('ap_courses', 'N/A')}\n"
                college_text += f"Honors Courses: {ap_info.get('honors_courses', 'N/A')}\n"
                college_text += f"When to Take: {ap_info.get('when_to_take', 'N/A')}\n"
                college_text += f"College Credit: {ap_info.get('college_credit', 'N/A')}\n"
            
            if "tips" in data["college_prep"]:
                college_text += "\nCollege Prep Tips:\n"
                for i, tip in enumerate(data["college_prep"]["tips"], 1):
                    college_text += f"{i}. {tip}\n"
            
            if "testing" in data["college_prep"]:
                testing = data["college_prep"]["testing"]
                college_text += "\nTesting Information:\n"
                college_text += f"SAT: {testing.get('sat', 'N/A')}\n"
                college_text += f"ACT: {testing.get('act', 'N/A')}\n"
                college_text += f"Preparation: {testing.get('prep', 'N/A')}\n"
                college_text += f"When to Take: {testing.get('when', 'N/A')}\n"
            
            documents.append(Document(page_content=college_text, metadata={"category": "college_prep"}))
        
        return documents
    
    def _load_pdf_documents(self) -> List[Document]:
        """Load PDF documents from pdf_dir directory."""
        documents = []
        
        if not os.path.exists(self.pdf_dir):
            logger.info(f"PDF directory '{self.pdf_dir}' not found. Skipping PDF loading.")
            return documents
        
        pdf_files = list(pathlib.Path(self.pdf_dir).glob("*.pdf"))
        
        if not pdf_files:
            logger.info(f"No PDF files found in '{self.pdf_dir}'. Continuing with JSON data only.")
            return documents
        
        logger.info(f"Loading {len(pdf_files)} PDF file(s) from '{self.pdf_dir}'...")
        
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                file_title = pathlib.Path(pdf_file).stem
                
                # Add metadata to each document page
                for doc in pdf_docs:
                    doc.metadata['file_title'] = file_title
                    doc.metadata['source_type'] = 'pdf'
                    doc.metadata['source_file'] = pdf_file.name
                
                documents.extend(pdf_docs)
                logger.info(f"✅ Loaded {len(pdf_docs)} pages from '{pdf_file.name}'")
            except Exception as e:
                logger.error(f"❌ Error loading PDF '{pdf_file}': {e}")
                continue
        
        return documents
    
    def _initialize_vectorstore(self) -> Chroma:
        """Initialize or load the Chroma vectorstore from both JSON and PDF sources."""
        # Check if vectorstore already exists
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            logger.info(f"Loading existing vectorstore from {self.persist_dir}")
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
        
        # Create new vectorstore from JSON + PDF data (hybrid approach)
        logger.info("Creating new vectorstore from JSON data and PDFs (hybrid approach)...")
        all_documents = []
        
        # 1. Load JSON documents
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                json_docs = self._json_to_documents(data)
                
                # Mark JSON documents with metadata
                for doc in json_docs:
                    doc.metadata['source_type'] = 'json'
                
                all_documents.extend(json_docs)
                logger.info(f"✅ Loaded {len(json_docs)} documents from JSON")
            except Exception as e:
                logger.error(f"❌ Error loading JSON file: {e}")
        else:
            logger.warning(f"⚠️ JSON file '{self.data_path}' not found. Continuing with PDFs only.")
        
        # 2. Load PDF documents
        pdf_docs = self._load_pdf_documents()
        all_documents.extend(pdf_docs)
        
        # Validate we have at least some documents
        if not all_documents:
            raise ValueError(
                "No documents found! Please add JSON data file or PDF files.\n"
                f"Expected JSON at: {self.data_path}\n"
                f"Expected PDFs at: {self.pdf_dir}"
            )
        
        # Split all documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunked_docs = text_splitter.split_documents(all_documents)
        
        logger.info(f"📊 Total chunks created: {len(chunked_docs)}")
        logger.info(f"   - From JSON: {len([d for d in chunked_docs if d.metadata.get('source_type') == 'json'])}")
        logger.info(f"   - From PDFs: {len([d for d in chunked_docs if d.metadata.get('source_type') == 'pdf'])}")
        
        # Create and persist vectorstore
        vectorstore = Chroma.from_documents(
            chunked_docs,
            self.embeddings,
            persist_directory=self.persist_dir
        )
        vectorstore.persist()
        logger.info(f"✅ Vectorstore created and persisted to {self.persist_dir}")
        
        return vectorstore
    
    def _check_mental_health(self, question: str) -> bool:
        """Check if question contains mental health crisis keywords."""
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in MENTAL_HEALTH_KEYWORDS)
    
    def query_with_rag(
        self,
        question: str,
        conversation_history: List[dict] = None,
        top_k: int = 5
    ) -> str:
        """
        Query the chatbot with RAG. Returns response string.
        """
        if not question.strip():
            return "Please ask me a question about academics, courses, or graduation requirements!"
        
        # Check for mental health concerns
        if self._check_mental_health(question):
            return (
                "I'm here to help with academic questions, but for personal mental health concerns, "
                "please reach out to a real counselor immediately. You can contact:\n\n"
                "- Ms. Sarah Johnson: sarah.johnson@glhs.edu, (555) 123-4567\n"
                "- Mr. Michael Chen: michael.chen@glhs.edu, (555) 123-4568\n"
                "- Ms. Emily Rodriguez: emily.rodriguez@glhs.edu, (555) 123-4569\n\n"
                "If this is an emergency, please contact 988 (Suicide & Crisis Lifeline) or 911."
            )
        
        # Retrieve relevant documents
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Build conversation history string
        history_text = ""
        if conversation_history:
            history_text = "Previous conversation:\n"
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                history_text += f"{role.capitalize()}: {content}\n"
            history_text += "\n"
        
        # Build prompt
        user_message_content = f"{history_text}Relevant Information:\n{context}\n\nQuestion: {question}\n\nPlease provide a helpful, friendly answer based on the information above. If the information doesn't fully answer the question, say so and suggest they speak with a counselor."
        
        # Generate response
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message_content)
        ]
        
        try:
            response = self.llm(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error. Please try again or contact a counselor for assistance."


# Global chatbot instance (initialized on first use)
_chatbot_instance = None


def get_chatbot() -> GLHSChatbot:
    """Get or create the global chatbot instance."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = GLHSChatbot()
    return _chatbot_instance

