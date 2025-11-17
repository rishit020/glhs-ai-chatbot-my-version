"""
RAG-powered chatbot for Green Level High School counseling.
Handles vector store initialization, document retrieval, and response generation.
"""
import os
import json
import logging
import pathlib
from typing import List, Tuple, Any
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader

# System prompt template
SYSTEM_PROMPT = """
You are Green Level AI Counselor, a virtual academic advisor for Green Level High School.
You help students with:
- Course planning and selection (Academic, Honors, and AP courses)
- Understanding course difficulty and workload
- Graduation requirements
- College planning and pathways
- Major exploration and career pathways
- Club information and extracurricular opportunities
- Application terminology and processes
- Scheduling questions and requirements

Always provide accurate, helpful information based on the school's data.
Never provide personal or emotional advice—redirect to a school counselor if needed.
If you don't have specific information, acknowledge it and suggest speaking with a counselor.
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
        data_dir: str = "./data",
        pdf_dir: str = "./data/pdf_docs",
        persist_dir: str = "./chroma_db",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_model: str = "text-embedding-ada-002",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
        self.data_dir = data_dir
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
    
    def _json_to_documents(self, data: Any, source_file: str = "") -> List[Document]:
        """Convert JSON data structure to LangChain Documents. Handles arrays, objects, and nested structures."""
        documents = []
        filename = os.path.basename(source_file) if source_file else "unknown"
        
        # Handle arrays (most common structure)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    doc = self._process_dict_item(item, filename)
                    if doc:
                        documents.append(doc)
        
        # Handle objects/dictionaries
        elif isinstance(data, dict):
            # Check for known structures first (legacy support)
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
                documents.append(Document(page_content=req_text, metadata={"category": "graduation_requirements", "source_file": filename}))
            
            # Process other known structures
            if "courses" in data:
                for course in data["courses"]:
                    course_text = f"COURSE: {course.get('name', 'Unknown')} ({course.get('code', 'N/A')})\n"
                    course_text += f"Credits: {course.get('credits', 'N/A')}\n"
                    course_text += f"Category: {course.get('category', 'N/A')}\n"
                    course_text += f"Description: {course.get('description', 'No description available.')}\n"
                    if course.get('prerequisites'):
                        course_text += f"Prerequisites: {', '.join(course['prerequisites'])}\n"
                    documents.append(Document(page_content=course_text, metadata={"category": "courses", "code": course.get('code', ''), "source_file": filename}))
            
            if "counselors" in data:
                counselors_text = "SCHOOL COUNSELORS:\n"
                for counselor in data["counselors"]:
                    counselors_text += f"\n{counselor.get('name', 'Unknown')}\n"
                    counselors_text += f"Email: {counselor.get('email', 'N/A')}\n"
                    counselors_text += f"Phone: {counselor.get('phone', 'N/A')}\n"
                    counselors_text += f"Specialization: {counselor.get('specialization', 'N/A')}\n"
                    counselors_text += f"Office: {counselor.get('office', 'N/A')}\n"
                documents.append(Document(page_content=counselors_text, metadata={"category": "counselors", "source_file": filename}))
            
            if "policies" in data:
                policies_text = "SCHOOL POLICIES:\n"
                for policy_type, policy_data in data["policies"].items():
                    policies_text += f"\n{policy_type.upper()} POLICIES:\n"
                    for key, value in policy_data.items():
                        policies_text += f"{key.replace('_', ' ').title()}: {value}\n"
                documents.append(Document(page_content=policies_text, metadata={"category": "policies", "source_file": filename}))
            
            if "electives" in data:
                electives_text = "AVAILABLE ELECTIVES:\n"
                for category, elective_list in data["electives"].items():
                    electives_text += f"\n{category.replace('_', ' ').title()}:\n"
                    electives_text += ", ".join(elective_list) + "\n"
                documents.append(Document(page_content=electives_text, metadata={"category": "electives", "source_file": filename}))
            
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
                documents.append(Document(page_content=college_text, metadata={"category": "college_prep", "source_file": filename}))
            
            # Generic processing for unknown structures
            else:
                doc = self._process_dict_item(data, filename)
                if doc:
                    documents.append(doc)
        
        return documents
    
    def _process_dict_item(self, item: dict, source_file: str = "") -> Document:
        """Process a single dictionary item into a Document."""
        if not isinstance(item, dict):
            return None
        
        filename = os.path.basename(source_file) if source_file else "unknown"
        text_parts = []
        
        # Determine category based on common keys
        category = "general"
        if "course_name" in item:
            category = "course_difficulty"
            text_parts.append(f"COURSE DIFFICULTY PROFILE: {item.get('course_name', 'Unknown')}")
            if "category" in item:
                text_parts.append(f"Category: {item.get('category')}")
            if "overall_difficulty_tier" in item:
                text_parts.append(f"Overall Difficulty Tier: {item.get('overall_difficulty_tier')}")
            if "notes_and_context" in item:
                text_parts.append(f"Notes: {item.get('notes_and_context')}")
            
            if "skill_profile" in item:
                skill = item["skill_profile"]
                text_parts.append("\nSKILL REQUIREMENTS:")
                text_parts.append(f"Math Strength Required: {skill.get('math_strength_required', 'N/A')}")
                text_parts.append(f"Reading/Writing Strength Required: {skill.get('reading_writing_strength_required', 'N/A')}")
                text_parts.append(f"Conceptual Thinking: {skill.get('conceptual_thinking', 'N/A')}")
                text_parts.append(f"Memorization Load: {skill.get('memorization_load', 'N/A')}")
            
            if "counselor_profile" in item:
                counselor = item["counselor_profile"]
                text_parts.append("\nCOUNSELOR INFORMATION:")
                text_parts.append(f"Workload: {counselor.get('workload_profile', 'N/A')}")
                text_parts.append(f"Exam Format: {counselor.get('exam_format', 'N/A')}")
        
        elif "College" in item:
            # Check if it's a college pathway (has Stats_Middle_50) or college filter (has Academic_Profile)
            if "Stats_Middle_50" in item or "Major_Specific_Recommendations" in item:
                category = "college_pathway"
                text_parts.append(f"COLLEGE: {item.get('College', 'Unknown')}")
                if "Stats_Middle_50" in item:
                    stats = item["Stats_Middle_50"]
                    text_parts.append(f"GPA Range: {stats.get('GPA_Weighted', 'N/A')}")
                    text_parts.append(f"SAT Range: {stats.get('SAT_Range', 'N/A')}")
                    text_parts.append(f"ACT Range: {stats.get('ACT_Range', 'N/A')}")
                if "World_Language_Recommendation" in item:
                    text_parts.append(f"World Language: {item.get('World_Language_Recommendation')}")
                if "AP_Credit_Policy" in item:
                    text_parts.append(f"AP Credit Policy: {item.get('AP_Credit_Policy')}")
                if "Major_Specific_Recommendations" in item:
                    text_parts.append("\nMAJOR-SPECIFIC RECOMMENDATIONS:")
                    for major in item["Major_Specific_Recommendations"]:
                        text_parts.append(f"\nMajor Cluster: {major.get('Major_Cluster', 'N/A')}")
                        text_parts.append(f"Academic Emphasis: {major.get('Academic_Strength_Emphasis', 'N/A')}")
                        if "Recommended_GLHS_Courses" in major:
                            text_parts.append(f"Recommended Courses: {', '.join(major['Recommended_GLHS_Courses'])}")
                        if "Recommended_GLHS_Clubs" in major:
                            text_parts.append(f"Recommended Clubs: {', '.join(major['Recommended_GLHS_Clubs'])}")
                        if "Recommended_Local_Opportunities" in major:
                            text_parts.append(f"Recommended Opportunities: {', '.join(major['Recommended_Local_Opportunities'])}")
            else:
                # College filter data
                category = "college_filter"
                text_parts.append(f"COLLEGE: {item.get('College', 'Unknown')}")
                if "Academic_Profile" in item:
                    text_parts.append(f"Academic Profile: {item.get('Academic_Profile')}")
                if "Environment" in item:
                    text_parts.append(f"Environment: {item.get('Environment')}")
                if "Size" in item:
                    text_parts.append(f"Size: {item.get('Size')}")
                if "Core_Strengths" in item:
                    text_parts.append(f"Core Strengths: {', '.join(item['Core_Strengths'])}")
                if "Tags" in item:
                    text_parts.append(f"Tags: {', '.join(item['Tags'])}")
                if "Cost_Tag" in item:
                    text_parts.append(f"Cost: {item.get('Cost_Tag')}")
                if "Testing_Policy" in item:
                    text_parts.append(f"Testing Policy: {item.get('Testing_Policy')}")
                if "Flagship_Scholarships" in item:
                    text_parts.append(f"Flagship Scholarships: {', '.join(item['Flagship_Scholarships'])}")
                if "Admissions_URL" in item:
                    text_parts.append(f"Admissions URL: {item.get('Admissions_URL')}")
                if "Net_Price_Calculator_URL" in item:
                    text_parts.append(f"Net Price Calculator: {item.get('Net_Price_Calculator_URL')}")
                if "Scholarship_Page_URL" in item:
                    text_parts.append(f"Scholarship Page: {item.get('Scholarship_Page_URL')}")
        
        elif "pathway_name" in item:
            category = "major_pathway"
            text_parts.append(f"MAJOR PATHWAY: {item.get('pathway_name', 'Unknown')}")
            if "description" in item:
                text_parts.append(f"Description: {item.get('description')}")
            if "related_majors" in item:
                text_parts.append(f"Related Majors: {', '.join(item['related_majors'])}")
            if "key_classes" in item:
                text_parts.append("\nKEY CLASSES:")
                for key, classes in item["key_classes"].items():
                    text_parts.append(f"{key.replace('_', ' ').title()}: {', '.join(classes)}")
            if "related_clubs" in item:
                text_parts.append(f"Related Clubs: {', '.join(item['related_clubs'])}")
            if "recommended_local_opportunities" in item:
                text_parts.append(f"Recommended Opportunities: {', '.join(item['recommended_local_opportunities'])}")
        
        elif "club_name" in item:
            category = "club"
            text_parts.append(f"CLUB: {item.get('club_name', 'Unknown')}")
            if "category" in item:
                text_parts.append(f"Category: {item.get('category')}")
            if "advisor" in item:
                text_parts.append(f"Advisor: {item.get('advisor')}")
            if "room" in item:
                text_parts.append(f"Room: {item.get('room')}")
        
        elif "opportunity_id" in item:
            category = "opportunity"
            text_parts.append(f"OPPORTUNITY: {item.get('name', 'Unknown')}")
            if "type" in item:
                text_parts.append(f"Type: {item.get('type')}")
            if "description" in item:
                text_parts.append(f"Description: {item.get('description')}")
            if "tags" in item:
                text_parts.append(f"Tags: {', '.join(item['tags'])}")
        
        elif "Question" in item:
            category = "faq"
            text_parts.append(f"QUESTION: {item.get('Question', 'Unknown')}")
            if "Answer" in item:
                text_parts.append(f"ANSWER: {item.get('Answer')}")
        
        elif "term" in item:
            category = "glossary"
            text_parts.append(f"TERM: {item.get('term', 'Unknown')}")
            if "definition" in item:
                text_parts.append(f"DEFINITION: {item.get('definition')}")
        
        elif "course_name" in item and "level" in item:
            category = "course_catalog"
            text_parts.append(f"COURSE: {item.get('course_name', 'Unknown')}")
            text_parts.append(f"Level: {item.get('level')}")
            if "category" in item:
                text_parts.append(f"Category: {item.get('category')}")
            if "grade_level" in item:
                text_parts.append(f"Grade Level: {item.get('grade_level')}")
            if "prerequisite_text" in item:
                text_parts.append(f"Prerequisites: {item.get('prerequisite_text')}")
        
        elif "total_credits_required" in item:
            category = "graduation_requirements"
            text_parts.append(f"GRADUATION REQUIREMENTS: {item.get('total_credits_required')} credits required")
            if "subject_requirements" in item:
                text_parts.append("\nSUBJECT REQUIREMENTS:")
                for req in item["subject_requirements"]:
                    text_parts.append(f"{req.get('subject')}: {req.get('credits')} credits - {req.get('notes', '')}")
        
        elif "file_manifest" in item or "global_rules" in item:
            category = "system_metadata"
            # Process metadata file
            if "file_manifest" in item:
                text_parts.append("FILE MANIFEST:")
                for key, value in item["file_manifest"].items():
                    text_parts.append(f"{key}: {value}")
            if "global_rules" in item:
                text_parts.append("\nGLOBAL RULES:")
                for key, value in item["global_rules"].items():
                    if isinstance(value, dict):
                        text_parts.append(f"{key}: {json.dumps(value)}")
                    else:
                        text_parts.append(f"{key}: {value}")
        
        else:
            # Generic fallback - convert all key-value pairs
            category = "general"
            for key, value in item.items():
                if isinstance(value, (list, dict)):
                    text_parts.append(f"{key.replace('_', ' ').title()}: {json.dumps(value)}")
                else:
                    text_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        
        if not text_parts:
            return None
        
        content = "\n".join(text_parts)
        return Document(
            page_content=content,
            metadata={
                "category": category,
                "source_file": filename
            }
        )
    
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
    
    def _load_all_json_files(self) -> List[Document]:
        """Automatically discover and load all JSON files from the data directory."""
        all_documents = []
        
        if not os.path.exists(self.data_dir):
            logger.warning(f"⚠️ Data directory '{self.data_dir}' not found.")
            return all_documents
        
        # Find all JSON and LJSON files
        json_files = list(pathlib.Path(self.data_dir).glob("*.json"))
        json_files.extend(pathlib.Path(self.data_dir).glob("*.ljson"))
        
        if not json_files:
            logger.warning(f"⚠️ No JSON files found in '{self.data_dir}'")
            return all_documents
        
        logger.info(f"📁 Found {len(json_files)} JSON file(s) in '{self.data_dir}'")
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                json_docs = self._json_to_documents(data, str(json_file))
                
                # Mark JSON documents with metadata
                for doc in json_docs:
                    doc.metadata['source_type'] = 'json'
                    if 'source_file' not in doc.metadata:
                        doc.metadata['source_file'] = json_file.name
                
                all_documents.extend(json_docs)
                logger.info(f"✅ Loaded {len(json_docs)} documents from '{json_file.name}'")
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON syntax error in '{json_file.name}': {e}")
                continue
            except Exception as e:
                logger.error(f"❌ Error loading '{json_file.name}': {e}")
                continue
        
        return all_documents
    
    def _initialize_vectorstore(self) -> Chroma:
        """Initialize or load the Chroma vectorstore from all JSON and PDF sources."""
        # Check if vectorstore already exists
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            logger.info(f"Loading existing vectorstore from {self.persist_dir}")
            logger.info("💡 To rebuild with new data, delete the chroma_db folder and restart.")
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
        
        # Create new vectorstore from all JSON + PDF data
        logger.info("Creating new vectorstore from all JSON files and PDFs...")
        all_documents = []
        
        # 1. Load all JSON documents automatically
        json_docs = self._load_all_json_files()
        all_documents.extend(json_docs)
        
        # 2. Load PDF documents
        pdf_docs = self._load_pdf_documents()
        all_documents.extend(pdf_docs)
        
        # Validate we have at least some documents
        if not all_documents:
            raise ValueError(
                "No documents found! Please add JSON data files or PDF files.\n"
                f"Expected JSON files in: {self.data_dir}\n"
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

