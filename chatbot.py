"""
Chatbot implementation with RAG (Retrieval-Augmented Generation) for Green Level High School.
Includes greeting detection and scope filtering.
"""
import os
import re
import json
import pathlib
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from utils import load_json_data

try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class GLHSChatbot:
    """Chatbot for Green Level High School with RAG capabilities."""
    
    def __init__(self):
        """Initialize the chatbot with vector store and LLM."""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        # Initialize vector store
        persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
        
        # Check if database exists and has data, if not, build it
        if not self._vector_db_exists(persist_directory):
            print("Vector database not found. Building from JSON files...")
            self._build_vector_database(persist_directory)
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Load school data for context
        self.school_data = self._load_school_data()
        
        # Greeting patterns (case-insensitive, flexible)
        self.greeting_patterns = [
            r'\b(hi|hello|hey|greetings|howdy)\b',
            r'\bwhat\'?s\s+up\b',
            r'\bhow\s+are\s+you\b',
            r'\bhow\s+do\s+you\s+do\b',
            r'\bgood\s+(morning|afternoon|evening|day)\b',
            r'\bnice\s+to\s+meet\s+you\b',
            r'\bhey\s+there\b',
            r'\bhi\s+there\b',
            r'\bhello\s+there\b',
        ]
        
        # School-related keywords for scope detection
        self.school_keywords = [
            'school', 'academic', 'course', 'class', 'grade', 'gpa', 'credit',
            'graduation', 'college', 'university', 'counselor', 'schedule',
            'curriculum', 'requirement', 'prerequisite', 'honors', 'ap', 'academic',
            'teacher', 'student', 'homework', 'assignment', 'exam', 'test', 'quiz',
            'semester', 'year', 'freshman', 'sophomore', 'junior', 'senior',
            'green level', 'glhs', 'wcpss', 'wake county', 'counseling',
            'major', 'career', 'pathway', 'club', 'extracurricular', 'sport',
            'scholarship', 'application', 'admission', 'transcript', 'diploma'
        ]
    
    def _load_school_data(self) -> Dict:
        """Load school data from JSON files."""
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        school_data = {}
        
        try:
            school_data["glhs_info"] = load_json_data(
                os.path.join(data_dir, "glhs_info.json")
            )
        except Exception as e:
            print(f"Warning: Could not load glhs_info.json: {e}")
        
        return school_data
    
    def _vector_db_exists(self, persist_directory: str) -> bool:
        """Check if vector database exists and has data."""
        if not os.path.exists(persist_directory):
            return False
        
        # Check if directory has content
        try:
            path = pathlib.Path(persist_directory)
            if not any(path.iterdir()):
                return False
            
            # Try to load the collection to verify it has data
            test_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            # Try to get collection count
            collection = test_store._collection
            if collection and collection.count() > 0:
                return True
        except Exception:
            pass
        
        return False
    
    def _is_glhs_relevant(self, text: str) -> bool:
        """Filter content to only include GLHS-relevant information."""
        text_lower = text.lower()
        
        # GLHS-specific keywords (positive indicators)
        glhs_keywords = [
            "green level", "glhs", "green level high school",
            "course", "graduation", "credit", "prerequisite", "honors", "ap ",
            "advanced placement", "gpa", "grade point average", "schedule",
            "registration", "elective", "required", "math", "science", "english",
            "social studies", "world language", "arts", "cte", "healthful living",
            "physical education"
        ]
        
        # School-specific keywords to exclude
        exclude_keywords = [
            "apex high school", "cary high school", "garner high school",
            "holly springs high school", "leesville road high school",
            "middle creek high school", "panther creek high school",
            "wakefield high school", "wake forest high school", "enloe high school",
            "millbrook high school", "sanderson high school", "broughton high school",
            "athens drive high school", "fuquay-varina high school",
            "green hope high school", "heritage high school", "hillside high school",
            "knightdale high school", "rolesville high school", "southeast raleigh",
            "southeast raleigh high school", "wake early college", "wake stem",
            "wake young men's", "wake young women's"
        ]
        
        # Check for exclusion keywords first
        for exclude in exclude_keywords:
            if exclude in text_lower:
                return False
        
        # Check for GLHS or general keywords
        for keyword in glhs_keywords:
            if keyword in text_lower:
                return True
        
        # Include general educational content
        educational_keywords = [
            "requirement", "curriculum", "program", "pathway", "endorsement", "diploma"
        ]
        
        for keyword in educational_keywords:
            if keyword in text_lower:
                return True
        
        return False
    
    def _load_pdf_files(self, pdf_dir: str) -> List[Document]:
        """Load PDF files and filter for GLHS-relevant content."""
        if not PDF_SUPPORT:
            return []
        
        documents = []
        
        if not os.path.exists(pdf_dir):
            return documents
        
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            file_path = os.path.join(pdf_dir, pdf_file)
            try:
                reader = PdfReader(file_path)
                all_text = []
                total_pages = len(reader.pages)
                relevant_pages = 0
                
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            if self._is_glhs_relevant(page_text):
                                all_text.append(page_text)
                                relevant_pages += 1
                            elif any(keyword in page_text.lower() for keyword in 
                                   ["course", "graduation", "credit", "requirement", 
                                    "curriculum", "program", "pathway"]):
                                all_text.append(page_text)
                                relevant_pages += 1
                    except Exception:
                        continue
                
                if all_text:
                    full_text = "\n\n".join(all_text)
                    sections = re.split(r'\n{2,}', full_text)
                    filtered_sections = [s for s in sections if s.strip() and self._is_glhs_relevant(s)]
                    
                    if filtered_sections:
                        final_text = "\n\n".join(filtered_sections)
                        doc = Document(
                            page_content=final_text,
                            metadata={"source": pdf_file, "type": "pdf", "file": pdf_file}
                        )
                        documents.append(doc)
            except Exception as e:
                print(f"Warning: Could not load PDF {pdf_file}: {e}")
        
        return documents
    
    def _build_vector_database(self, persist_directory: str):
        """Build vector database from JSON files and PDFs."""
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, "data")
        
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200
        
        all_documents = []
        
        # Load JSON files
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
            "system_metadata.json",
            "wcpss_planning_guide_glhs.json",
            "class_directory.json"
        ]
        
        for json_file in json_files:
            file_path = os.path.join(data_dir, json_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    text_content = json.dumps(data, indent=2, ensure_ascii=False)
                    doc = Document(
                        page_content=text_content,
                        metadata={"source": json_file, "type": "json", "file": json_file}
                    )
                    all_documents.append(doc)
                except Exception as e:
                    print(f"Warning: Could not load {json_file}: {e}")
        
        # Load PDF files
        pdf_dir = os.path.join(data_dir, "pdf_docs")
        pdf_docs = self._load_pdf_files(pdf_dir)
        all_documents.extend(pdf_docs)
        
        if not all_documents:
            print("Warning: No documents found to build vector database!")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(all_documents)
        
        # Create vector store
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
        
        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        print(f"Vector database built with {len(chunks)} chunks from {len(all_documents)} documents")
    
    def _is_greeting(self, text: str) -> bool:
        """Check if the input is a greeting."""
        text_lower = text.lower().strip()
        
        # Check against greeting patterns
        for pattern in self.greeting_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check for very short messages that are likely greetings
        if len(text_lower.split()) <= 3:
            if text_lower in ['hi', 'hello', 'hey', 'sup', 'yo', 'hiya']:
                return True
        
        return False
    
    def _is_school_related(self, text: str) -> bool:
        """
        Check if the input is related to school, academics, or education.
        More permissive: allows any question with academic/school-related keywords.
        """
        text_lower = text.lower()
        
        # Check if it's a greeting (greetings are always allowed)
        if self._is_greeting(text):
            return True
        
        # Expanded school-related keywords - if ANY of these appear, consider it school-related
        school_keywords = [
            # School context
            'green level', 'glhs', 'wcpss', 'wake county',
            'at this school', 'at glhs', 'at green level',
            'school\'s', 'schools', 'our school', 'the school',
            'counselor', 'counseling', 'counselors',
            # Academic terms
            'course', 'class', 'classes', 'schedule', 'scheduling',
            'graduation', 'requirement', 'requirements', 'prerequisite', 'prerequisites',
            'credit', 'credits', 'gpa', 'grade point average', 'transcript', 'diploma',
            'curriculum', 'semester', 'year', 'freshman', 'sophomore', 'junior', 'senior',
            'honors', 'ap ', 'advanced placement', 'academic', 'academics',
            'teacher', 'teachers', 'student', 'students',
            # School activities
            'club', 'clubs', 'extracurricular', 'sport', 'sports', 'scholarship',
            # College/career
            'college prep', 'college preparation', 'admission', 'application', 'applications',
            'college', 'university', 'major', 'majors', 'career', 'pathway', 'pathways',
            # Academic subjects (when used in school context)
            'math', 'mathematics', 'science', 'english', 'history', 'social studies',
            'biology', 'chemistry', 'physics', 'world language', 'foreign language',
            'arts', 'art', 'music', 'pe', 'physical education', 'health',
            # Planning/guidance
            'plan my', 'planning', 'what should i take', 'what classes should',
            'recommend', 'recommendation', 'advice', 'guidance',
            # Question patterns that suggest school context
            'what classes', 'which classes', 'what courses', 'which courses',
            'how do i', 'can i', 'should i take', 'when is', 'where is',
            'help with school', 'school help'
        ]
        
        # If it contains ANY school-related keyword, it's school-related
        if any(keyword in text_lower for keyword in school_keywords):
            return True
        
        return False
    
    def _generate_greeting_response(self, text: str) -> str:
        """Generate a friendly response to greetings."""
        text_lower = text.lower().strip()
        
        # Personalized greeting responses
        if 'how are you' in text_lower or 'how do you do' in text_lower:
            return "I'm doing great, thank you for asking! I'm here to help you with questions about Green Level High School, course planning, graduation requirements, college preparation, and academic counseling. What can I help you with today?"
        
        if 'what\'s up' in text_lower or 'whats up' in text_lower or 'sup' in text_lower:
            return "Not much! I'm here to help you with anything related to Green Level High School—courses, graduation requirements, college prep, scheduling, and more. What's on your mind?"
        
        # Default friendly greeting
        return "Hello! I'm the Green Level High School AI counselor. I can help you with questions about courses, graduation requirements, college preparation, scheduling, and academic planning. How can I assist you today?"
    
    def _is_homework_or_test_question(self, text: str) -> bool:
        """Detect if the question is asking for homework/test answers or academic problem solving."""
        text_lower = text.lower()
        
        # Patterns that indicate homework/test questions
        homework_patterns = [
            r'\bsolve\s+(this|that|the)\s+(problem|equation|question)',
            r'\bwhat\s+is\s+\d+\s*[+\-*/]\s*\d+',  # Math problems like "what is 1+1"
            r'\bcalculate\s+', r'\bcompute\s+', r'\bevaluate\s+',
            r'\banswer\s+(this|that|the)\s+(question|problem)',
            r'\bhelp\s+me\s+(solve|with|do)\s+(this|my|the)\s+(homework|assignment|problem)',
            r'\bwhat\s+is\s+the\s+answer\s+to',
            r'\bhow\s+do\s+i\s+(solve|calculate|find)',
            r'\bexplain\s+(how|why)\s+to\s+(solve|calculate)',
            r'\btest\s+(question|answer)', r'\bquiz\s+(question|answer)',
            r'\bhomework\s+(help|question|problem)',
            r'\bassignment\s+(help|question|problem)',
        ]
        
        for pattern in homework_patterns:
            if re.search(pattern, text_lower):
                # But allow if it's about school's homework policies or test schedules
                if any(kw in text_lower for kw in ['policy', 'schedule', 'due date', 'when is', 'glhs', 'green level', 'school']):
                    return False
                return True
        
        # Check for math problems (simple arithmetic)
        if re.search(r'\b\d+\s*[+\-*/×÷]\s*\d+', text_lower):
            # But allow if asking about math classes at school
            if not any(kw in text_lower for kw in ['class', 'course', 'glhs', 'green level', 'school', 'math class']):
                return True
        
        # Check for general "what is X" questions that aren't about the school
        # But allow questions about school classes (e.g., "what is AP Biology class like?")
        if re.search(r'\bwhat\s+is\s+', text_lower) or re.search(r'\bwhat\s+are\s+', text_lower):
            # If it mentions "class", "course", "like", "at", it's likely about school
            if any(kw in text_lower for kw in ['class', 'course', ' like', ' at ', 'offered', 'available']):
                return False
            
            # If it mentions school context, it's about school
            if any(kw in text_lower for kw in ['glhs', 'green level', 'school', 'counselor', 'requirement']):
                return False
            
            # Check for course/class names (AP, Honors, etc.) - these are likely about school
            if re.search(r'\b(ap|honors?|academic)\s+', text_lower):
                return False
            
            # Check if it's asking about a general concept (science, history, math, etc.)
            # These are likely general knowledge questions
            general_knowledge_patterns = [
                r'what\s+is\s+(photosynthesis|gravity|evolution|atoms?|molecules?|cells?|dna|rna)',
                r'what\s+is\s+(the\s+)?(speed\s+of\s+light|law\s+of|theory\s+of|formula\s+for)',
                r'what\s+is\s+\d+',  # "what is 5" (likely math)
                r'what\s+are\s+(atoms?|molecules?|cells?|genes?|proteins?)',
                r'who\s+(is|was|are|were)\s+',  # "who is/was" (general knowledge)
                r'when\s+(did|was|were)\s+',  # "when did/was" (history)
                r'where\s+(is|are|was|were)\s+',  # "where is" (geography)
            ]
            
            for pattern in general_knowledge_patterns:
                if re.search(pattern, text_lower):
                    return True
            
            # Very short "what is X" questions without school context are likely general knowledge
            # But if it contains course-related terms, it might be about school
            if len(text_lower.split()) <= 5:
                # Check if it might be a course name
                if not any(kw in text_lower for kw in ['math', 'science', 'english', 'history', 'biology', 'chemistry', 'physics']):
                    return True
        
        return False
    
    def _is_outside_scope(self, text: str) -> bool:
        """
        Determine if the question is completely outside the scope of school/academics.
        Only blocks: clearly unrelated topics (weather, recipes, etc.) and simple math problems without context.
        """
        # If it's a greeting, it's not outside scope
        if self._is_greeting(text):
            return False
        
        # If it's school-related, it's not outside scope
        if self._is_school_related(text):
            return False
        
        text_lower = text.lower()
        
        # Check for simple math problems without any school context (e.g., "what's 1+1")
        # This is the main thing we want to block
        if re.search(r'\bwhat\s+is\s+\d+\s*[+\-*/]\s*\d+', text_lower) or \
           re.search(r'\b\d+\s*[+\-*/×÷]\s*\d+', text_lower):
            # Only block if there's NO school/academic context
            if not any(kw in text_lower for kw in ['class', 'course', 'school', 'academic', 'math class', 'glhs', 'green level']):
                return True
        
        # Check for clearly unrelated topics (non-academic)
        unrelated_keywords = [
            'weather', 'recipe', 'cooking', 'sports score', 'movie', 'tv show',
            'celebrity', 'gossip', 'politics', 'religion', 'dating', 'relationship',
            'shopping', 'restaurant', 'travel', 'vacation', 'game', 'video game',
            'sports team', 'nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football game',
            'capital of', 'president of', 'who invented', 'trivia', 'fun fact'
        ]
        
        for keyword in unrelated_keywords:
            if keyword in text_lower:
                # Only block if it's clearly not school-related
                if not any(school_kw in text_lower for school_kw in ['school', 'class', 'course', 'academic', 'glhs', 'green level']):
                    return True
        
        # Check for homework/test question patterns (solving problems)
        if self._is_homework_or_test_question(text):
            return True
        
        # Default: give benefit of the doubt - if we're not sure, let RAG try to answer
        # This is more permissive - only block things we're CERTAIN are unrelated
        return False
    
    def _format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> List:
        """Convert conversation history to LangChain message format."""
        messages = []
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        
        return messages
    
    def query_with_rag(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Query the chatbot with RAG. Handles greetings, school-related queries, and scope filtering.
        
        Args:
            question: User's question
            conversation_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
        
        Returns:
            Response string
        """
        question = question.strip()
        
        if not question:
            return "I'm here to help! Please ask me a question about Green Level High School."
        
        # Handle greetings
        if self._is_greeting(question):
            return self._generate_greeting_response(question)
        
        # Check if outside scope (only for clearly unrelated topics)
        if self._is_outside_scope(question):
            return (
                "I'm designed to help with questions about Green Level High School, including "
                "courses, graduation requirements, college preparation, scheduling, and academic planning. "
                "I'm not able to answer questions outside of these topics. "
                "Is there something school-related I can help you with instead?"
            )
        
        # For school-related queries, use RAG
        try:
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(question)
            
            # Build context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
            
            # If no context retrieved, still proceed but LLM will handle it gracefully
            # Format conversation history
            history_messages = []
            if conversation_history:
                # Only include recent history (last 6 messages to avoid token limits)
                # Exclude the last message if it's the same as the current question (prevent duplication)
                recent_history = conversation_history[-6:]
                if (recent_history and 
                    recent_history[-1].get("role") == "user" and 
                    recent_history[-1].get("content", "").strip().lower() == question.strip().lower()):
                    recent_history = recent_history[:-1]  # Remove duplicate
                history_messages = self._format_conversation_history(recent_history)
            
            # Build system message
            system_content = (
                "You are a helpful AI counselor for Green Level High School (GLHS). "
                "You answer questions related to school, academics, courses, graduation requirements, "
                "school policies, schedules, clubs, events, counselors, college preparation, and academic planning. "
                "\n\n"
                "CRITICAL RULES:\n"
                "- Answer ANY question that is related to school, academics, courses, prerequisites, requirements, or education\n"
                "- DO NOT solve simple math problems without context (e.g., 'what is 1+1?')\n"
                "- DO NOT solve homework problems or provide test answers\n"
                "- DO NOT answer questions about completely unrelated topics (weather, recipes, etc.)\n"
                "- Use information from the provided context about GLHS when available\n"
                "- If the question is academic/school-related but you don't have specific information, "
                "provide helpful general guidance or explain what information might be needed\n"
                "\n"
                "Be friendly, professional, and accurate. Use the provided context to answer questions when available. "
                "If the context doesn't contain enough information, say so politely and suggest what information might be helpful. "
                "Always be encouraging and supportive of students' academic goals."
            )
            
            # Build user prompt with context
            user_prompt = f"Context from school documents:\n{context}\n\n"
            user_prompt += f"Question: {question}\n\n"
            user_prompt += "Please provide a helpful answer based on the context above."
            
            # Create messages
            messages = [SystemMessage(content=system_content)]
            messages.extend(history_messages)
            messages.append(HumanMessage(content=user_prompt))
            
            # Generate response
            # For langchain 0.3.x, use __call__ method
            response = self.llm(messages)
            return response.content.strip()
            
        except Exception as e:
            print(f"Error in RAG query: {e}")
            return (
                "I encountered an error while processing your question. "
                "Please try rephrasing your question or ask about something else. "
                "I'm here to help with Green Level High School topics!"
            )


# Singleton instance
_chatbot_instance = None


def get_chatbot() -> GLHSChatbot:
    """Get or create the chatbot singleton instance."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = GLHSChatbot()
    return _chatbot_instance
