"""
Chatbot implementation with RAG (Retrieval-Augmented Generation) for Green Level High School.
Includes greeting detection and scope filtering.
"""
import os
import re
import json
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from utils import load_json_data


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
    
    def _build_vector_database(self, persist_directory: str):
        """Build vector database from JSON files."""
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
            "system_metadata.json"
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
        Check if the input is DIRECTLY related to Green Level High School.
        Allows: school courses, schedules, requirements, policies, clubs, events, guidance.
        Blocks: general knowledge, homework problems, test answers.
        """
        text_lower = text.lower()
        
        # Check if it's a greeting (greetings are always allowed)
        if self._is_greeting(text):
            return True
        
        # Explicit school context keywords
        school_context_keywords = [
            'green level', 'glhs', 'wcpss', 'wake county',
            'at this school', 'at glhs', 'at green level',
            'school\'s', 'schools', 'our school', 'the school',
            'counselor', 'counseling', 'counselors'
        ]
        
        has_school_context = any(keyword in text_lower for keyword in school_context_keywords)
        
        # School-related keywords that imply school context
        school_keywords = [
            'course', 'class', 'schedule', 'graduation', 'requirement',
            'prerequisite', 'credit', 'gpa', 'transcript', 'diploma',
            'club', 'extracurricular', 'sport', 'scholarship',
            'college prep', 'college preparation', 'admission', 'application',
            'curriculum', 'semester', 'year', 'freshman', 'sophomore', 'junior', 'senior',
            'honors', 'ap ', 'academic', 'teacher', 'student'
        ]
        
        has_school_keyword = any(keyword in text_lower for keyword in school_keywords)
        
        # Casual school-related phrases (these imply school context)
        casual_school_phrases = [
            'how\'s school', 'how is school', 'school going', 'classes going',
            'what classes', 'which classes', 'what courses', 'which courses',
            'help with school', 'school help', 'plan my', 'planning',
            'what should i take', 'what classes should', 'recommend',
            'college', 'major', 'career', 'pathway', 'what\'s the schedule',
            'when is', 'where is', 'how do i', 'can i', 'should i take'
        ]
        
        has_casual_school_phrase = any(phrase in text_lower for phrase in casual_school_phrases)
        
        # If it has explicit school context, it's school-related
        if has_school_context:
            return True
        
        # If it has school keywords AND casual school phrases, it's likely school-related
        if has_school_keyword and has_casual_school_phrase:
            return True
        
        # If it's asking about courses/classes/schedule without explicit context, 
        # assume it's about school (common pattern)
        if any(kw in text_lower for kw in ['course', 'class', 'schedule', 'requirement']) and \
           any(phrase in text_lower for phrase in ['what', 'which', 'how', 'when', 'where', 'can', 'should']):
            return True
        
        # Allow casual check-ins about school
        if has_casual_school_phrase:
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
        Determine if the question is outside the scope of Green Level High School.
        Blocks: homework/test questions, general knowledge, math problems, science questions, etc.
        """
        # If it's a greeting, it's not outside scope
        if self._is_greeting(text):
            return False
        
        # If it's directly school-related, it's not outside scope
        if self._is_school_related(text):
            return False
        
        # Check for homework/test questions
        if self._is_homework_or_test_question(text):
            return True
        
        # Check for clearly unrelated topics
        unrelated_keywords = [
            'weather', 'recipe', 'cooking', 'sports score', 'movie', 'tv show',
            'celebrity', 'gossip', 'politics', 'religion', 'dating', 'relationship',
            'shopping', 'restaurant', 'travel', 'vacation', 'game', 'video game',
            'sports team', 'nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football game',
            'capital of', 'president of', 'who invented', 'when was', 'where is',
            'trivia', 'fun fact'
        ]
        
        text_lower = text.lower()
        for keyword in unrelated_keywords:
            if keyword in text_lower:
                # But check if it might still be school-related (e.g., "sports team at school")
                if not any(school_kw in text_lower for school_kw in ['school', 'class', 'course', 'academic', 'glhs', 'green level']):
                    return True
        
        # Check for general knowledge/science questions without school context
        science_math_indicators = [
            'what is photosynthesis', 'what is gravity', 'what is the speed of light',
            'how does', 'why does', 'explain the', 'define', 'what causes',
            'formula for', 'equation for', 'theory of', 'law of'
        ]
        
        for indicator in science_math_indicators:
            if indicator in text_lower:
                # Only block if no school context
                if not any(kw in text_lower for kw in ['class', 'course', 'glhs', 'green level', 'school', 'at school']):
                    return True
        
        # Very short questions that aren't greetings and don't have school context
        if len(text.split()) <= 4 and not self._is_greeting(text):
            if not any(kw in text_lower for kw in ['glhs', 'green level', 'school', 'class', 'course', 'counselor']):
                # Likely a general knowledge question
                return True
        
        # Check if it's a very vague question that could be school-related
        # Give benefit of the doubt for questions that might be about school
        vague_but_possible_school = [
            'what', 'how', 'when', 'where', 'why', 'can', 'should', 'do'
        ]
        
        # If it's a question word but very short and no clear topic, might be school-related
        if any(word in text_lower.split()[:3] for word in vague_but_possible_school):
            if len(text.split()) <= 5:
                # Very short questions might be school-related, let RAG try
                return False
        
        # Default: if we can't determine it's school-related and it's not clearly a greeting,
        # block it to be safe (strict mode)
        return True
    
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
                "You ONLY answer questions that are DIRECTLY related to Green Level High School, including: "
                "courses offered at GLHS, graduation requirements, school policies, schedules, "
                "clubs, events, counselors, college preparation guidance, and general school-related guidance. "
                "\n\n"
                "CRITICAL RULES:\n"
                "- DO NOT answer general knowledge questions (e.g., 'what is photosynthesis?', 'what is 1+1?')\n"
                "- DO NOT solve homework problems, test questions, or provide answers to assignments\n"
                "- DO NOT answer questions about topics outside of Green Level High School\n"
                "- ONLY use information from the provided context about GLHS\n"
                "- If the question is not about GLHS specifically, politely redirect to school-related topics\n"
                "\n"
                "Be friendly, professional, and accurate. Use only the provided context to answer questions. "
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
