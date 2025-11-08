# Green Level AI Counselor Chatbot

A RAG-powered AI chatbot for Green Level High School that helps students with academic counseling, course planning, and graduation requirements.

## Features

- 🤖 AI-powered responses using GPT-4o-mini
- 📚 RAG (Retrieval-Augmented Generation) with ChromaDB vector search
- 🔄 **Hybrid Approach**: Supports both JSON structured data AND PDF documents
- 💬 Interactive chat interface with Green Level branding
- ⚡ Quick action buttons for common questions
- 🧠 Session-based conversation memory
- 🛡️ Safety guardrails for mental health concerns

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

⚠️ **IMPORTANT**: The OpenAI API key **ONLY** goes in the `.env` file. Do NOT put it in code files!

Create a `.env` file in the `glhs-chatbot` root directory (same folder as `app.py`):

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**File location**: `glhs-chatbot/.env`

**Critical Notes:**
- ✅ The `.env` file is already in `.gitignore` (your key won't be committed)
- ❌ Do NOT put your API key in `app.py`, `chatbot.py`, or any other code files
- ❌ Do NOT hardcode the key anywhere in the code
- ✅ Replace `your_openai_api_key_here` with your actual OpenAI API key from https://platform.openai.com/api-keys

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Project Structure

```
glhs-chatbot/
├── app.py                 # Flask routes and main application
├── chatbot.py             # RAG logic, vector store management
├── utils.py               # Helper functions (session management)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore             # Git ignore rules
│
├── templates/
│   └── index.html         # Main chat interface
│
├── static/
│   ├── style.css          # Styling (Green Level colors)
│   └── script.js          # Frontend chat logic
│
├── data/
│   ├── glhs_info.json     # Structured school data (courses, requirements, etc.)
│   └── pdf_docs/          # PDF documents (handbooks, catalogs, etc.)
│
└── chroma_db/             # ChromaDB vector database (auto-generated)
```

## How It Works

1. **Hybrid Data Loading**: The system loads data from two sources:
   - **JSON File** (`data/glhs_info.json`): Structured data (courses, credits, counselors, policies)
   - **PDF Documents** (`data/pdf_docs/`): Rich documents (handbooks, catalogs, detailed policies)
   
   Both sources are combined into a single vector store for comprehensive retrieval.

2. **Vector Store**: Data from both sources is converted to documents, chunked, and embedded using OpenAI embeddings, then stored in ChromaDB for fast retrieval.

3. **Query Processing**: When a user asks a question:
   - Relevant chunks are retrieved from the vector store
   - Context is combined with conversation history
   - GPT-4o-mini generates a response based on the retrieved information

4. **Session Management**: Each user session maintains conversation history for context-aware responses.

## Safety Features

The chatbot includes safeguards for mental health concerns:
- Detects keywords related to mental health crises
- Automatically redirects users to real counselors
- Provides emergency contact information

## Customization

### Adding/Updating Data

**Option 1: JSON File (Structured Data)**
Edit `data/glhs_info.json` to update:
- Course information
- Counselor contact details
- Graduation requirements
- School policies

**Option 2: PDF Documents (Rich Context)**
Place PDF files in `data/pdf_docs/` folder:
- Student handbook PDFs
- Course catalog PDFs
- Policy documents
- Any school documents

**Option 3: Hybrid (Recommended)**
Use both! JSON for quick structured queries, PDFs for detailed explanations.

After updating JSON or adding/removing PDFs, delete the `chroma_db/` directory and restart the app to rebuild the vector store.

### Styling

Customize colors and styling in `static/style.css`. Green Level brand colors:
- Primary green: `#00843D`
- White: `#FFFFFF`
- Black: `#000000`

## Troubleshooting

### "OPENAI_API_KEY not found"
- Make sure you've created a `.env` file with your OpenAI API key

### "Data file not found"
- Ensure `data/glhs_info.json` exists in the `data/` directory

### Vector store issues
- Delete the `chroma_db/` directory and restart to rebuild

## License

This project is for Green Level High School use.
