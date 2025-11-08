# Quick Setup Guide

## Prerequisites
- Python 3.8 or higher
- OpenAI API key

## Installation Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file (REQUIRED - API Key Setup):**
   
   ⚠️ **IMPORTANT**: The OpenAI API key **ONLY** goes in the `.env` file. Do NOT put it anywhere else!
   
   Create a file named `.env` in the `glhs-chatbot` directory (same folder as `app.py`) with:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```
   
   **File location**: `glhs-chatbot/.env` (in the root of the project folder)
   
   **Important Notes:**
   - ✅ The `.env` file is already in `.gitignore` (safe from version control)
   - ✅ Do NOT put your API key in any Python files
   - ✅ Do NOT commit the `.env` file to git
   - ✅ Replace `sk-your-actual-key-here` with your actual OpenAI API key

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:5000`

## First Run

On the first run, the system will:
- Load the JSON data from `data/glhs_info.json`
- Convert it to documents and chunk them
- Create embeddings using OpenAI
- Build and persist the ChromaDB vector store

This may take a minute or two. Subsequent runs will be faster as the vector store is reused.

## Troubleshooting

- **"OPENAI_API_KEY not found"**: 
  - Make sure your `.env` file exists in `glhs-chatbot/` folder (same location as `app.py`)
  - Verify the file is named exactly `.env` (not `env.txt` or `.env.txt`)
  - Check that the file contains: `OPENAI_API_KEY=sk-your-key-here` (no quotes around the key)
  - Make sure there are no spaces around the `=` sign
- **Import errors**: Run `pip install -r requirements.txt` again
- **Port already in use**: Change the port in `app.py` (line 148) from 5000 to another port

## API Key Security

**Remember:**
- ✅ Your API key ONLY belongs in the `.env` file
- ✅ The `.env` file is already excluded from git (in `.gitignore`)
- ❌ Never share your `.env` file or API key
- ❌ Never commit the `.env` file to version control

