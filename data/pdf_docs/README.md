# PDF Documents Directory

**Note: PDF support has been removed. The chatbot currently uses JSON files only.**

This directory is kept for potential future use. If you need to add PDF support in the future, you would need to:
1. Implement PDF text extraction (or OCR for scanned PDFs)
2. Update the `build_vector_db.py` and `chatbot.py` files to load PDFs
3. Add required PDF processing libraries to `requirements.txt`

For now, all data should be in JSON format in the parent `data/` directory.
