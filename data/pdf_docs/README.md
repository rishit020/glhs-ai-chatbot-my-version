# PDF Documents Directory

Place your school's PDF documents in this folder to add rich context to the AI counselor.

## How It Works

- All PDF files in this directory will be automatically loaded when the chatbot starts
- PDFs are parsed, chunked, and added to the vector database
- The AI can retrieve information from both JSON and PDF sources when answering questions

## What PDFs to Add

Recommended PDF documents:

1. **Student Handbook** - `student_handbook.pdf`
   - School policies, rules, procedures
   - Attendance policies
   - Discipline procedures

2. **Course Catalog** - `course_catalog.pdf`
   - Detailed course descriptions
   - Prerequisites and requirements
   - Course sequences

3. **Graduation Requirements** - `graduation_requirements.pdf`
   - Detailed credit breakdown
   - Course requirements by subject
   - Special programs info

4. **Academic Policies** - `academic_policies.pdf`
   - Grading policies
   - Honor roll requirements
   - Testing policies

5. **College Prep Guide** - `college_prep.pdf`
   - AP/Honors information
   - College application process
   - Scholarship information

## Adding PDFs

1. Simply copy your PDF files into this `pdf_docs/` folder
2. Delete the `chroma_db/` folder (if it exists) to rebuild the database
3. Restart the application - PDFs will be automatically loaded

## Best Practices

- Use clear, descriptive filenames (e.g., `student_handbook_2024.pdf`)
- Ensure PDFs are text-based (not scanned images)
- Keep file sizes reasonable (<50MB per file recommended)
- Update PDFs as school information changes

## Notes

- The system will work with just JSON, just PDFs, or both (hybrid)
- If no PDFs are found, the system will continue with JSON data only
- PDF processing happens automatically - no code changes needed!

