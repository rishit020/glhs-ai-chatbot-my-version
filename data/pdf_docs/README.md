# PDF Documents Directory

**PDF support is now enabled!**

Place PDF files in this directory to include them in the RAG model training. The system will:
1. Extract text from PDFs using `pypdf`
2. Filter content to only include GLHS-relevant information
3. Exclude content specific to other WCPSS schools
4. Include general course, graduation, and program information applicable to GLHS

## Current PDFs:
- `WCPSS_2025-2026_High_School_Program_Planning_Guide.pdf` - WCPSS program planning guide (filtered for GLHS-relevant content)

## Filtering Logic:
The system automatically filters out:
- Content specific to other high schools (Apex, Cary, Garner, etc.)
- School-specific programs not available at GLHS

The system includes:
- GLHS-specific information
- General course information
- Graduation requirements
- Program pathways applicable to all schools
- CTE, Arts, and elective information
