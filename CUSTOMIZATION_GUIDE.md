# Customization Guide: Adding Your School's Data

## ✅ **YES, the code is 100% flexible!**

You can completely customize the data for your school. Here's why and how:

### Why It's Flexible

1. **All fields are optional** - The code uses `.get()` with defaults, so missing fields won't break anything
2. **Sections are optional** - Each major section (courses, counselors, policies, etc.) checks `if "key" in data` before processing
3. **Dynamic structure** - Policies, electives, and other sections can have any keys/values you want
4. **Graceful fallbacks** - Missing data shows "N/A" instead of crashing

---

## 📝 How to Customize

### Option 1: Edit the JSON File (Easiest)

Just edit `data/glhs_info.json` with your school's information:

**What you can change:**
- ✅ All credit requirements numbers
- ✅ All course names, codes, descriptions
- ✅ Counselor names, emails, phone numbers
- ✅ All policy text
- ✅ Elective lists (add/remove any)
- ✅ College prep information

**What's optional (you can remove):**
- Any section you don't need
- Any field within a section
- Entire categories (e.g., if you don't have AP courses, just remove that section)

### Option 2: Add Custom Sections

You can even add your own custom sections! The code will process them through the "policies" or other flexible sections.

---

## 🎯 Quick Start: Replacing Data

### Step 1: Graduation Requirements

Replace the numbers in `graduation_requirements`:

```json
{
  "graduation_requirements": {
    "total_credits": 24,  // ← Change to your school's requirement
    "english_credits": 4,
    "math_credits": 4,
    // ... add or remove any credit categories
    "description": "Your school's graduation description here"
  }
}
```

### Step 2: Courses

Replace the course list with your actual courses:

```json
{
  "courses": [
    {
      "code": "YOUR_CODE",  // ← Your course codes
      "name": "Your Course Name",
      "credits": 1,
      "description": "Your course description",
      "prerequisites": ["COURSE_CODE"],  // ← Optional
      "category": "English"  // ← Any category you want
    }
    // Add as many courses as you need
  ]
}
```

### Step 3: Counselors

Replace with your actual counselors:

```json
{
  "counselors": [
    {
      "name": "Actual Counselor Name",
      "email": "real@yourschool.edu",
      "phone": "(555) 123-4567",
      "specialization": "Your specialization text",
      "office": "Room 101"
    }
    // Add all your counselors
  ]
}
```

### Step 4: Policies

Customize policies - you can add any policy categories:

```json
{
  "policies": {
    "attendance": {
      "tardies": "Your tardy policy",
      "absences": "Your absence policy",
      "makeup_work": "Your makeup policy"
    },
    "academic": {
      // Add any academic policies
      "gpa_requirements": "Your GPA policy",
      "grade_appeals": "Your appeals process"
    },
    "your_custom_category": {  // ← You can add ANY category
      "custom_rule": "Your custom rule text"
    }
  }
}
```

### Step 5: Electives

Add your school's electives:

```json
{
  "electives": {
    "arts": ["Art I", "Band", "Your electives..."],
    "technology": ["Computer Science", "Your tech courses..."],
    "custom_category": ["Option 1", "Option 2"]  // ← Add any category
  }
}
```

---

## 📄 Adding PDFs (Hybrid Approach)

After we implement PDF support, you can also:

1. **Drop PDFs in `data/pdf_docs/` folder:**
   - Student handbook PDFs
   - Course catalog PDFs
   - Policy documents
   - Any other school documents

2. **The system will automatically:**
   - Parse the PDFs
   - Extract all text
   - Add to the vector database
   - Make it searchable by the AI

**No code changes needed** - just add PDF files!

---

## 🔄 Updating Data After First Run

### If you change the JSON file:

1. Delete the `chroma_db/` folder
2. Restart the app - it will rebuild the vector store with new data

### If you add/remove PDFs:

1. Delete the `chroma_db/` folder  
2. Restart the app - it will rebuild with all PDFs in the folder

---

## 💡 Examples of What You Can Do

### Example 1: Different Credit System

If your school uses units instead of credits:

```json
{
  "graduation_requirements": {
    "total_units": 220,  // ← Just change the field name
    "english_units": 40,
    // ... the AI will understand from context
  }
}
```

### Example 2: Add Sports/Activities

Add a new section:

```json
{
  "extracurriculars": {
    "sports": ["Football", "Basketball", "Soccer"],
    "clubs": ["Debate", "Robotics", "Yearbook"]
  }
}
```

The AI will still be able to answer questions about these if they're in the JSON (or in PDFs).

### Example 3: State-Specific Requirements

Add state-specific sections:

```json
{
  "state_requirements": {
    "required_tests": "Your state tests",
    "civics_requirement": "Your state's civics requirement"
  }
}
```

---

## ✅ Validation

The code will:
- ✅ Work even if sections are missing
- ✅ Handle missing fields gracefully (shows "N/A")
- ✅ Process whatever structure you provide
- ✅ Combine JSON + PDFs seamlessly (after hybrid implementation)

---

## 🚀 Ready to Customize?

1. Open `data/glhs_info.json`
2. Replace all the example data with your school's data
3. Save the file
4. Delete `chroma_db/` folder (if it exists)
5. Restart the app

That's it! The AI will now answer questions based on YOUR school's data.

---

## Need Help?

The JSON structure is just a template - modify it however fits your school. The AI is smart enough to understand the context regardless of the exact structure you use!

