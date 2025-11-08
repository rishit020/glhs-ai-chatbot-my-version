# 🔑 OpenAI API Key Setup Guide

## ⚠️ CRITICAL: Where to Put Your API Key

**The OpenAI API key goes ONLY in the `.env` file. Nowhere else!**

---

## Step-by-Step Instructions

### 1. Get Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (it starts with `sk-`)

### 2. Create the `.env` File

1. Navigate to the `glhs-chatbot/` folder (where `app.py` is located)
2. Create a new file named exactly `.env` (not `.env.txt` or `env.txt`)
3. Add this single line to the file:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```
4. Replace `sk-your-actual-key-here` with your actual key from step 1
5. Save the file

### 3. Verify the File Location

Your file structure should look like this:
```
glhs-chatbot/
├── .env                    ← Your API key goes HERE
├── app.py
├── chatbot.py
├── requirements.txt
└── ...
```

---

## ✅ Do's and Don'ts

### ✅ DO:
- Put your API key in `.env` file only
- Keep the `.env` file in the `glhs-chatbot/` folder
- Use the exact format: `OPENAI_API_KEY=sk-your-key`
- Keep your API key secret and private

### ❌ DON'T:
- Put your API key in `app.py` or any Python file
- Put your API key in `chatbot.py` or any code file
- Hardcode the key anywhere in the source code
- Commit the `.env` file to git (it's already in `.gitignore`)
- Share your `.env` file or API key with anyone
- Put quotes around the key: `OPENAI_API_KEY="sk-..."` ← Wrong!

---

## File Format Example

Your `.env` file should look exactly like this:

```
OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234
```

**Notes:**
- No quotes around the key
- No spaces before or after the `=` sign
- The key starts with `sk-` or `sk-proj-`
- One line only

---

## Testing Your Setup

After creating the `.env` file:

1. Run the app: `python app.py`
2. You should see: `Chatbot initialized successfully`
3. You should NOT see: `OPENAI_API_KEY not found in environment variables!`

If you see an error, check:
- File is named exactly `.env` (not `.env.txt`)
- File is in `glhs-chatbot/` folder (same as `app.py`)
- Key format is correct: `OPENAI_API_KEY=sk-...`
- No extra spaces or quotes

---

## Security Reminders

- ✅ The `.env` file is already in `.gitignore` - your key won't be committed
- ✅ Never share your API key publicly
- ✅ Never put your key in code files
- ✅ If your key is leaked, revoke it immediately at https://platform.openai.com/api-keys

---

## Troubleshooting

**Error: "OPENAI_API_KEY not found"**
- Check file is named `.env` (with the dot at the start)
- Check file is in `glhs-chatbot/` folder
- Check the format: `OPENAI_API_KEY=sk-your-key` (no quotes, no spaces)

**Error: "Invalid API key"**
- Verify you copied the entire key
- Check there are no extra spaces
- Make sure your OpenAI account is active

**Still having issues?**
- Double-check the file location matches the structure above
- Verify `python-dotenv` is installed: `pip install python-dotenv`
- Restart your terminal/IDE after creating the `.env` file

