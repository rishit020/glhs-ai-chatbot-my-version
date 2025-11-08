@echo off
echo Fixing dependency conflicts...
echo.

echo Step 1: Removing unnecessary packages that cause conflicts...
pip uninstall langchain-classic langgraph-prebuilt langgraph langgraph-checkpoint langgraph-sdk langchain-ollama -y 2>nul

echo.
echo Step 2: Installing compatible langchain versions...
pip install "langchain-core==0.3.76" "langchain==0.3.25" "langchain-community==0.3.25" "langchain-text-splitters==0.3.11" --force-reinstall

echo.
echo Step 3: Installing all requirements...
pip install -r requirements.txt

echo.
echo Step 4: Checking for remaining conflicts...
pip check

echo.
echo Done! Dependency conflicts should be resolved.
echo.
echo Testing installation...
python -c "import langchain; import langchain_community; import chromadb; print('All imports successful!')"

pause

