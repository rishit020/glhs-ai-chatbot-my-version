@echo off
echo ========================================
echo DEPENDENCY VERIFICATION CHECKLIST
echo ========================================
echo.

echo [1/4] Checking for dependency conflicts...
pip check
if %errorlevel% equ 0 (
    echo ✅ No conflicts found!
) else (
    echo ⚠️ Conflicts detected - see above
)
echo.

echo [2/4] Testing package imports...
python -c "import langchain; import langchain_community; import langchain_core; import chromadb; print('✅ All imports successful!')"
if %errorlevel% equ 0 (
    echo ✅ Imports working!
) else (
    echo ❌ Import errors detected
)
echo.

echo [3/4] Checking installed versions...
pip list | findstr "langchain chromadb"
echo.

echo [4/4] Testing chatbot initialization...
cd glhs-chatbot
python -c "from chatbot import get_chatbot; chatbot = get_chatbot(); print('✅ Chatbot initialized successfully!')"
if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo ✅ ALL CHECKS PASSED!
    echo Your dependencies are fixed and ready!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo ⚠️ CHATBOT INITIALIZATION FAILED
    echo Check the error message above
    echo ========================================
)
echo.
pause

