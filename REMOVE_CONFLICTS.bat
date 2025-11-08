@echo off
echo ========================================
echo REMOVING CONFLICTING PACKAGES
echo ========================================
echo.
echo Removing packages that require langchain-core>=1.0.0...
echo (These are not needed for the chatbot)
echo.

pip uninstall langchain-classic langgraph langgraph-prebuilt langgraph-checkpoint langgraph-sdk langchain-ollama -y

echo.
echo Verifying no conflicts remain...
pip check

echo.
if %errorlevel% equ 0 (
    echo ✅ SUCCESS! All conflicts resolved!
    echo.
    echo Run 'pip check' again to verify.
) else (
    echo ⚠️ Some conflicts may remain. Check output above.
)

pause

