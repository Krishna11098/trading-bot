@echo off
REM Streamlit App Launcher for AlgoTrading Bot

echo ========================================
echo AlgoTrading Bot - Streamlit Dashboard
echo ========================================
echo.
echo Starting Streamlit application...
echo Access the dashboard at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Note: Virtual environment not found
)

REM Run Streamlit app
streamlit run app.py

pause
