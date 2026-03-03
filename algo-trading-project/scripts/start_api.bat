@echo off
REM Batch script to start the FastAPI server

echo ====================================
echo Starting FastAPI Server
echo ====================================
echo.

cd /d "%~dp0.."

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting API on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

