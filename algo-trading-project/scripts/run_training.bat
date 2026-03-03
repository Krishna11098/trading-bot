@echo off
REM Batch script to run model training

echo ====================================
echo Starting Model Training Pipeline
echo ====================================
echo.

cd /d "%~dp0.."

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Running training script...
python src\modeling\train_model.py

echo.
echo Training complete!
echo ====================================
pause

