@echo off
REM Batch script to run backtesting

echo ====================================
echo Starting Backtest
echo ====================================
echo.

cd /d "%~dp0.."

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Running backtest...
python src\strategy\backtest.py

echo.
echo Backtest complete!
echo ====================================
pause

