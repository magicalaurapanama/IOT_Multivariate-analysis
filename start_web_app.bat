@echo off
echo ========================================
echo    IOT Anomaly Detection Web App
echo ========================================
echo.
echo Starting the web application...
echo.
echo The app will open in your web browser at:
echo http://localhost:8501
echo.
echo To stop the server, close this window or press Ctrl+C
echo.
echo ========================================

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Launch the Streamlit app
python launch_app.py

pause
