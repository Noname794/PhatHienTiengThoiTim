@echo off
echo ========================================
echo Heart Sound Classifier - Setup
echo ========================================
echo.

echo [1/3] Setting up Backend...
cd backend

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -r requirements.txt

echo Backend setup complete!
cd ..

echo.
echo [2/3] Setting up Frontend...
cd frontend

echo Installing Node.js dependencies...
call npm install

echo Frontend setup complete!
cd ..

echo.
echo [3/3] Verifying model file...
if exist "results\models\1dcnn_method_best.h5" (
    echo Model file found!
) else (
    echo WARNING: Model file not found at results\models\1dcnn_method_best.h5
    echo Please ensure the model file exists before running the application.
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application, run: start.bat
echo Or manually:
echo   1. Backend: cd backend ^&^& python app.py
echo   2. Frontend: cd frontend ^&^& npm start
echo.
pause
