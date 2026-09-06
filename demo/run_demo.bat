@echo off
echo ============================================================
echo   DeadReckon - AI-Powered Navigation Demo
echo ============================================================
echo.

:: Navigate to demo directory
cd /d "%~dp0"

:: Install backend dependencies if needed
echo [1/3] Checking dependencies...
pip install fastapi uvicorn --quiet 2>nul

:: Start backend server
echo [2/3] Starting backend server...
cd backend
start "DeadReckon Backend" cmd /k "python -m uvicorn server:app --host 0.0.0.0 --port 8000"

:: Wait for server to start
echo [3/3] Waiting for server to start...
timeout /t 4 /nobreak >nul

:: Open browser
echo.
echo ============================================================
echo   Demo is running!
echo   Open: http://localhost:8000
echo ============================================================
echo.
start http://localhost:8000

echo Press any key to stop the demo server...
pause >nul
taskkill /FI "WindowTitle eq DeadReckon Backend*" >nul 2>nul
