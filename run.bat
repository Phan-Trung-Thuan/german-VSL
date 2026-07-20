@echo off
echo.
echo  =====================================================
echo   German Speech ^-^> Sign Language Pipeline
echo  =====================================================
echo.
echo  Starting backend server...
echo  Open: http://127.0.0.1:8000
echo.
conda activate canary && python backend\main.py
pause
