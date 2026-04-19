@echo off
setlocal
cd /d "%~dp0"
echo Starting Streamlit demo from %cd%
echo App: app\demo.py
echo URL: http://localhost:8501
python -m streamlit run app/demo.py --server.headless true --server.port 8501
endlocal
