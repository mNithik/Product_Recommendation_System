$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Starting Streamlit demo from $projectRoot" -ForegroundColor Cyan
Write-Host "App: app/demo.py" -ForegroundColor Cyan
Write-Host "URL: http://localhost:8501" -ForegroundColor Green

python -m streamlit run app/demo.py --server.headless true --server.port 8501
