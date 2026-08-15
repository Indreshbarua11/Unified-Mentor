$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

& "$ProjectRoot\.venv\Scripts\python.exe" -m streamlit run "$ProjectRoot\app.py" --server.port 8501 --server.headless true
