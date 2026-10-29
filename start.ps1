$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$assistantPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $assistantPython)) {
    throw '请先按 README 创建 .venv 并安装 requirements.txt。'
}
& $assistantPython -m streamlit run "$PSScriptRoot\qwen_agent_demo.py" --server.address 127.0.0.1 --server.port 8504 --server.headless true --browser.gatherUsageStats false --client.toolbarMode viewer
