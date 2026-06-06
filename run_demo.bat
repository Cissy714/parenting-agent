@echo off
echo ========================================
echo   育儿智能助手 Demo
echo ========================================
echo.
echo 正在启动 Streamlit 应用...
echo 打开浏览器访问 http://localhost:8501
echo 按 Ctrl+C 停止
echo.

streamlit run demo.py --server.port 8501
