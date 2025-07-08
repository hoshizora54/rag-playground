#!/usr/bin/env python3
"""Запуск Streamlit интерфейса для RAG системы."""

import sys
import os
import subprocess

# Добавляем текущую директорию в путь Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Запуск Streamlit приложения."""
    app_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_ui.py")
    
    # Запускаем streamlit с нужными параметрами
    cmd = [
        "streamlit", "run", app_file,
        "--server.port", "7860",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 