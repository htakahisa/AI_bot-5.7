@echo off
rem 1. イメージのビルド
docker build -t ue5-ai-train .

rem 2. コンテナの実行（%cd% に修正）
docker run -it --rm --gpus all -v "%cd%":/app ue5-ai-train

pause