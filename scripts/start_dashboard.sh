#!/usr/bin/env bash
# 启动可视化管理系统后端（FastAPI）。
# 本机（mock 模式）或开发板均可运行。
set -e
cd "$(dirname "$0")/../web/backend"

PORT="${1:-8000}"
echo "启动 Web 后端：http://127.0.0.1:${PORT}/"
uvicorn app:app --host 0.0.0.0 --port "${PORT}" --reload
