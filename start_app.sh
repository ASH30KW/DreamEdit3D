#!/bin/bash
# Startup script for DreamEdit3D
#
# Requires an OpenAI API key for GPT-4V-based automatic concept naming.
# Set it in your shell before running:
#   export OPENAI_API_KEY="sk-..."
# or create a `.env` file in this directory containing the same line.

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY is not set - GPT-4V auto-naming will be disabled."
fi

echo "============================================================"
echo "  Starting DreamEdit3D App"
echo "============================================================"

python main.py
