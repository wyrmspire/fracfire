#!/bin/bash

# Use venv python if available, else system python
if [ -f ".venv312/Scripts/python.exe" ]; then
    PYTHON_CMD=".venv312/Scripts/python.exe"
else
    PYTHON_CMD="python"
fi

echo "Using python: $PYTHON_CMD"
$PYTHON_CMD scripts/generate_code_dump.py
