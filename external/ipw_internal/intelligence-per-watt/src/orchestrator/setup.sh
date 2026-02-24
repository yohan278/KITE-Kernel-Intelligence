#!/bin/bash
# Create a new virtual environment
python -m venv new_env
source new_env/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
