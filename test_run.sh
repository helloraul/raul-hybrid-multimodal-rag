#!/bin/bash
# test_run.sh

echo "� Running Python virtual environment setup..."
source venv/Scripts/activate  # Use 'source venv/bin/activate' on Mac/Linux

echo "� Running unit tests with pytest..."
python -m pytest tests/

echo "� Running main script..."
python src/main.py sample_docs/sample1.pdf "What is the main content of this document?"
