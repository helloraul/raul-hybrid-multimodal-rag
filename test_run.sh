#!/bin/bash
# test_run.sh

echo "í´„ Running Python virtual environment setup..."
source venv/Scripts/activate  # Use 'source venv/bin/activate' on Mac/Linux

echo "í·ª Running unit tests with pytest..."
python -m pytest tests/

echo "íº€ Running main script..."
python src/main.py sample_docs/sample1.pdf "What is the main content of this document?"
