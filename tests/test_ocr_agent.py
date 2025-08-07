import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from agents.ocr_agent import ocr_and_route

def test_ocr_and_route():
    output = ocr_and_route("dummy_path.pdf")
    assert output["type"] == "pdf"
