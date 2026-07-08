"""
conftest.py - pytest configuration for the Medical Symptom Triage Chatbot project.
Ensures the project root is in sys.path so `src.*` imports work correctly.
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))
