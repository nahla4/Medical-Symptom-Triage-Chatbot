"""
Unit Tests for Triage Classifier Module.
Validates P1-P5 triage level assignments based on clinical parameters and RAG suggestions.
"""
import pytest
from src.triage.triage_classifier import TriageClassifier

@pytest.fixture
def classifier():
    return TriageClassifier()

def test_triage_p1_critical_symptoms(classifier):
    # Test red flag symptom triggers (e.g. chest pain)
    entities = {
        "symptoms": ["chest_pain"],
        "intensity": "severe",
        "duration": "15 minutes",
        "location": "chest"
    }
    suspected_diseases = [{"disease_name": "Heart attack"}]
    
    result = classifier.classify(entities, suspected_diseases)
    assert result["triage_level"] == "P1"
    assert "SAMU 15" in result["action_fr"]
    assert "911" in result["action_en"]

def test_triage_p2_meningitis_suspicion(classifier):
    # Test meningitis suspect (stiff neck + high fever)
    entities = {
        "symptoms": ["stiff_neck", "high_fever", "headache"],
        "intensity": "severe",
        "duration": "1 day",
        "location": "head"
    }
    # RAG suggests something, but triage rules should catch meningitis
    suspected_diseases = [{"disease_name": "Migraine"}]
    
    result = classifier.classify(entities, suspected_diseases)
    assert result["triage_level"] == "P2"
    assert "urgences" in result["action_fr"].lower()

def test_triage_p2_severe_abdominal_pain(classifier):
    # Test P2 severe pain in critical location
    entities = {
        "symptoms": ["abdominal_pain"],
        "intensity": "severe",
        "duration": "2 hours",
        "location": "abdomen"
    }
    suspected_diseases = [{"disease_name": "Peptic ulcer diseae"}]
    
    result = classifier.classify(entities, suspected_diseases)
    assert result["triage_level"] == "P2"

def test_triage_p3_infectious_diseases(classifier):
    # Test P3 for tropical/acute infections (e.g. Malaria)
    entities = {
        "symptoms": ["high_fever", "chills", "sweating"],
        "intensity": "moderate",
        "duration": "3 days",
        "location": None
    }
    suspected_diseases = [{"disease_name": "Malaria"}]
    
    result = classifier.classify(entities, suspected_diseases)
    assert result["triage_level"] == "P3"
    assert "24 heures" in result["action_fr"]

def test_triage_p4_chronic_condition(classifier):
    # Test P4 for chronic stable diseases (e.g. GERD)
    entities = {
        "symptoms": ["acidity", "vomiting"],
        "intensity": "mild",
        "duration": "2 weeks",
        "location": "abdomen"
    }
    suspected_diseases = [{"disease_name": "GERD"}]
    
    result = classifier.classify(entities, suspected_diseases)
    assert result["triage_level"] == "P4"
    assert "48 à 72 heures" in result["action_fr"]

def test_triage_p5_minor_cold(classifier):
    # Test P5 for basic cold symptoms
    entities = {
        "symptoms": ["cough", "runny_nose", "congestion"],
        "intensity": "mild",
        "duration": "2 days",
        "location": "throat"
    }
    suspected_diseases = [{"disease_name": "Common Cold"}]
    
    result = classifier.classify(entities, suspected_diseases)
    assert result["triage_level"] == "P5"
    assert "pharmacien" in result["action_fr"]
