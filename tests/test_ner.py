"""
Unit Tests for Named Entity Recognition (NER) Module.
Tests symptom, duration, intensity, and location extraction for English and French inputs.
The HuggingFace NER pipeline is mocked to prevent slow model downloads during testing.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.ner.ner_extractor import MedicalNERExtractor


@pytest.fixture
def extractor():
    """Create a NER extractor with the HuggingFace pipeline mocked out."""
    with patch("src.ner.ner_extractor.MedicalNERExtractor._init_deep_learning_ner") as mock_init:
        mock_init.return_value = None  # Skip model download
        ext = MedicalNERExtractor()
        ext.ner_pipeline = None  # Ensure rule-based mode
        return ext


def test_english_symptom_extraction(extractor):
    # Test matching standard English symptom strings
    text = "I have a severe headache and high fever since yesterday"
    result = extractor.extract_symptoms(text)

    assert "headache" in result["symptoms"]
    assert "high_fever" in result["symptoms"]
    assert result["duration"] == "since yesterday"
    assert result["intensity"] == "severe"
    assert result["location"] == "head"


def test_french_symptom_mapping(extractor):
    # Test bilingual French to English symptom mapping
    text = "J'ai de fortes démangeaisons et des boutons sur la peau depuis 3 jours"
    result = extractor.extract_symptoms(text)

    assert "itching" in result["symptoms"]
    assert "nodal_skin_eruptions" in result["symptoms"]
    assert result["duration"] == "depuis 3 jours"
    assert result["intensity"] == "severe"  # "fortes" maps to severe
    assert result["location"] == "skin"


def test_intensity_numeric_scale(extractor):
    # Test parsing of numeric pain scales (e.g. 9/10)
    text = "J'ai une grosse douleur au ventre évaluée à 9/10"
    result = extractor.extract_symptoms(text)

    assert "abdominal_pain" in result["symptoms"] or "belly_pain" in result["symptoms"]
    assert "9/10" in result["intensity"]
    assert result["location"] == "abdomen"


def test_no_symptoms_matched(extractor):
    # Test input with no medical symptoms
    text = "Hello, I would like to ask a general question about my health."
    result = extractor.extract_symptoms(text)

    assert len(result["symptoms"]) == 0
    assert result["duration"] is None
    assert result["intensity"] is None
    assert result["location"] is None


def test_english_chest_pain(extractor):
    # Test chest pain extraction with severity
    text = "I have severe chest pain and difficulty breathing"
    result = extractor.extract_symptoms(text)

    assert "chest_pain" in result["symptoms"]
    assert "breathlessness" in result["symptoms"]
    assert result["intensity"] == "severe"
    assert result["location"] == "chest"


def test_french_fever_cough(extractor):
    # Test French fever and cough symptoms
    text = "J'ai de la fièvre et une toux depuis 2 jours"
    result = extractor.extract_symptoms(text)

    assert "high_fever" in result["symptoms"]
    assert "cough" in result["symptoms"]
    assert result["duration"] == "depuis 2 jours"
