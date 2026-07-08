"""
End-to-End Pipeline Integration Test for Medical Symptom Triage Chatbot.
Simulates clinical triage queries through the full pipeline:
Safety pre-filter -> NER Extraction -> RAG Query -> Triage Classifier -> LLM/Mock Generator -> Safety post-disclaimer.
"""
import os
import sys
import json

# Fix Windows console encoding for Unicode/emoji output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.safety.safety_filter import SafetyFilter
from src.ner.ner_extractor import MedicalNERExtractor
from src.rag.rag_pipeline import MedicalRAGPipeline
from src.triage.triage_classifier import TriageClassifier
from src.llm.llm_handler import MedicalLLMHandler

def run_integration_pipeline(query: str):
    print(f"\n========================================")
    print(f"PATIENT QUERY: '{query}'")
    print(f"========================================")
    
    # 1. Safety Pre-Filter
    safety = SafetyFilter()
    lang = safety.detect_language(query)
    print(f"[Safety] Detected Language: {lang.upper()}")
    
    if safety.detect_immediate_emergency(query):
        print("[Safety] *** EMERGENCY OVERRIDE TRIGGERED! ***")
        response_dict = safety.get_emergency_response(lang)
        final_text = response_dict["action_fr"] if lang == "fr" else response_dict["action_en"]
        final_text_with_disclaimer = safety.inject_disclaimer(final_text, lang)
        
        print("\n--- FINAL CHATBOT RESPONSE (SAFETY OVERRIDE) ---")
        print(final_text_with_disclaimer)
        return
        
    print("[Safety] No immediate emergency keywords triggered. Proceeding to pipeline.")
    
    # 2. NER Entity Extraction
    print("[NER] Extracting entities...")
    ner = MedicalNERExtractor()
    entities = ner.extract_symptoms(query)
    print(f"[NER] Extracted Symptoms: {entities['symptoms']}")
    print(f"[NER] Duration: {entities['duration']} | Intensity: {entities['intensity']} | Location: {entities['location']}")
    
    # 3. RAG Retrieval
    print("[RAG] Querying vector knowledge base...")
    rag = MedicalRAGPipeline()
    suspected_diseases = rag.search_conditions(query, entities["symptoms"], k=3)
    
    print("[RAG] Top Candidate Pathologies:")
    for cond in suspected_diseases:
        print(f"  - {cond['disease_name']} ({cond['icd10']}) - Conf: {cond['score']}")
        
    # 4. Triage Classification
    print("[Triage] Classifying case priority...")
    triage = TriageClassifier()
    triage_result = triage.classify(entities, suspected_diseases)
    print(f"[Triage] Assigned Level: {triage_result['triage_level']} ({triage_result['label_en']})")
    print(f"[Triage] Recommended Action: {triage_result['action_en']}")
    
    # 5. LLM Response Generation
    print("[LLM] Generating clinical explanation...")
    llm = MedicalLLMHandler()
    explanation = llm.generate_response(query, entities, triage_result, suspected_diseases, lang)
    
    # 6. Safety Post-Disclaimer
    print("[Safety] Injecting mandatory medical disclaimer...")
    final_output = safety.inject_disclaimer(explanation, lang)
    
    print("\n--- FINAL CHATBOT RESPONSE ---")
    print(final_output)
    print("========================================\n")

if __name__ == "__main__":
    # Test Scenario 1: Immediate P1 Safety Override
    run_integration_pipeline("I have crushing chest pain and I feel like I'm having a heart attack!")
    
    # Test Scenario 2: Acute P3 Infectious Condition (English)
    run_integration_pipeline("I have a high fever and chills for 3 days and my joints hurt")
    
    # Test Scenario 3: Banal P5 Self-Care (French)
    run_integration_pipeline("J'ai le nez qui coule, je tousse un peu et j'ai mal a la gorge depuis hier")

