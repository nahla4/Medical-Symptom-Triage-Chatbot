"""
Dataset Downloader & Verifier for Medical Triage Chatbot.
Checks for the presence of ICD-10 and Kaggle Disease-Symptom datasets.
"""
import os
import sys

def verify_datasets():
    raw_dir = os.path.join("data", "raw")
    disease_symptom_path = os.path.join(raw_dir, "disease_symptom", "DiseaseAndSymptoms.csv")
    icd10_path = os.path.join(raw_dir, "icd10", "icd102019syst_codes.txt")
    
    print("=== Checking datasets ===")
    
    # Check Disease Symptom Dataset
    if os.path.exists(disease_symptom_path):
        print(f"[OK] Disease & Symptom dataset found at: {disease_symptom_path}")
    else:
        print(f"[ERROR] Disease & Symptom dataset NOT found at: {disease_symptom_path}")
        print("Please place the DiseaseAndSymptoms.csv dataset in data/raw/disease_symptom/")
        
    # Check ICD-10 Dataset
    if os.path.exists(icd10_path):
        print(f"[OK] ICD-10 codebook dataset found at: {icd10_path}")
    else:
        print(f"[ERROR] ICD-10 codebook dataset NOT found at: {icd10_path}")
        print("Please place the ICD-10 files in data/raw/icd10/")
        
    if os.path.exists(disease_symptom_path) and os.path.exists(icd10_path):
        print("\nAll required raw datasets are available. Preprocessing can begin.")
        return True
    else:
        print("\nSome raw datasets are missing. Please restore them.")
        return False

if __name__ == "__main__":
    verify_datasets()
