"""
Dataset Preprocessing for Medical Triage Chatbot.
Parses DiseaseAndSymptoms.csv and ICD-10 codes, maps diseases,
attaches descriptions/precautions, and generates a structured medical knowledge base JSON.
"""
import os
import json
import pandas as pd

# Define the dictionary of descriptions, precautions, and ICD-10 codes for the 41 diseases.
# These details are compiled from standard medical definitions to enrich the RAG knowledge base.
DISEASE_METADATA = {
    "Fungal infection": {
        "icd10": "B35.9",
        "description": "An inflammatory skin condition caused by a fungus, typically leading to itching, scaling, and skin rashes.",
        "precautions": ["Keep affected area clean and dry", "Use over-the-counter antifungal creams", "Avoid sharing personal items like towels", "Wear breathable cotton clothing"]
    },
    "Allergy": {
        "icd10": "T78.4",
        "description": "An immune system reaction to a foreign substance (allergen) that is typically harmless, causing symptoms like sneezing, itching, or rashes.",
        "precautions": ["Identify and avoid known allergens", "Use antihistamines", "Apply cool compresses to itchy skin", "Seek medical care if breathing is affected"]
    },
    "GERD": {
        "icd10": "K21.9",
        "description": "Gastroesophageal Reflux Disease is a chronic digestive disease where stomach acid flows back into the food pipe, irritating the lining.",
        "precautions": ["Avoid fatty, spicy, or acidic foods", "Do not lie down immediately after eating", "Eat smaller, more frequent meals", "Elevate the head of your bed"]
    },
    "Chronic cholestasis": {
        "icd10": "K83.1",
        "description": "A condition where the flow of bile from the liver is reduced or blocked, leading to jaundice, itching, and potential liver damage.",
        "precautions": ["Consult a gastroenterologist", "Avoid alcohol and liver-toxic substances", "Follow a low-fat diet", "Take prescribed medication for itching"]
    },
    "Drug Reaction": {
        "icd10": "L27.0",
        "description": "An adverse reaction of the skin or body to a medication, ranging from mild rashes to severe systemic reactions.",
        "precautions": ["Stop the suspected medication immediately", "Consult the prescribing doctor", "Use antihistamines for itching", "Seek emergency care if face/throat swells"]
    },
    "Peptic ulcer diseae": {
        "icd10": "K27.9",
        "description": "Sores that develop on the inside lining of the stomach and the upper part of the small intestine, causing burning stomach pain.",
        "precautions": ["Avoid NSAIDs like ibuprofen or aspirin", "Limit spicy foods, caffeine, and alcohol", "Take antacids or acid-reducing medications", "Manage stress levels"]
    },
    "AIDS": {
        "icd10": "B24",
        "description": "Acquired Immunodeficiency Syndrome is a chronic, potentially life-threatening condition caused by the Human Immunodeficiency Virus (HIV).",
        "precautions": ["Follow Antiretroviral Therapy (ART) strictly", "Practice safe sex", "Avoid contact with raw or undercooked foods", "Get regular medical checkups"]
    },
    "Diabetes ": {
        "icd10": "E11.9",
        "description": "A group of diseases that affect how the body uses blood sugar, characterized by high blood glucose levels.",
        "precautions": ["Monitor blood glucose levels regularly", "Follow a balanced, low-glycemic diet", "Engage in regular physical exercise", "Take insulin or oral medications as prescribed"]
    },
    "Gastroenteritis": {
        "icd10": "A09.9",
        "description": "Inflammation of the stomach and intestines, typically resulting from bacterial or viral infection, causing diarrhea and vomiting.",
        "precautions": ["Stay hydrated with oral rehydration solutions", "Follow the BRAT diet (Bananas, Rice, Applesauce, Toast)", "Wash hands frequently", "Avoid dairy, caffeine, and fatty foods"]
    },
    "Bronchial Asthma": {
        "icd10": "J45.9",
        "description": "A chronic condition that inflames and narrows the airways, causing difficulty breathing, chest tightness, wheezing, and coughing.",
        "precautions": ["Use prescribed rescue and control inhalers", "Avoid known asthma triggers like dust or smoke", "Keep a peak flow meter handy", "Seek emergency help during severe attacks"]
    },
    "Hypertension ": {
        "icd10": "I10",
        "description": "High blood pressure is a common condition where the long-term force of blood against artery walls is high enough to cause health issues.",
        "precautions": ["Reduce salt intake in diet", "Exercise regularly", "Maintain a healthy weight", "Take antihypertensive medication regularly"]
    },
    "Migraine": {
        "icd10": "G43.9",
        "description": "A neurological condition that causes intense, throbbing headaches, often accompanied by nausea, vomiting, and sensitivity to light/sound.",
        "precautions": ["Rest in a dark, quiet room", "Identify and avoid triggers (stress, caffeine, certain foods)", "Apply cold or warm compresses to the head", "Take prescribed migraine medication early"]
    },
    "Cervical spondylosis": {
        "icd10": "M47.8",
        "description": "Age-related wear and tear affecting the spinal disks in your neck, which can cause pain and stiffness.",
        "precautions": ["Do neck exercises to improve flexibility", "Maintain good posture while sitting or standing", "Use an orthopedic neck pillow", "Apply heat/cold packs to the neck"]
    },
    "Paralysis (brain hemorrhage)": {
        "icd10": "I61.9",
        "description": "Loss of muscle function in part of the body, here specifically caused by bleeding in the brain (intracerebral hemorrhage). A medical emergency.",
        "precautions": ["Call emergency medical services immediately", "Do not move the patient unnecessarily", "Keep patient calm and lying down", "Note the exact time symptoms started"]
    },
    "Jaundice": {
        "icd10": "R17",
        "description": "A yellow staining of the skin and sclerae (whites of the eyes) caused by high levels of bilirubin in the blood.",
        "precautions": ["Consult a doctor to determine the underlying cause", "Avoid alcohol completely", "Stay well hydrated", "Eat a low-fat, liver-friendly diet"]
    },
    "Malaria": {
        "icd10": "B54",
        "description": "A serious and sometimes fatal disease caused by a parasite that commonly infects a certain type of mosquito which feeds on humans.",
        "precautions": ["Take prescribed antimalarial medications fully", "Use mosquito nets and repellents", "Wear long-sleeved clothing outdoors", "Seek immediate medical attention for high fever"]
    },
    "Chicken pox": {
        "icd10": "B01.9",
        "description": "A highly contagious viral infection causing an itchy, blister-like rash on the skin.",
        "precautions": ["Avoid scratching the blisters (risk of infection/scars)", "Take lukewarm oatmeal baths", "Use calamine lotion to soothe itching", "Isolate from high-risk individuals"]
    },
    "Dengue": {
        "icd10": "A90",
        "description": "A mosquito-borne viral disease causing a severe flu-like illness, and sometimes a potentially lethal complication called dengue hemorrhagic fever.",
        "precautions": ["Stay hydrated to replace fluids lost to fever", "Avoid aspirin or ibuprofen (risk of bleeding); use paracetamol instead", "Rest as much as possible", "Protect against further mosquito bites"]
    },
    "Typhoid": {
        "icd10": "A01.0",
        "description": "A bacterial infection caused by Salmonella typhi, transmitted through contaminated food or water, causing high fever, weakness, and abdominal pain.",
        "precautions": ["Take prescribed antibiotics for the full duration", "Drink boiled or bottled water only", "Eat thoroughly cooked hot foods", "Wash hands thoroughly with soap"]
    },
    "hepatitis A": {
        "icd10": "B15.9",
        "description": "A highly contagious liver infection caused by the hepatitis A virus, typically spread through contaminated food or water.",
        "precautions": ["Get plenty of bed rest", "Eat small, high-calorie meals", "Avoid alcohol and liver-taxing drugs", "Practice strict hand hygiene"]
    },
    "Hepatitis B": {
        "icd10": "B16.9",
        "description": "A serious liver infection caused by the hepatitis B virus, transmitted through contact with infectious body fluids.",
        "precautions": ["Consult a doctor for antiviral treatment options", "Avoid alcohol to protect the liver", "Practice safe sex and do not share needles", "Ensure family members are vaccinated"]
    },
    "Hepatitis C": {
        "icd10": "B18.2",
        "description": "An infection caused by the hepatitis C virus that attacks the liver and leads to inflammation, transmitted through blood-to-blood contact.",
        "precautions": ["Consult a specialist for modern direct-acting antivirals", "Avoid alcohol", "Do not share razors or toothbrushes", "Monitor liver health regularly"]
    },
    "Hepatitis D": {
        "icd10": "B17.0",
        "description": "A liver disease caused by the hepatitis D virus, which only occurs in people who are already infected with hepatitis B.",
        "precautions": ["Receive specialized medical supervision", "Avoid alcohol completely", "Prevent transmission to others", "Manage hepatitis B infection actively"]
    },
    "Hepatitis E": {
        "icd10": "B17.2",
        "description": "A liver disease caused by the hepatitis E virus, usually transmitted via the fecal-oral route through contaminated water.",
        "precautions": ["Ensure drinking water is safe/boiled", "Rest and maintain adequate nutrition", "Avoid alcohol", "Pregnant women must seek immediate medical care"]
    },
    "Alcoholic hepatitis": {
        "icd10": "K70.1",
        "description": "Inflammation of the liver caused by drinking alcohol. Continued drinking can lead to cirrhosis and liver failure.",
        "precautions": ["Stop drinking alcohol completely and permanently", "Follow a high-protein, nutrient-rich diet", "Seek therapy or support groups for addiction", "Consult a hepatologist"]
    },
    "Tuberculosis": {
        "icd10": "A15.9",
        "description": "A potentially serious infectious disease that mainly affects the lungs, caused by the bacterium Mycobacterium tuberculosis.",
        "precautions": ["Take all prescribed anti-TB medications for 6+ months without fail", "Stay in a well-ventilated room", "Cover your mouth when coughing or sneezing", "Get regular sputum examinations"]
    },
    "Common Cold": {
        "icd10": "J00",
        "description": "A viral infection of your nose and throat (upper respiratory tract). It's usually harmless.",
        "precautions": ["Drink plenty of warm fluids (tea, broth)", "Get sufficient rest", "Use saline nasal sprays or rinses", "Take paracetamol for body aches"]
    },
    "Pneumonia": {
        "icd10": "J18.9",
        "description": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus, causing cough, fever, and difficulty breathing.",
        "precautions": ["Take prescribed antibiotics or antivirals fully", "Use a humidifier or take warm baths", "Avoid smoking or secondhand smoke", "Get plenty of rest and drink fluids"]
    },
    "Dimorphic hemmorhoids(piles)": {
        "icd10": "I84.9",
        "description": "Swollen veins in your anus and lower rectum, similar to varicose veins. Can cause pain, itching, and bleeding during bowel movements.",
        "precautions": ["Eat high-fiber foods (fruits, vegetables, grains)", "Drink plenty of water", "Avoid straining during bowel movements", "Take warm sitz baths"]
    },
    "Heart attack": {
        "icd10": "I21.9",
        "description": "A medical emergency when blood flow to a part of the heart muscle is blocked. Immediate treatment is vital to prevent heart damage.",
        "precautions": ["Call emergency services (15, 112, 911) immediately", "Chew an aspirin if recommended by emergency dispatchers", "Sit down and stay calm while waiting for help", "Be prepared for CPR if patient loses consciousness"]
    },
    "Varicose veins": {
        "icd10": "I83.9",
        "description": "Gnarled, enlarged veins, most commonly in the legs and feet, caused by weak or damaged vein walls and valves.",
        "precautions": ["Avoid standing or sitting for long periods", "Wear compression stockings", "Elevate your legs when resting", "Exercise regularly to improve leg circulation"]
    },
    "Hypothyroidism": {
        "icd10": "E03.9",
        "description": "A condition in which the thyroid gland doesn't produce enough thyroid hormone, slowing down the metabolism.",
        "precautions": ["Take prescribed thyroid hormone replacement (levothyroxine) daily", "Have thyroid levels checked regularly", "Eat a balanced, fiber-rich diet", "Report signs of dosage issues (fatigue vs heart racing) to doctor"]
    },
    "Hyperthyroidism": {
        "icd10": "E05.9",
        "description": "The production of too much thyroid hormone by the thyroid gland, which can accelerate your body's metabolism.",
        "precautions": ["Take anti-thyroid medications as prescribed", "Follow a low-iodine diet if recommended", "Monitor heart rate and weight", "Seek medical advice for eye changes or severe anxiety"]
    },
    "Hypoglycemia": {
        "icd10": "E16.2",
        "description": "A condition characterized by an abnormally low level of blood sugar (glucose), requiring immediate sugar intake.",
        "precautions": ["Consume 15g of fast-acting sugar (fruit juice, candy) immediately", "Check blood glucose after 15 minutes", "Eat a snack with complex carbs and protein afterwards", "Carry glucose tablets at all times"]
    },
    "Osteoarthristis": {
        "icd10": "M19.9",
        "description": "The most common form of arthritis, featuring wear and tear of joint cartilage, leading to pain, stiffness, and swelling.",
        "precautions": ["Engage in low-impact exercise (swimming, cycling)", "Maintain a healthy weight to reduce joint load", "Apply heat for stiffness and cold for swelling", "Use assistive devices if necessary"]
    },
    "Arthritis": {
        "icd10": "M13.9",
        "description": "Inflammation of one or more joints, causing pain and stiffness that can worsen with age.",
        "precautions": ["Stay active with gentle stretching and range-of-motion exercises", "Use anti-inflammatory diets or supplements", "Consult a rheumatologist for diagnosis", "Protect joints from excessive strain"]
    },
    "(vertigo) Paroymsal  Positional Vertigo": {
        "icd10": "H81.1",
        "description": "Benign Paroxysmal Positional Vertigo (BPPV) is a disorder of the inner ear that causes brief episodes of severe spinning sensations (vertigo).",
        "precautions": ["Avoid sudden neck or head movements", "Sit down immediately when feeling dizzy", "Perform Epley maneuver (if instructed by a professional)", "Avoid sleeping on the affected side"]
    },
    "Acne": {
        "icd10": "L70.9",
        "description": "A skin condition that occurs when hair follicles become plugged with oil and dead skin cells, causing pimples, blackheads, or whiteheads.",
        "precautions": ["Wash face twice daily with a gentle cleanser", "Avoid squeezing or popping pimples", "Use non-comedogenic (pore-clogging) cosmetics", "Consult a dermatologist for topical/oral treatments"]
    },
    "Urinary tract infection": {
        "icd10": "N39.0",
        "description": "An infection in any part of the urinary system, usually the kidneys, ureters, bladder, or urethra, causing burning pain during urination.",
        "precautions": ["Drink plenty of water to flush out bacteria", "Urinate after sexual intercourse", "Wipe from front to back after using the toilet", "Take the full course of prescribed antibiotics"]
    },
    "Psoriasis": {
        "icd10": "L40.9",
        "description": "A skin disease that causes red, itchy, scaly patches, most commonly on the knees, elbows, trunk, and scalp.",
        "precautions": ["Keep skin well-moisturized", "Avoid triggers like stress, skin injuries, and smoking", "Expose skin to moderate, controlled sunlight", "Use prescribed topical creams or systemic therapies"]
    },
    "Impetigo": {
        "icd10": "L01.0",
        "description": "A highly contagious bacterial skin infection causing sores and blisters, common in children.",
        "precautions": ["Wash sores with soap and water; cover loosely", "Wash infected person's clothes/linen separately", "Keep fingernails cut short", "Apply prescribed antibiotic ointment"]
    }
}

def clean_symptom(s):
    """Normalize symptom string: lowercase, strip, replace spaces/dashes with underscores."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = s.replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def preprocess_datasets():
    print("=== Processing raw datasets ===")
    
    # 1. Load symptoms CSV
    csv_path = os.path.join("data", "raw", "disease_symptom", "DiseaseAndSymptoms.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Extract mappings
    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
    
    disease_symptoms = {}
    all_unique_symptoms = set()
    
    for _, row in df.iterrows():
        raw_disease = row["Disease"]
        # Standardize disease name matching metadata keys
        disease = raw_disease.strip()
        
        # Collect symptoms for this row
        row_symptoms = []
        for col in symptom_cols:
            val = row[col]
            if pd.notna(val):
                clean_sym = clean_symptom(val)
                if clean_sym:
                    row_symptoms.append(clean_sym)
                    all_unique_symptoms.add(clean_sym)
        
        if disease not in disease_symptoms:
            disease_symptoms[disease] = set()
        disease_symptoms[disease].update(row_symptoms)
        
    print(f"Found {len(disease_symptoms)} unique diseases in CSV.")
    print(f"Found {len(all_unique_symptoms)} unique symptoms in CSV.")
    
    # Combine CSV data with medical metadata
    processed_knowledge = {}
    
    for disease_name, symptoms in disease_symptoms.items():
        # Match against our metadata (case-insensitive or exact)
        meta = DISEASE_METADATA.get(disease_name)
        if not meta:
            # Fallback if spelling is slightly off
            matches = [k for k in DISEASE_METADATA.keys() if k.lower().strip() == disease_name.lower().strip()]
            if matches:
                meta = DISEASE_METADATA[matches[0]]
            else:
                meta = {
                    "icd10": "U00.0",  # Unspecified fallback code
                    "description": f"Condition characterized by: {', '.join(symptoms).replace('_', ' ')}.",
                    "precautions": ["Consult a medical professional for advice."]
                }
                
        processed_knowledge[disease_name] = {
            "disease_name": disease_name,
            "icd10": meta["icd10"],
            "description": meta["description"],
            "precautions": meta["precautions"],
            "symptoms": sorted(list(symptoms))
        }
        
    # Ensure processed directory exists
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    output_path = os.path.join(processed_dir, "medical_knowledge.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_knowledge, f, indent=4, ensure_ascii=False)
        
    print(f"[OK] Saved processed knowledge base of {len(processed_knowledge)} diseases to {output_path}")
    
    # Save a flat list of symptoms for the NER dictionary
    symptoms_list_path = os.path.join(processed_dir, "symptoms_list.json")
    with open(symptoms_list_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(all_unique_symptoms)), f, indent=4, ensure_ascii=False)
    print(f"[OK] Saved list of {len(all_unique_symptoms)} symptoms to {symptoms_list_path}")

if __name__ == "__main__":
    preprocess_datasets()
