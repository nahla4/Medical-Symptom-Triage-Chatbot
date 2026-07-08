"""
Named Entity Recognition (NER) Module for Medical Triage Chatbot.
Extracts symptoms, duration, intensity, and anatomical location from user input in French or English.
Features a robust regex/dictionary parser with an optional Hugging Face Transformer NER model fallback.
"""
import re
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# French to English symptom mapping (Bilingual support F11)
FRENCH_TO_ENGLISH_SYMPTOMS = {
    # Itching & Skin
    "démangeaison": "itching", "demangeaison": "itching", "démangeaisons": "itching", "demangeaisons": "itching", "gratte": "itching", "gratter": "itching",
    "éruption cutanée": "skin_rash", "eruption cutanee": "skin_rash", "éruptions cutanées": "skin_rash", "eruptions cutanees": "skin_rash", "rougeurs": "skin_rash", "plaques": "skin_rash", "rougeur": "skin_rash",
    "boutons": "nodal_skin_eruptions", "bouton": "nodal_skin_eruptions", "éruptions nodulaires": "nodal_skin_eruptions", "eruptions nodulaires": "nodal_skin_eruptions",
    "plaques décolorées": "dischromic_patches", "plaques decolorees": "dischromic_patches", "taches décolorées": "dischromic_patches",
    "ampoules": "blister", "cloches": "blister", "bulles": "blister",
    "peau qui pèle": "skin_peeling", "peau qui pele": "skin_peeling",
    "squames argentées": "silver_like_dusting", "squames argentees": "silver_like_dusting",
    "plaie rouge autour du nez": "red_sore_around_nose",
    "croûte jaune suintante": "yellow_crust_ooze", "croute jaune suintante": "yellow_crust_ooze",
    "cicatrices": "scurring", "points noirs": "blackheads", "boutons de pus": "pus_filled_pimples",
    
    # General / Fever / Pain
    "frissons": "chills", "frisson": "chills", "froid": "chills", "tremblements": "shivering", "tremblement": "shivering",
    "douleurs articulaires": "joint_pain", "douleur articulaire": "joint_pain", "douleur articulation": "joint_pain",
    "mal à l'estomac": "stomach_pain", "maux d'estomac": "stomach_pain", "douleur estomac": "stomach_pain",
    "acidité": "acidity", "acidite": "acidity", "brûlures d'estomac": "acidity", "brulures destomac": "acidity", "aigreurs": "acidity",
    "aphtes": "ulcers_on_tongue", "ulcères sur la langue": "ulcers_on_tongue", "ulceres sur la langue": "ulcers_on_tongue",
    "fonte musculaire": "muscle_wasting", "perte de muscle": "muscle_wasting",
    "vomissement": "vomiting", "vomissements": "vomiting", "vomir": "vomiting",
    "brûlure en urinant": "burning_micturition", "brulure en urinant": "burning_micturition", "miction douloureuse": "burning_micturition",
    "traces de sang dans les urines": "spotting_urination",
    "fatigue": "fatigue", "fatigué": "fatigue", "fatiguee": "fatigue", "épuisé": "fatigue", "epuise": "fatigue",
    "prise de poids": "weight_gain", "grossir": "weight_gain",
    "anxiété": "anxiety", "anxiete": "anxiety", "angoisse": "anxiety",
    "mains et pieds froids": "cold_hands_and_feets",
    "sautes d'humeur": "mood_swings", "sautes dhumeur": "mood_swings",
    "perte de poids": "weight_loss", "maigrir": "weight_loss",
    "agitation": "restlessness", "léthargie": "lethargy", "lethargie": "lethargy",
    "plaques dans la gorge": "patches_in_throat",
    "taux de sucre irrégulier": "irregular_sugar_level", "taux de sucre irregulier": "irregular_sugar_level",
    "toux": "cough", "tousser": "cough", "rhume": "cough",
    "forte fièvre": "high_fever", "fièvre élevée": "high_fever", "fievre elevee": "high_fever", "température": "high_fever", "fievre": "high_fever", "fièvre": "high_fever",
    "yeux cernés": "sunken_eyes", "yeux cernes": "sunken_eyes", "yeux creux": "sunken_eyes",
    "essoufflement": "breathlessness", "difficulté à respirer": "breathlessness", "difficulte a respirer": "breathlessness", "manque de souffle": "breathlessness", "mal à respirer": "breathlessness",
    "sueur": "sweating", "sueurs": "sweating", "transpiration": "sweating",
    "déshydratation": "dehydration", "deshydratation": "dehydration", "soif intense": "dehydration",
    "indigestion": "indigestion",
    "maux de tête": "headache", "mal à la tête": "headache", "mal a la tete": "headache", "céphalée": "headache", "cephalee": "headache",
    "peau jaune": "yellowish_skin", "jaunisse": "yellowish_skin",
    "urines foncées": "dark_urine", "urines foncees": "dark_urine",
    "nausée": "nausea", "nausee": "nausea", "nausées": "nausea", "nausees": "nausea",
    "perte d'appétit": "loss_of_appetite", "perte dappetit": "loss_of_appetite",
    "douleur derrière les yeux": "pain_behind_the_eyes", "douleur derriere les yeux": "pain_behind_the_eyes",
    "mal de dos": "back_pain", "douleur au dos": "back_pain",
    "constipation": "constipation",
    "douleur abdominale": "abdominal_pain", "douleur abdomen": "abdominal_pain", "mal au ventre": "abdominal_pain", "maux de ventre": "abdominal_pain", "douleur au ventre": "abdominal_pain",
    "diarrhée": "diarrhée", "diarrhee": "diarrhoea",
    "fièvre légère": "mild_fever", "fievre legere": "mild_fever",
    "urines jaunes": "yellow_urine",
    "yeux jaunes": "yellowing_of_eyes",
    "insuffisance hépatique aiguë": "acute_liver_failure", "insuffisance hepatique aigue": "acute_liver_failure",
    "rétention d'eau": "fluid_overload", "retention deau": "fluid_overload", "surcharge en fluide": "fluid_overload",
    "gonflement de l'estomac": "swelling_of_stomach", "gonflement de lestomac": "swelling_of_stomach",
    "ganglions gonflés": "swelled_lymph_nodes", "ganglions gonfles": "swelled_lymph_nodes",
    "malaise": "malaise", "sensation de malaise": "malaise",
    "vision floue": "blurred_and_distorted_vision",
    "crachats": "phlegm", "mucus": "phlegm", "glaires": "phlegm",
    "irritation de la gorge": "throat_irritation", "gorge irritée": "throat_irritation", "gorge irritee": "throat_irritation",
    "yeux rouges": "redness_of_eyes",
    "pression sinusale": "sinus_pressure",
    "nez qui coule": "runny_nose",
    "congestion nasale": "congestion", "nez bouché": "congestion", "nez bouche": "congestion",
    "douleur à la poitrine": "chest_pain", "douleur a la poitrine": "chest_pain", "douleur thoracique": "chest_pain", "mal au thorax": "chest_pain",
    "faiblesse dans les membres": "weakness_in_limbs",
    "rythme cardiaque rapide": "fast_heart_rate", "palpitations": "fast_heart_rate", "palpitation": "fast_heart_rate",
    "douleur lors de la défécation": "pain_during_bowel_movements", "douleur en allant à la selle": "pain_during_bowel_movements",
    "douleur anale": "pain_in_anal_region",
    "sang dans les selles": "bloody_stool",
    "irritation de l'anus": "irritation_in_anus",
    "douleur au cou": "neck_pain", "mal au cou": "neck_pain",
    "vertige": "dizziness", "étourdissement": "dizziness", "etourdissement": "dizziness", "tête qui tourne": "dizziness", "tete qui tourne": "dizziness",
    "crampes": "cramps",
    "ecchymoses": "bruising", "bleus": "bruising",
    "obésité": "obesity", "obesite": "obesity", "surpoids": "obesity",
    "jambes gonflées": "swollen_legs", "jambes gonflees": "swollen_legs",
    "veines gonflées": "swollen_blood_vessels", "veines gonflees": "swollen_blood_vessels",
    "visage et yeux bouffis": "puffy_face_and_eyes",
    "thyroïde hypertrophiée": "enlarged_thyroid", "thyroide hypertrophiee": "enlarged_thyroid",
    "ongles cassants": "brittle_nails",
    "extrémités gonflées": "swollen_extremeties", "extremites gonflees": "swollen_extremeties",
    "faim excessive": "excessive_hunger",
    "contacts extra-conjugaux": "extra_marital_contacts",
    "lèvres sèches et picotements": "drying_and_tingling_lips", "levres seches": "drying_and_tingling_lips",
    "élocution difficile": "slurred_speech", "elocution difficile": "slurred_speech", "parole confuse": "slurred_speech",
    "douleur au genou": "knee_pain",
    "douleur à la hanche": "hip_joint_pain", "douleur a la hanche": "hip_joint_pain",
    "faiblesse musculaire": "muscle_weakness",
    "raideur de la nuque": "stiff_neck", "cou raide": "stiff_neck",
    "gonflement des articulations": "swelling_joints",
    "raideur des mouvements": "movement_stiffness",
    "sensation de rotation": "spinning_movements",
    "perte d'équilibre": "loss_of_balance", "perte dequilibre": "loss_of_balance",
    "instabilité": "unsteadiness", "instabilite": "unsteadiness",
    "faiblesse d'une moitié du corps": "weakness_of_one_half_of_body", "paralysie partielle": "weakness_of_one_half_of_body",
    "perte d'odorat": "loss_of_smell", "perte dodorat": "loss_of_smell",
    "gêne à la vessie": "bladder_discomfort", "gene a la vessie": "bladder_discomfort",
    "urine malodorante": "foul_smell_of_urine",
    "envie d'uriner constante": "continuous_feel_of_urine", "envie duriner constante": "continuous_feel_of_urine",
    "flatulences": "passage_of_gases", "gaz": "passage_of_gases",
    "démangeaison interne": "internal_itching", "demangeaison interne": "internal_itching",
    "état léthargique": "toxic_look_typhos", "etat lethargique": "toxic_look_typhos",
    "dépression": "depression", "depression": "depression", "tristesse": "depression",
    "irritabilité": "irritability", "irritabilite": "irritability",
    "douleur musculaire": "muscle_pain", "courbatures": "muscle_pain",
    "altération de la conscience": "altered_sensorium", "alteration de la conscience": "altered_sensorium",
    "taches rouges sur le corps": "red_spots_over_body",
    "règles anormales": "abnormal_menstruation", "regles anormales": "abnormal_menstruation",
    "yeux qui pleurent": "watering_from_eyes",
    "appétit augmenté": "increased_appetite", "appetit augmente": "increased_appetite",
    "polyurie": "polyuria", "uriner fréquemment": "polyuria", "uriner frequemment": "polyuria",
    "antécédents familiaux": "family_history", "antecedents familiaux": "family_history",
    "crachats muqueux": "mucoid_sputum",
    "crachats rouillés": "rusty_sputum", "crachats rouilles": "rusty_sputum",
    "manque de concentration": "lack_of_concentration",
    "troubles visuels": "visual_disturbances",
    "transfusion sanguine reçue": "receiving_blood_transfusion", "transfusion sanguine recue": "receiving_blood_transfusion",
    "injections non stériles reçues": "receiving_unsterile_injections", "injections non steriles recues": "receiving_unsterile_injections",
    "coma": "coma",
    "saignement d'estomac": "stomach_bleeding", "saignement destomac": "stomach_bleeding",
    "gonflement de l'abdomen": "distention_of_abdomen", "gonflement de labdomen": "distention_of_abdomen",
    "consommation d'alcool fréquente": "history_of_alcohol_consumption", "consommation dalcool frequente": "history_of_alcohol_consumption",
    "sang dans les crachats": "blood_in_sputum",
    "veines saillantes sur les mollets": "prominent_veins_on_calf",
    "marche douloureuse": "painful_walking",
    "ongles enflammés": "inflammatory_nails", "ongles enflammes": "inflammatory_nails",
    "petits creux sur les ongles": "small_dents_in_nails",
}

# English symptom mapping (mapping phrases/synonyms to standard dataset column values)
ENGLISH_SYMPTOMS_MAPPING = {
    "itching": "itching", "itchy": "itching", "itch": "itching",
    "skin rash": "skin_rash", "rash": "skin_rash", "red spots": "skin_rash",
    "nodal skin eruptions": "nodal_skin_eruptions", "skin eruptions": "nodal_skin_eruptions", "nodules": "nodal_skin_eruptions",
    "dischromic patches": "dischromic_patches", "discolored patches": "dischromic_patches",
    "shivering": "shivering", "shiver": "shivering",
    "chills": "chills", "feeling cold": "chills",
    "joint pain": "joint_pain", "pain in joints": "joint_pain",
    "stomach pain": "stomach_pain", "belly ache": "stomach_pain", "stomach ache": "stomach_pain",
    "acidity": "acidity", "heartburn": "acidity", "acid reflux": "acidity",
    "ulcers on tongue": "ulcers_on_tongue", "tongue ulcers": "ulcers_on_tongue",
    "muscle wasting": "muscle_wasting", "muscle loss": "muscle_wasting",
    "vomiting": "vomiting", "throw up": "vomiting", "vomit": "vomiting",
    "burning micturition": "burning_micturition", "burning urination": "burning_micturition", "painful pee": "burning_micturition",
    "spotting urination": "spotting_urination", "blood in urine": "spotting_urination",
    "fatigue": "fatigue", "tired": "fatigue", "weakness": "fatigue", "exhausted": "fatigue",
    "weight gain": "weight_gain", "gaining weight": "weight_gain",
    "anxiety": "anxiety", "anxious": "anxiety", "panic": "anxiety",
    "cold hands and feet": "cold_hands_and_feets", "cold limbs": "cold_hands_and_feets",
    "mood swings": "mood_swings", "moody": "mood_swings",
    "weight loss": "weight_loss", "losing weight": "weight_loss",
    "restlessness": "restlessness", "restless": "restlessness",
    "lethargy": "lethargy", "lethargic": "lethargy",
    "patches in throat": "patches_in_throat", "throat patches": "patches_in_throat",
    "irregular sugar level": "irregular_sugar_level", "unstable blood sugar": "irregular_sugar_level",
    "cough": "cough", "coughing": "cough",
    "high fever": "high_fever", "fever": "high_fever", "very hot": "high_fever",
    "sunken eyes": "sunken_eyes", "dark circles under eyes": "sunken_eyes",
    "breathlessness": "breathlessness", "shortness of breath": "breathlessness", "difficulty breathing": "breathlessness", "dyspnea": "breathlessness",
    "sweating": "sweating", "perspiring": "sweating", "sweat": "sweating",
    "dehydration": "dehydration", "dehydrated": "dehydration",
    "indigestion": "indigestion", "upset stomach": "indigestion",
    "headache": "headache", "head pain": "headache", "migraine": "headache",
    "yellowish skin": "yellowish_skin", "yellow skin": "yellowish_skin",
    "dark urine": "dark_urine", "brown urine": "dark_urine",
    "nausea": "nausea", "nauseous": "nausea",
    "loss of appetite": "loss_of_appetite", "not hungry": "loss_of_appetite", "appetite loss": "loss_of_appetite",
    "pain behind the eyes": "pain_behind_the_eyes", "eye pain": "pain_behind_the_eyes",
    "back pain": "back_pain", "lumbago": "back_pain",
    "constipation": "constipation", "constipated": "constipation",
    "abdominal pain": "abdominal_pain", "stomach pain": "abdominal_pain", "cramps in belly": "abdominal_pain",
    "diarrhoea": "diarrhoea", "diarrhea": "diarrhoea", "loose stool": "diarrhoea",
    "mild fever": "mild_fever", "low fever": "mild_fever",
    "yellow urine": "yellow_urine",
    "yellowing of eyes": "yellowing_of_eyes", "yellow eyes": "yellowing_of_eyes",
    "acute liver failure": "acute_liver_failure", "liver failure": "acute_liver_failure",
    "fluid overload": "fluid_overload", "water retention": "fluid_overload",
    "swelling of stomach": "swelling_of_stomach", "bloated stomach": "swelling_of_stomach",
    "swelled lymph nodes": "swelled_lymph_nodes", "swollen glands": "swelled_lymph_nodes",
    "malaise": "malaise", "unwell": "malaise",
    "blurred and distorted vision": "blurred_and_distorted_vision", "blurry vision": "blurred_and_distorted_vision",
    "phlegm": "phlegm", "mucus": "phlegm",
    "throat irritation": "throat_irritation", "sore throat": "throat_irritation",
    "redness of eyes": "redness_of_eyes", "red eyes": "redness_of_eyes",
    "sinus pressure": "sinus_pressure", "sinus pain": "sinus_pressure",
    "runny nose": "runny_nose", "sneezing": "runny_nose",
    "congestion": "congestion", "stuffy nose": "congestion",
    "chest pain": "chest_pain", "pain in chest": "chest_pain", "chest tightness": "chest_pain",
    "weakness in limbs": "weakness_in_limbs", "weak legs": "weakness_in_limbs", "weak arms": "weakness_in_limbs",
    "fast heart rate": "fast_heart_rate", "racing heart": "fast_heart_rate", "tachycardia": "fast_heart_rate",
    "pain during bowel movements": "pain_during_bowel_movements",
    "pain in anal region": "pain_in_anal_region", "anal pain": "pain_in_anal_region",
    "bloody stool": "bloody_stool", "blood in stool": "bloody_stool",
    "irritation in anus": "irritation_in_anus",
    "neck pain": "neck_pain", "stiff neck": "neck_pain",
    "dizziness": "dizziness", "dizzy": "dizziness", "lightheaded": "dizziness",
    "cramps": "cramps", "muscle cramps": "cramps",
    "bruising": "bruising", "bruises": "bruising",
    "obesity": "obesity", "fat": "obesity", "overweight": "obesity",
    "swollen legs": "swollen_legs",
    "swollen blood vessels": "swollen_blood_vessels",
    "puffy face and eyes": "puffy_face_and_eyes", "swollen face": "puffy_face_and_eyes",
    "enlarged thyroid": "enlarged_thyroid", "goiter": "enlarged_thyroid",
    "brittle nails": "brittle_nails",
    "swollen extremeties": "swollen_extremeties", "swollen hands": "swollen_extremeties",
    "excessive hunger": "excessive_hunger", "always hungry": "excessive_hunger",
    "extra marital contacts": "extra_marital_contacts",
    "drying and tingling lips": "drying_and_tingling_lips",
    "slurred speech": "slurred_speech", "difficulty speaking": "slurred_speech",
    "knee pain": "knee_pain",
    "hip joint pain": "hip_joint_pain", "hip pain": "hip_joint_pain",
    "muscle weakness": "muscle_weakness",
    "stiff neck": "stiff_neck", "neck stiffness": "stiff_neck",
    "swelling joints": "swelling_joints", "swollen joints": "swelling_joints",
    "movement stiffness": "movement_stiffness", "stiff movements": "movement_stiffness",
    "spinning movements": "spinning_movements", "vertigo": "spinning_movements",
    "loss of balance": "loss_of_balance", "unbalanced": "loss_of_balance",
    "unsteadiness": "unsteadiness", "unsteady": "unsteadiness",
    "weakness of one half of body": "weakness_of_one_half_of_body", "hemiplegia": "weakness_of_one_half_of_body",
    "loss of smell": "loss_of_smell", "anosmia": "loss_of_smell",
    "bladder discomfort": "bladder_discomfort", "bladder pain": "bladder_discomfort",
    "foul smell of urine": "foul_smell_of_urine", "smelly urine": "foul_smell_of_urine",
    "continuous feel of urine": "continuous_feel_of_urine", "frequent urination": "continuous_feel_of_urine",
    "passage of gases": "passage_of_gases", "gas": "passage_of_gases", "farting": "passage_of_gases",
    "internal itching": "internal_itching",
    "toxic look (typhos)": "toxic_look_typhos",
    "depression": "depression", "depressed": "depression",
    "irritability": "irritability", "irritable": "irritability",
    "muscle pain": "muscle_pain", "body ache": "muscle_pain", "sore muscles": "muscle_pain",
    "altered sensorium": "altered_sensorium", "confusion": "altered_sensorium", "disoriented": "altered_sensorium",
    "red spots over body": "red_spots_over_body",
    "belly pain": "belly_pain", "stomach pain": "belly_pain",
    "abnormal menstruation": "abnormal_menstruation", "irregular periods": "abnormal_menstruation",
    "dischromic _patches": "dischromic_patches", "discoloured patches": "dischromic_patches",
    "watering from eyes": "watering_from_eyes", "watery eyes": "watering_from_eyes",
    "increased appetite": "increased_appetite", "polyuria": "polyuria",
    "family history": "family_history",
    "mucoid sputum": "mucoid_sputum", "rusty sputum": "rusty_sputum",
    "lack of concentration": "lack_of_concentration", "cannot focus": "lack_of_concentration",
    "visual disturbances": "visual_disturbances", "vision problems": "visual_disturbances",
    "receiving blood transfusion": "receiving_blood_transfusion",
    "receiving unsterile injections": "receiving_unsterile_injections",
    "coma": "coma", "unconscious": "coma",
    "stomach bleeding": "stomach_bleeding", "blood in vomit": "stomach_bleeding",
    "distention of abdomen": "distention_of_abdomen", "swollen belly": "distention_of_abdomen",
    "history of alcohol consumption": "history_of_alcohol_consumption", "drinking alcohol": "history_of_alcohol_consumption",
    "blood in sputum": "blood_in_sputum", "coughing up blood": "blood_in_sputum",
    "prominent veins on calf": "prominent_veins_on_calf",
    "palpitations": "palpitations", "heart palpitations": "palpitations",
    "painful walking": "painful_walking", "hurt to walk": "painful_walking",
    "pus filled pimples": "pus_filled_pimples",
    "blackheads": "blackheads", "scurring": "scurring",
    "skin peeling": "skin_peeling", "peeling skin": "skin_peeling",
    "silver like dusting": "silver_like_dusting",
    "small dents in nails": "small_dents_in_nails",
    "inflammatory nails": "inflammatory_nails",
    "blister": "blister", "blisters": "blister",
    "red sore around nose": "red_sore_around_nose", "sores around nose": "red_sore_around_nose",
    "yellow crust ooze": "yellow_crust_ooze"
}

class MedicalNERExtractor:
    def __init__(self):
        """Initializes the medical entity extractor."""
        self.french_dict = FRENCH_TO_ENGLISH_SYMPTOMS
        self.english_dict = ENGLISH_SYMPTOMS_MAPPING
        
        # Load BERT-based NER model if transformers is available, otherwise default to rule-based fallback.
        self.ner_pipeline = None
        self._init_deep_learning_ner()

    def _init_deep_learning_ner(self):
        """Attempts to load the clinical HuggingFace NER model as a preference."""
        try:
            from transformers import pipeline
            # Using samrawal/bert-base-uncased_clinical-ner as specified in implementation plan.
            # Set to local-files-only=False to allow downloading.
            logger.info("Attempting to load clinical NER pipeline from Hugging Face...")
            self.ner_pipeline = pipeline("ner", model="samrawal/bert-base-uncased_clinical-ner", aggregation_strategy="simple")
            logger.info("[SUCCESS] Hugging Face Clinical NER loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load Hugging Face NER pipeline: {e}. Falling back to Rule/Dictionary-based NER.")
            self.ner_pipeline = None

    def extract_symptoms(self, text: str) -> dict:
        """
        Parses the user query to extract medical symptoms, duration, intensity, and location.
        Returns a dict containing extracted attributes.
        """
        if not text:
            return {"symptoms": [], "duration": None, "intensity": None, "location": None, "raw_symptoms_matched": []}
            
        normalized_text = text.lower().strip()
        
        # 1. Extract metadata (duration, intensity, location) using regex
        duration = self._extract_duration(normalized_text)
        intensity = self._extract_intensity(normalized_text)
        location = self._extract_location(normalized_text)
        
        # 2. Extract symptoms (dictionary + phrase matching)
        extracted_symptoms = set()
        raw_matches = []
        
        # Try matching French keys first
        for fr_key, en_val in self.french_dict.items():
            # Use regex word boundaries or phrase matching to avoid substring collision (like "mal" in "maladie")
            pattern = r'\b' + re.escape(fr_key) + r'\b'
            # For multi-word phrases, drop the word boundary requirement slightly if needed, or stick to regex search
            if len(fr_key.split()) > 1:
                pattern = re.escape(fr_key)
                
            if re.search(pattern, normalized_text):
                extracted_symptoms.add(en_val)
                raw_matches.append(fr_key)
                
        # Try matching English keys
        for en_key, en_val in self.english_dict.items():
            pattern = r'\b' + re.escape(en_key) + r'\b'
            if len(en_key.split()) > 1:
                pattern = re.escape(en_key)
                
            if re.search(pattern, normalized_text):
                extracted_symptoms.add(en_val)
                raw_matches.append(en_key)
                
        # 3. If BERT NER is available, run it to see if we can find additional details
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text)
                for ent in entities:
                    word = ent['word'].lower().strip()
                    # Check if the extracted word matches any known English/French symptom
                    for en_key, en_val in self.english_dict.items():
                        if word in en_key or en_key in word:
                            extracted_symptoms.add(en_val)
                            raw_matches.append(word)
            except Exception as e:
                logger.error(f"Error during BERT NER inference: {e}")

        # Clean duplicates in raw matches
        raw_matches = list(set(raw_matches))

        return {
            "symptoms": sorted(list(extracted_symptoms)),
            "duration": duration,
            "intensity": intensity,
            "location": location,
            "raw_symptoms_matched": raw_matches
        }

    def _extract_duration(self, text: str) -> str:
        """Extracts the duration (e.g., '3 days', 'depuis hier') using regex."""
        # English patterns: e.g. "for 3 days", "since yesterday", "for a week", "2 weeks"
        en_patterns = [
            r'(for|since)\s+\d+\s+(day|week|month|hour)s?',
            r'\d+\s+(day|week|month|hour)s?',
            r'since\s+(yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'for\s+a\s+(week|month|day)'
        ]
        # French patterns: e.g. "depuis 3 jours", "depuis hier", "pendant une semaine", "depuis 2 semaines"
        fr_patterns = [
            r'depuis\s+\d+\s+(jour|semaine|mois|heure)s?',
            r'depuis\s+(hier|avant-hier|lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)',
            r'pendant\s+une\s+(semaine|journée|heure)',
            r'durant\s+\d+\s+(jour|semaine|mois|heure)s?'
        ]
        
        for pattern in en_patterns + fr_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

    def _extract_intensity(self, text: str) -> str:
        """Extracts intensity indicators like scale (e.g. 8/10) or qualitative descriptions."""
        # Look for numeric scales like "8/10", "7 sur 10", "scale 9"
        scale_match = re.search(r'\b(\d+)\s*(?:/|sur)\s*10\b', text)
        if scale_match:
            return f"severity score: {scale_match.group(1)}/10"
            
        # English terms
        en_intensity = {
            "severe": ["severe", "extremely painful", "unbearable", "very intense", "crushing", "acute"],
            "moderate": ["moderate", "medium", "mildly painful", "tolerable"],
            "mild": ["mild", "minor", "slight", "little bit"]
        }
        # French terms
        fr_intensity = {
            "severe": ["severe", "sévère", "insupportable", "très intense", "aigüe", "aigue", "atroce", "violent", "forte", "fortes", "fort", "forts", "grosse", "gros", "intense"],
            "moderate": ["modéré", "modere", "moyen", "passable"],
            "mild": ["léger", "leger", "légère", "legere", "faible", "un peu"]
        }
        
        # Search intensity terms
        for level, terms in en_intensity.items():
            for term in terms:
                if term in text:
                    return level
        for level, terms in fr_intensity.items():
            for term in terms:
                if term in text:
                    return level
                    
        return None

    def _extract_location(self, text: str) -> str:
        """Extracts the anatomical location of the symptoms using a dictionary."""
        locations = {
            # English
            "chest": ["chest", "thorax", "breast"],
            "abdomen": ["stomach", "abdomen", "belly", "gut"],
            "head": ["head", "forehead", "temple"],
            "throat": ["throat", "neck"],
            "eyes": ["eye", "eyes"],
            "limbs": ["leg", "legs", "arm", "arms", "foot", "feet", "hand", "hands", "joint", "joints", "knee", "hip"],
            "back": ["back", "spine", "lumber"],
            "skin": ["skin", "face", "scalp"],
            
            # French
            "poitrine": ["poitrine", "thorax", "cœur", "coeur"],
            "abdomen_fr": ["ventre", "estomac", "abdomen"],
            "tête": ["tête", "tete", "crâne", "sinus"],
            "gorge": ["gorge", "cou", "nuque"],
            "yeux": ["oeil", "yeux"],
            "membres": ["bras", "jambe", "jambes", "pied", "pieds", "main", "mains", "articulation", "articulations", "genou", "hanche"],
            "dos": ["dos", "colonne"],
            "peau": ["peau", "visage", "cuir chevelu"]
        }
        
        for standard_loc, terms in locations.items():
            for term in terms:
                if r'\b' + re.escape(term) + r'\b' in text or term in text:
                    # Return English standardized location equivalent
                    if standard_loc.endswith("_fr"):
                        return standard_loc[:-3]
                    elif standard_loc == "poitrine":
                        return "chest"
                    elif standard_loc == "tête":
                        return "head"
                    elif standard_loc == "yeux":
                        return "eyes"
                    elif standard_loc == "membres":
                        return "limbs"
                    elif standard_loc == "dos":
                        return "back"
                    elif standard_loc == "peau":
                        return "skin"
                    return standard_loc
        return None

if __name__ == "__main__":
    extractor = MedicalNERExtractor()
    test_cases = [
        "I have a severe headache and high fever since yesterday",
        "J'ai de fortes démangeaisons et des boutons rouges sur la peau depuis 3 jours",
        "J'ai une grosse douleur à la poitrine de 9/10"
    ]
    for tc in test_cases:
        print(f"\nText: {tc}")
        print(extractor.extract_symptoms(tc))
