"""
Triage Classifier Module for Medical Triage Chatbot.
Evaluates extracted symptoms, intensity, duration, and RAG candidate diseases 
to assign a standardized medical triage level (P1 to P5) and recommend concrete actions.
"""
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Triage definitions from the Cahier des Charges
TRIAGE_LEVELS = {
    "P1": {
        "label_fr": "Urgence vitale",
        "label_en": "Immediate Emergency (Life Threatening)",
        "action_fr": "Appel immédiat des services d'urgence (SAMU 15, 112 ou 911). Ne vous déplacez pas vous-même.",
        "action_en": "Call emergency services (15, 112, or 911) immediately. Do not attempt to drive yourself."
    },
    "P2": {
        "label_fr": "Urgence",
        "label_en": "Emergency",
        "action_fr": "Rendez-vous aux urgences hospitalières les plus proches dans l'heure.",
        "action_en": "Go to the nearest hospital emergency room within the hour."
    },
    "P3": {
        "label_fr": "Semi-urgence",
        "label_en": "Semi-emergency",
        "action_fr": "Consultez un médecin ou rendez-vous dans un centre de garde sous 24 heures.",
        "action_en": "Consult a doctor or visit an urgent care clinic within 24 hours."
    },
    "P4": {
        "label_fr": "Non urgent",
        "label_en": "Non-urgent",
        "action_fr": "Prenez rendez-vous avec votre médecin généraliste sous 48 à 72 heures.",
        "action_en": "Schedule an appointment with your general practitioner within 48 to 72 hours."
    },
    "P5": {
        "label_fr": "Conseil",
        "label_en": "Advice / Self-care",
        "action_fr": "Automédication, conseil auprès d'un pharmacien ou consultation de télémédecine.",
        "action_en": "Self-care, consult your local pharmacist, or schedule a telemedicine consultation."
    }
}

class TriageClassifier:
    def __init__(self):
        # Define P1 symptoms (Red Flags)
        self.p1_symptoms = {
            "chest_pain", 
            "breathlessness", 
            "coma", 
            "altered_sensorium", 
            "weakness_of_one_half_of_body", 
            "slurred_speech",
            "stomach_bleeding",
            "acute_liver_failure"
        }
        
        # Serious diseases that require at least P3 (within 24h)
        self.p3_diseases = {
            "Malaria", "Typhoid", "Dengue", "Pneumonia", "Tuberculosis",
            "hepatitis A", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
            "Alcoholic hepatitis", "Jaundice", "AIDS"
        }
        
        # Chronic/Stable diseases that default to P4
        self.p4_diseases = {
            "GERD", "Diabetes ", "Hypertension ", "Hypothyroidism", "Hyperthyroidism",
            "Osteoarthristis", "Arthritis", "Acne", "Psoriasis", "Varicose veins",
            "Cervical spondylosis", "Dimorphic hemmorhoids(piles)"
        }

    def classify(self, extracted_entities: dict, suspected_diseases: list) -> dict:
        """
        Classifies symptoms into P1-P5 triage level.
        Args:
            extracted_entities (dict): output of NER extractor.
            suspected_diseases (list): output of RAG search.
        Returns:
            dict: standardized triage classification.
        """
        symptoms = set(extracted_entities.get("symptoms", []))
        intensity = extracted_entities.get("intensity")
        duration = extracted_entities.get("duration")
        location = extracted_entities.get("location")
        
        reasons_fr = []
        reasons_en = []
        
        # Get top-1 disease name for disease-based rules
        top_disease = suspected_diseases[0]["disease_name"] if suspected_diseases else None
        
        # 1. P1 RULES (Urgence Vitale)
        p1_triggers = symptoms.intersection(self.p1_symptoms)
        
        # If any major P1 symptoms are matched
        if p1_triggers:
            for trig in p1_triggers:
                reasons_fr.append(f"Présence d'un symptôme critique : {trig.replace('_', ' ')}")
                reasons_en.append(f"Critical symptom detected: {trig.replace('_', ' ')}")
            return self._format_response("P1", reasons_fr, reasons_en)
            
        # Specific P1 conditions: Heart attack or stroke suspect
        if top_disease == "Heart attack":
            reasons_fr.append("Suspicion d'infarctus du myocarde (crise cardiaque).")
            reasons_en.append("Suspicion of acute myocardial infarction (heart attack).")
            return self._format_response("P1", reasons_fr, reasons_en)
            
        if top_disease == "Paralysis (brain hemorrhage)":
            reasons_fr.append("Suspicion d'accident vasculaire cérébral (AVC) ou hémorragie cérébrale.")
            reasons_en.append("Suspicion of stroke or cerebral hemorrhage.")
            return self._format_response("P1", reasons_fr, reasons_en)
            
        # 2. P2 RULES (Urgence - dans l'heure)
        # Severe intensity chest or abdominal pain
        if intensity == "severe" and location in ["chest", "abdomen"]:
            reasons_fr.append(f"Douleur sévère au niveau du/de la {location}.")
            reasons_en.append(f"Severe pain located in the {location}.")
            return self._format_response("P2", reasons_fr, reasons_en)
            
        # Meningitis suspect: stiff neck + high fever/headache
        if "stiff_neck" in symptoms and ("high_fever" in symptoms or "headache" in symptoms):
            reasons_fr.append("Raideur de la nuque accompagnée de fièvre/céphalée (suspicion de méningite).")
            reasons_en.append("Stiff neck combined with fever/headache (suspicion of meningitis).")
            return self._format_response("P2", reasons_fr, reasons_en)
            
        # Severe difficulty breathing/breathlessness or high pain scale
        if intensity and "severity score: 8" in intensity or intensity and "severity score: 9" in intensity or intensity and "severity score: 10" in intensity:
            reasons_fr.append("Intensité de la douleur évaluée à 8/10 ou plus.")
            reasons_en.append("Pain intensity score rated at 8/10 or higher.")
            return self._format_response("P2", reasons_fr, reasons_en)

        # 3. P3 RULES (Semi-urgence - sous 24h)
        # Serious infections like malaria, typhoid, dengue, pneumonia
        if top_disease in self.p3_diseases:
            reasons_fr.append(f"Suspicion de pathologie infectieuse ou aiguë : {top_disease}.")
            reasons_en.append(f"Suspicion of acute infectious pathology: {top_disease}.")
            return self._format_response("P3", reasons_fr, reasons_en)
            
        # Persistent high fever or moderate chest pain
        if "high_fever" in symptoms:
            reasons_fr.append("Présence d'une forte fièvre.")
            reasons_en.append("High fever detected.")
            return self._format_response("P3", reasons_fr, reasons_en)
            
        if "yellowish_skin" in symptoms or "yellowing_of_eyes" in symptoms:
            reasons_fr.append("Présence de jaunisse (ictère).")
            reasons_en.append("Presence of jaundice (yellowish skin/eyes).")
            return self._format_response("P3", reasons_fr, reasons_en)
            
        if intensity == "severe":
            reasons_fr.append("Douleur décrite comme sévère.")
            reasons_en.append("Symptoms reported as severe.")
            return self._format_response("P3", reasons_fr, reasons_en)

        # 4. P4 RULES (Non urgent - sous 48h-72h)
        # Chronic conditions or moderate symptoms
        if top_disease in self.p4_diseases:
            reasons_fr.append(f"Symptômes évocateurs d'une affection chronique stable : {top_disease}.")
            reasons_en.append(f"Symptoms consistent with a stable chronic condition: {top_disease}.")
            return self._format_response("P4", reasons_fr, reasons_en)
            
        # Mild fever or moderate pain
        if "mild_fever" in symptoms or intensity == "moderate":
            reasons_fr.append("Symptômes modérés ou fièvre légère.")
            reasons_en.append("Moderate symptoms or mild fever.")
            return self._format_response("P4", reasons_fr, reasons_en)
            
        # If symptoms have lasted a long time without worsening
        if duration and ("week" in duration or "month" in duration or "semaine" in duration or "mois" in duration):
            reasons_fr.append("Symptômes persistants depuis plusieurs semaines sans critères de gravité.")
            reasons_en.append("Symptoms persistent for weeks without red flags.")
            return self._format_response("P4", reasons_fr, reasons_en)

        # 5. P5 RULES (Conseil / Automédication)
        # Banal symptoms like cold, runy nose, mild itching, or if no symptoms matched
        reasons_fr.append("Symptômes bénins et stables, adaptés à l'automédication ou au conseil officinal.")
        reasons_en.append("Mild, stable symptoms suitable for self-care or pharmacist advice.")
        return self._format_response("P5", reasons_fr, reasons_en)

    def _format_response(self, level: str, reasons_fr: list, reasons_en: list) -> dict:
        """Helper to format triage dict."""
        level_info = TRIAGE_LEVELS[level]
        return {
            "triage_level": level,
            "label_fr": level_info["label_fr"],
            "label_en": level_info["label_en"],
            "action_fr": level_info["action_fr"],
            "action_en": level_info["action_en"],
            "reasons_fr": reasons_fr,
            "reasons_en": reasons_en
        }

if __name__ == "__main__":
    classifier = TriageClassifier()
    # Test P1
    print(classifier.classify({"symptoms": ["chest_pain"], "intensity": "severe", "duration": "10 minutes", "location": "chest"}, [{"disease_name": "Heart attack"}]))
    # Test P3
    print(classifier.classify({"symptoms": ["high_fever", "vomiting"], "intensity": "moderate", "duration": "2 days", "location": "abdomen"}, [{"disease_name": "Malaria"}]))
