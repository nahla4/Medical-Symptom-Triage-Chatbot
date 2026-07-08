"""
Safety Filter Module for Medical Triage Chatbot.
Provides a pre-LLM keyword filter to bypass LLM processing and issue immediate P1 alerts in critical situations.
Injects the mandatory medical disclaimer to all outgoing responses (post-processing).
"""
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of critical emergency terms (French & English) that trigger immediate P1 bypass
EMERGENCY_KEYWORDS = [
    # English
    r"chest pain", r"pain in chest", r"heart attack", r"infarction",
    r"cannot breathe", r"can't breathe", r"breathless", r"shortness of breath", r"dyspnea", r"choking",
    r"unconscious", r"passed out", r"fainted", r"loss of consciousness",
    r"stroke", r"paralysis", r"paralyzed", r"slurred speech", r"speech difficulty",
    r"poisoned", r"poisoning",
    r"severe bleeding", r"hemorrhage", r"bleeding heavily",
    r"suicidal", r"suicide", r"kill myself",
    
    # French
    r"douleur poitrine", r"douleur a la poitrine", r"douleur à la poitrine", r"crise cardiaque", r"infarctus",
    r"du mal a respirer", r"du mal à respirer", r"étouffe", r"etouffe", r"essoufflement", r"asphyxie",
    r"inconscient", r"perte de connaissance", r"évanoui", r"evanoui", r"coma",
    r"avc", r"paralysie", r"paralyse", r"paralysé", r"parole confuse", r"difficulté à parler",
    r"empoisonné", r"empoisonne", r"intoxication",
    r"hémorragie", r"hemorragie", r"saigne abondamment", r"saignement important",
    r"suicide", r"suicidaire", r"me tuer"
]

MEDICAL_DISCLAIMER_FR = (
    "\n\n---"
    "\n⚠️ **AVERTISSEMENT MÉDICAL DE SÉCURITÉ** :"
    "\nCe système est un outil d'aide à la décision préliminaire basé sur l'intelligence artificielle et ne remplace en aucun cas un diagnostic médical professionnel. "
    "En cas d'urgence vitale, ne perdez pas de temps à chatter et contactez immédiatement le SAMU (15), le numéro européen (112) ou les urgences."
)

MEDICAL_DISCLAIMER_EN = (
    "\n\n---"
    "\n⚠️ **MEDICAL SAFETY DISCLAIMER**:"
    "\nThis system is a preliminary decision support tool powered by AI and does not replace professional medical diagnosis, advice, or treatment. "
    "If you are experiencing a life-threatening emergency, do not wait for a chat response and contact emergency services (112, 911, or 15) immediately."
)

class SafetyFilter:
    def __init__(self):
        # Compile patterns for fast lookup
        self.emergency_patterns = [
            re.compile(r'\b' + pattern + r'\b', re.IGNORECASE) for pattern in EMERGENCY_KEYWORDS
        ]

    def detect_immediate_emergency(self, query: str) -> bool:
        """
        Scans the user query for immediate life-threatening terms.
        Returns True if an emergency is detected, triggering a bypass.
        """
        if not query:
            return False
            
        normalized_query = query.lower().strip()
        
        for pattern in self.emergency_patterns:
            if pattern.search(normalized_query):
                logger.warning(f"Safety trigger matched pattern: {pattern.pattern}")
                return True
                
        # Also check for multi-word exact matches that might bypass word boundaries
        for kw in ["avc", "crise cardiaque", "heart attack", "can't breathe"]:
            if kw in normalized_query:
                logger.warning(f"Safety trigger matched substring: {kw}")
                return True
                
        return False

    def get_emergency_response(self, detected_language: str = "fr") -> dict:
        """
        Returns a structured P1 emergency response bypass.
        """
        if detected_language == "fr":
            return {
                "triage_level": "P1",
                "label_fr": "Urgence vitale (Détection de sécurité)",
                "label_en": "Immediate Emergency (Safety Override)",
                "action_fr": "ATTENTION : Vos symptômes indiquent une urgence vitale potentielle. VEUILLEZ APPELER LE 15 ou LE 112 IMMÉDIATEMENT.",
                "action_en": "ATTENTION: Your symptoms suggest a potential life-threatening emergency. PLEASE CALL 911, 112, or 15 IMMEDIATELY.",
                "reasons_fr": ["Détection automatique de mots-clés d'urgence critique."],
                "reasons_en": ["Automatic detection of critical emergency keywords."],
                "suspected_conditions": [
                    {"disease_name": "Urgence Médicale Majeure", "icd10": "U00.0", "score": 1.0, "description": "Situation nécessitant une intervention médicale immédiate.", "precautions": ["Appeler les urgences", "Ne pas faire d'effort"]}
                ],
                "safe_override": True
            }
        else:
            return {
                "triage_level": "P1",
                "label_fr": "Urgence vitale (Détection de sécurité)",
                "label_en": "Immediate Emergency (Safety Override)",
                "action_fr": "ATTENTION : Vos symptômes indiquent une urgence vitale potentielle. VEUILLEZ APPELER LE 15 ou LE 112 IMMÉDIATEMENT.",
                "action_en": "ATTENTION: Your symptoms suggest a potential life-threatening emergency. PLEASE CALL 911, 112, or 15 IMMEDIATELY.",
                "reasons_fr": ["Détection automatique de mots-clés d'urgence critique."],
                "reasons_en": ["Automatic detection of critical emergency keywords."],
                "suspected_conditions": [
                    {"disease_name": "Major Medical Emergency", "icd10": "U00.0", "score": 1.0, "description": "Situation requiring immediate medical attention.", "precautions": ["Call emergency services", "Rest and stay calm"]}
                ],
                "safe_override": True
            }

    def inject_disclaimer(self, text: str, language: str = "fr") -> str:
        """
        Appends the medical disclaimer to the final chat text.
        """
        if language == "fr":
            return text + MEDICAL_DISCLAIMER_FR
        else:
            return text + MEDICAL_DISCLAIMER_EN

    def detect_language(self, text: str) -> str:
        """
        Simple heuristic language detector (FR vs EN).
        """
        french_words = {"je", "ai", "mal", "douleur", "fievre", "fièvre", "depuis", "hier", "toux", "tête", "tete", "poitrine", "ventre", "et", "est", "une", "un", "des", "les", "vomir", "vomissement"}
        english_words = {"i", "have", "pain", "fever", "cough", "since", "yesterday", "head", "chest", "stomach", "and", "is", "a", "an", "the", "vomit", "vomiting"}
        
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        fr_count = len(words.intersection(french_words))
        en_count = len(words.intersection(english_words))
        
        if fr_count >= en_count:
            return "fr"
        else:
            return "en"

if __name__ == "__main__":
    sf = SafetyFilter()
    print("Emergency detected:", sf.detect_immediate_emergency("J'ai une douleur intense à la poitrine"))
    print("Disclaimer FR:")
    print(sf.inject_disclaimer("Voici une explication médicale.", "fr"))
