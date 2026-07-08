"""
LLM Handler Module for Medical Triage Chatbot.
Orchestrates prompt templates and chains using LangChain.
Interfaces with OpenAI ChatOpenAI if credentials are set,
and falls back to a realistic local mock responder if offline or key is missing.
"""
import os
import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are an advanced medical triage chatbot assistant designed to help patients evaluate their symptoms and guide them to the correct care level.
You are given:
1. The user's query: "{query}"
2. Extracted symptoms: {symptoms}
3. Extracted metadata (Duration: {duration}, Intensity: {intensity}, Location: {location})
4. RAG search results of suspected conditions: {rag_conditions}
5. Assigned Triage Level: {triage_level} ({triage_label})
6. Recommended Action: {triage_action}

Your goals are:
- Briefly summarize the symptoms mentioned by the user.
- Provide a list of the top 3 potential conditions based on the RAG results. Include the estimated confidence, description, precautions, and the ICD-10 code for each condition.
- Present the Triage level and the Recommended Action clearly in a separate section.
- NEVER form a final medical diagnosis. Always use conditional language (e.g., "suggests", "may indicate", "could be associated with").
- Do NOT add a custom disclaimer here; the safety layer will append it automatically.
- Keep the language of your response in the user's language: {language} (either French 'fr' or English 'en'). If 'fr', write completely in French.

Format your response in structured Markdown (headings, lists, bold text) for high readability.
"""

class MedicalLLMHandler:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model_name = "gpt-3.5-turbo"
        self.llm = None
        self.chain = None
        
        self._init_llm()

    def _init_llm(self):
        """Initializes LangChain LLM if api key is present."""
        if self.api_key:
            try:
                from langchain_openai import ChatOpenAI
                logger.info(f"Initializing LangChain ChatOpenAI with model {self.model_name}...")
                self.llm = ChatOpenAI(model=self.model_name, temperature=0.2, api_key=self.api_key)
                
                prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
                self.chain = prompt | self.llm | StrOutputParser()
                logger.info("[SUCCESS] LangChain LLM Chain initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not load LangChain OpenAI. Falling back to local responder. Error: {e}")
                self.llm = None
        else:
            logger.info("No OPENAI_API_KEY found. Operating in local mock responder mode.")

    def generate_response(self, query: str, extracted_entities: dict, triage_result: dict, rag_conditions: list, language: str = "fr") -> str:
        """
        Generates the final response using either LangChain LLM or the mock local responder.
        """
        # Format the RAG results to pass to the model/mock
        rag_formatted = []
        for cond in rag_conditions:
            rag_formatted.append({
                "name": cond["disease_name"],
                "icd10": cond["icd10"],
                "score": cond["score"],
                "description": cond["description"],
                "precautions": cond["precautions"]
            })
            
        triage_label = triage_result["label_fr"] if language == "fr" else triage_result["label_en"]
        triage_action = triage_result["action_fr"] if language == "fr" else triage_result["action_en"]
        
        # If API key is active and langchain is setup, execute the chain
        if self.chain:
            try:
                inputs = {
                    "query": query,
                    "symptoms": ", ".join(extracted_entities.get("symptoms", [])),
                    "duration": extracted_entities.get("duration") or "unspecified",
                    "intensity": extracted_entities.get("intensity") or "unspecified",
                    "location": extracted_entities.get("location") or "unspecified",
                    "rag_conditions": json.dumps(rag_formatted, indent=2),
                    "triage_level": triage_result["triage_level"],
                    "triage_label": triage_label,
                    "triage_action": triage_action,
                    "language": "French" if language == "fr" else "English"
                }
                response_text = self.chain.invoke(inputs)
                return response_text
            except Exception as e:
                logger.error(f"Error during LLM chain execution: {e}. Falling back to mock responder.")

        # Fallback Mock Responder (Highly detailed, realistic, and matches exact styling requirements)
        return self._generate_mock_response(query, extracted_entities, triage_result, rag_formatted, language)

    def _generate_mock_response(self, query: str, extracted_entities: dict, triage_result: dict, rag_formatted: list, language: str) -> str:
        """Generates a high-quality mock response for offline/keyless environments."""
        symptoms_list = extracted_entities.get("symptoms", [])
        clean_syms = [s.replace("_", " ") for s in symptoms_list]
        
        duration = extracted_entities.get("duration")
        intensity = extracted_entities.get("intensity")
        location = extracted_entities.get("location")
        
        triage_level = triage_result["triage_level"]
        triage_label = triage_result["label_fr"] if language == "fr" else triage_result["label_en"]
        triage_action = triage_result["action_fr"] if language == "fr" else triage_result["action_en"]
        
        if language == "fr":
            # French response
            symptoms_str = ", ".join(clean_syms) if clean_syms else "non spécifiés"
            response = []
            response.append("### 📋 Synthèse de l'Analyse des Symptômes\n")
            response.append(f"Vous avez décrit les symptômes suivants : **{symptoms_str}**.")
            if duration or intensity or location:
                meta_details = []
                if duration: meta_details.append(f"durée : *{duration}*")
                if intensity: meta_details.append(f"intensité : *{intensity}*")
                if location: meta_details.append(f"localisation : *{location}*")
                response.append(f"Détails complémentaires identifiés : {', '.join(meta_details)}.")
                
            response.append("\n### 🔍 Pathologies Potentielles (Diagnostic Différentiel)")
            response.append("Sur la base de notre base de connaissances médicale, voici les conditions qui *pourraient* correspondre à vos symptômes :")
            
            if not rag_formatted:
                response.append("\n*Aucune pathologie spécifique n'a pu être associée avec un score de confiance suffisant.*")
            else:
                for idx, cond in enumerate(rag_formatted[:3]):
                    # Estimate confidence percentage based on score (e.g. score of 0.85 -> 85%)
                    conf = int(cond["score"] * 100) if cond["score"] <= 1.0 else int(cond["score"] / 2.0 * 100)
                    conf = min(max(conf, 15), 98) # Clamp between 15% and 98%
                    
                    response.append(f"\n**{idx+1}. {cond['name']}** (Code CIM-10: `{cond['icd10']}`)")
                    response.append(f"- **Indice de probabilité** : ~{conf}%")
                    response.append(f"- **Description** : *{cond['description']}*")
                    response.append(f"- **Mesures de précaution recommandées** :")
                    for prec in cond["precautions"][:3]:
                        response.append(f"  - {prec}")
                        
            response.append(f"\n### 🛡️ Évaluation du Triage et Recommandations")
            response.append(f"- **Niveau d'Urgence** : **{triage_level}** — *{triage_label}*")
            response.append(f"- **Action Recommandée** : **{triage_action}**")
            
            if triage_result.get("reasons_fr"):
                response.append("- **Motif(s) d'évaluation** :")
                for reason in triage_result["reasons_fr"]:
                    response.append(f"  - {reason}")
                    
            return "\n".join(response)
        else:
            # English response
            symptoms_str = ", ".join(clean_syms) if clean_syms else "unspecified"
            response = []
            response.append("### 📋 Symptom Analysis Summary\n")
            response.append(f"You described the following symptoms: **{symptoms_str}**.")
            if duration or intensity or location:
                meta_details = []
                if duration: meta_details.append(f"duration: *{duration}*")
                if intensity: meta_details.append(f"intensity: *{intensity}*")
                if location: meta_details.append(f"location: *{location}*")
                response.append(f"Additional details identified: {', '.join(meta_details)}.")
                
            response.append("\n### 🔍 Potential Conditions (Differential Diagnosis)")
            response.append("Based on our medical knowledge base, the following conditions *could* be associated with your symptoms:")
            
            if not rag_formatted:
                response.append("\n*No specific pathology could be associated with high confidence.*")
            else:
                for idx, cond in enumerate(rag_formatted[:3]):
                    conf = int(cond["score"] * 100) if cond["score"] <= 1.0 else int(cond["score"] / 2.0 * 100)
                    conf = min(max(conf, 15), 98)
                    
                    response.append(f"\n**{idx+1}. {cond['name']}** (ICD-10 Code: `{cond['icd10']}`)")
                    response.append(f"- **Probability Index** : ~{conf}%")
                    response.append(f"- **Description** : *{cond['description']}*")
                    response.append(f"- **Recommended Precautions** :")
                    for prec in cond["precautions"][:3]:
                        response.append(f"  - {prec}")
                        
            response.append(f"\n### 🛡️ Triage Assessment & Action")
            response.append(f"- **Triage Level** : **{triage_level}** — *{triage_label}*")
            response.append(f"- **Recommended Action** : **{triage_action}**")
            
            if triage_result.get("reasons_en"):
                response.append("- **Assessment Reason(s)** :")
                for reason in triage_result["reasons_en"]:
                    response.append(f"  - {reason}")
                    
            return "\n".join(response)

if __name__ == "__main__":
    handler = MedicalLLMHandler()
    q = "J'ai de la fièvre et je tousse"
    ents = {"symptoms": ["cough", "high_fever"], "duration": "3 days", "intensity": "moderate", "location": "throat"}
    triage = {"triage_level": "P3", "label_fr": "Semi-urgence", "label_en": "Semi-emergency", "action_fr": "Consulter sous 24h", "action_en": "Consult in 24h", "reasons_fr": ["Fièvre élevée"]}
    rag = [{"disease_name": "Malaria", "icd10": "B54", "score": 0.8, "description": "Parasitic disease.", "precautions": ["Take meds"]}]
    print(handler.generate_response(q, ents, triage, rag, "fr"))
