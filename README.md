# MedTriage AI - Medical Symptom Triage Chatbot 🏥

## Project Overview
MedTriage AI is an advanced, bilingual (English/French) conversational AI web application designed to help patients evaluate their symptoms and guide them to the correct care level. The system uses a combination of Natural Language Processing (NLP) for symptom extraction, Retrieval-Augmented Generation (RAG) for potential condition matching, a rules-based Triage Classifier, and an LLM for providing a structured, readable summary. 

**Note**: This is a decision-support tool. It does *not* provide definitive medical diagnoses.

## Features
- **Bilingual Support**: Fully understands and responds in English and French.
- **Safety-First**: Immediate emergency detection (P1 bypass) with a mandatory medical disclaimer injected into every response.
- **NER Extraction**: Extracts symptoms, duration, intensity, and anatomical location using a robust regex/dictionary parser with Hugging Face Transformer fallback.
- **RAG Pipeline**: Leverages SentenceTransformers and ChromaDB (or a numpy-based fallback) to retrieve related conditions and precautions.
- **Triage Classifier**: Assigns standardized triage levels (P1 to P5) based on extracted entities and RAG results.
- **LLM Handler**: Orchestrates prompt templates via LangChain and interfaces with OpenAI to synthesize a cohesive, empathetic response (includes a rich mock responder if no API key is provided).
- **Premium UI**: A polished, responsive Streamlit web interface featuring custom CSS, dynamic badges, and interactive elements.

## Tech Stack
- **Frontend**: Streamlit, Custom CSS
- **NLP & NER**: Hugging Face Transformers (`sentence-transformers`, `bert-base-uncased_clinical-ner`), Regex
- **RAG & Vector Store**: ChromaDB, NumPy/Scikit-learn (fallback)
- **LLM Orchestration**: LangChain, OpenAI API (`gpt-3.5-turbo`)
- **Backend / Data**: Python 3.11, Pandas, FastAPI, Uvicorn
- **Testing**: Pytest

## Setup Instructions (Windows)

1. **Clone the repository**:
   ```powershell
   git clone <your-repo-url>
   cd Medical-Symptom-Triage-Chatbot
   ```

2. **Create a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the project root and add your OpenAI API key (optional, will use a rich mock responder if omitted):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## How to Run the App Locally

Start the Streamlit application using the following command:
```powershell
streamlit run app/chatbot_app.py
```
The app should automatically open in your default browser at `http://localhost:8501`.

## How to Run Tests

The project includes an extensive suite of unit tests for the NER extractor and Triage classifier.

To run the tests:
```powershell
pytest tests/ -v
```

## Deployment Instructions

### Docker

You can run the application using Docker and Docker Compose.

1. **Build and Run with Docker Compose**:
   ```powershell
   docker-compose up --build
   ```

2. **Build and Run with standard Docker**:
   ```powershell
   docker build -t medtriage-ai .
   docker run -p 8501:8501 -e OPENAI_API_KEY=your_key medtriage-ai
   ```

## ⚠️ Medical Disclaimer

**MEDICAL SAFETY DISCLAIMER**:
This system is a preliminary decision support tool powered by AI and does not replace professional medical diagnosis, advice, or treatment. If you are experiencing a life-threatening emergency, do not wait for a chat response and contact emergency services (112, 911, or 15) immediately.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.