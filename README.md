# AI Code Smell Detector

A **Java code smell detection tool** powered by a machine learning model.  
This project demonstrates an **end-to-end workflow** for analyzing Java code and detecting common code smells such as **LONG_METHOD, GOD_CLASS, and HIGH_COMPLEXITY** using a **Python CNN model** with a **Java backend integration**.

---

## Features

- **Java code analysis** for common code smells
- **CNN-based ML model** trained on Java source code
- **FastAPI REST API** serving predictions
- **Spring Boot backend** integration
- Top-2 smell predictions with probabilities
- Easy to extend to more smells or larger datasets

---

## Project Structure
ai-code-smell-detector/
├── api/ # Python FastAPI code serving predictions
├── training/ # Python ML pipeline and dataset preparation
├── src/main/java/ # Java backend (Spring Boot) integration
├── dataset/ # Sample Java code dataset for training
├── requirements.txt # Python dependencies
└── README.md


---

## Getting Started

### Prerequisites

- Python 3.10+  
- PyTorch  
- FastAPI & Uvicorn  
- Java 17+  
- Spring Boot 3+  

---

### Python Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Train the model (optional)
python -m training.train

# Run the FastAPI service
uvicorn api.predict_api:app --reload --port 8000


