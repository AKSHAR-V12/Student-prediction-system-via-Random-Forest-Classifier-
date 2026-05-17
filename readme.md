# AKSHAR Student Performance Analyzer 🎓

A full-stack AI-powered web application designed to analyze student performance using machine learning.  
Built with **FastAPI** (backend) and a custom **HTML frontend**, this project demonstrates end-to-end skills in API development, ML model integration, and UI design.

---

## 🚀 Features
- **Machine Learning Model**: RandomForestClassifier trained on student performance data (`aiml.py`).
- **Backend API**: FastAPI with endpoints for prediction (`api.py`).
- **Frontend UI**: Interactive sliders and input fields (`index.html`) for user-friendly data entry.
- **Deployment Ready**: Configured for Render/Railway with `requirements.txt` and `Procfile`.
- **CORS Enabled**: Secure communication between frontend and backend.
- **Prediction Output**: Pass/Fail classification with confidence, performance level, strengths, weaknesses, and suggestions.

---

## 🛠️ Tech Stack
- **Backend**: FastAPI, Uvicorn
- **ML/AI**: Scikit-learn, Pandas, NumPy, Joblib
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render / Railway (HTTPS ready)

---

## 📂 Project Structure
student_project/
│
├── api.py              # FastAPI entry point
├── aiml.py             # Model training script
├── student_model.pkl   # Trained ML model
├── requirements.txt    # Dependencies
├── Procfile            # Deployment run command
├── frontend/
│   └── index.html      # User interface
└── README.md           # Project documentation



---

## ⚡ How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/student-analyzer-api.git
   cd student-analyzer-api

# install dependencies
pip install -r requirements.txt


<!-- start fastapi server -->
uvicorn api:app --reload






