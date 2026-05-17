from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib


# =========================
# Load Saved Model
# =========================

model = joblib.load('student_model.pkl')


# =========================
# Feature Names
# =========================

feature_names = [

    'study_hours',
    'attendance',
    'assignments_completed',
    'sleep_hours',
    'screen_time',
    'mock_test_score',
    'motivation_level',
    'internet_usage_hours'
]


# =========================
# Create FastAPI App
# =========================

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Hello from FastAPI"}

# =========================
# Enable CORS
# =========================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================
# Feature Classification Functions
# =========================

def classify_study_hours(value):

    if value < 2:
        return "Bad ❌"

    elif value <= 5:
        return "Medium ⚠"

    else:
        return "Good ✅"


def classify_attendance(value):

    if value < 60:
        return "Bad ❌"

    elif value <= 80:
        return "Medium ⚠"

    else:
        return "Good ✅"


def classify_assignments(value):

    if value < 4:
        return "Bad ❌"

    elif value <= 7:
        return "Medium ⚠"

    else:
        return "Good ✅"


def classify_sleep_hours(value):

    if value < 5:
        return "Bad ❌"

    elif value <= 7:
        return "Medium ⚠"

    else:
        return "Good ✅"


def classify_screen_time(value):

    if value > 8:
        return "Bad ❌"

    elif value >= 4:
        return "Medium ⚠"

    else:
        return "Good ✅"


def classify_mock_test_score(value):

    if value < 40:
        return "Bad ❌"

    elif value <= 70:
        return "Medium ⚠"

    else:
        return "Good ✅"


def classify_motivation_level(value):

    if value < 4:
        return "Bad ❌"

    elif value <= 7:
        return "Medium ⚠"

    else:
        return "Good ✅"


def classify_internet_usage(value):

    if value > 8:
        return "Bad ❌"

    elif value >= 4:
        return "Medium ⚠"

    else:
        return "Good ✅"


# =========================
# Prediction Route
# =========================

@app.post("/predict")

def predict(data: dict):

    # =========================
    # Convert Input Into DataFrame
    # =========================

    sample_student = pd.DataFrame([{

        'study_hours': data['study_hours'],

        'attendance': data['attendance'],

        'assignments_completed': data['assignments_completed'],

        'sleep_hours': data['sleep_hours'],

        'screen_time': data['screen_time'],

        'mock_test_score': data['mock_test_score'],

        'motivation_level': data['motivation_level'],

        'internet_usage_hours': data['internet_usage_hours']
    }])

    # =========================
    # Prediction
    # =========================

    prediction = model.predict(sample_student)[0]

    # =========================
    # Prediction Probability
    # =========================

    probabilities = model.predict_proba(sample_student)[0]

    confidence = round(max(probabilities) * 100, 2)

    # =========================
    # Human Readable Result
    # =========================

    if prediction == 1:

        result = "The student is likely to pass."

    else:

        result = "The student is likely to fail."

    # =========================
    # Feature Analysis
    # =========================

    analysis = {

        "study_hours":
        classify_study_hours(data['study_hours']),

        "attendance":
        classify_attendance(data['attendance']),

        "assignments_completed":
        classify_assignments(data['assignments_completed']),

        "sleep_hours":
        classify_sleep_hours(data['sleep_hours']),

        "screen_time":
        classify_screen_time(data['screen_time']),

        "mock_test_score":
        classify_mock_test_score(data['mock_test_score']),

        "motivation_level":
        classify_motivation_level(data['motivation_level']),

        "internet_usage_hours":
        classify_internet_usage(data['internet_usage_hours'])
    }

    # =========================
    # Strength Analysis
    # =========================

    strengths = []

    if data['study_hours'] >= 7:

        strengths.append(
            "Excellent study discipline detected."
        )

    if data['attendance'] >= 85:

        strengths.append(
            "Strong attendance consistency."
        )

    if data['mock_test_score'] >= 75:

        strengths.append(
            "High mock test performance."
        )

    if data['assignments_completed'] >= 8:

        strengths.append(
            "Excellent assignment completion rate."
        )

    if data['motivation_level'] >= 8:

        strengths.append(
            "Strong motivation level detected."
        )

    # =========================
    # Weakness Analysis
    # =========================

    weaknesses = []

    if data['study_hours'] <= 3:

        weaknesses.append(
            "Low study hours may negatively affect performance."
        )

    if data['attendance'] <= 65:

        weaknesses.append(
            "Poor attendance may reduce classroom understanding."
        )

    if data['screen_time'] >= 7:

        weaknesses.append(
            "High screen time may reduce concentration."
        )

    if data['sleep_hours'] <= 5:

        weaknesses.append(
            "Poor sleep habits may affect learning ability."
        )

    if data['mock_test_score'] <= 45:

        weaknesses.append(
            "Low mock test performance detected."
        )

    # =========================
    # Performance Score System
    # =========================

    performance_score = 0

    performance_score += data['study_hours']

    performance_score += data['attendance'] / 10

    performance_score += data['mock_test_score'] / 10

    performance_score += data['motivation_level']

    # =========================
    # AI Suggestions System
    # =========================

    suggestions = []

    performance_level = ""

    # -------------------------
    # Excellent Student
    # -------------------------

    if performance_score >= 30:

        performance_level = "EXCELLENT"

        suggestions.append(
            "Maintain your current academic consistency."
        )

        suggestions.append(
            "Start solving advanced-level academic problems."
        )

        suggestions.append(
            "Participate in olympiads and competitive exams."
        )

        suggestions.append(
            "Develop leadership and communication skills."
        )

        suggestions.append(
            "Explore AI, coding, and future technical skills."
        )

    # -------------------------
    # Good Student
    # -------------------------

    elif performance_score >= 22:

        performance_level = "GOOD"

        suggestions.append(
            "Improve study consistency further."
        )

        suggestions.append(
            "Practice more mock tests weekly."
        )

        suggestions.append(
            "Reduce distractions during study sessions."
        )

        suggestions.append(
            "Focus more on conceptual understanding."
        )

        suggestions.append(
            "Improve time management skills."
        )

    # -------------------------
    # Average Student
    # -------------------------

    elif performance_score >= 15:

        performance_level = "AVERAGE"

        suggestions.append(
            "Create a fixed daily study routine."
        )

        suggestions.append(
            "Improve attendance consistency."
        )

        suggestions.append(
            "Reduce unnecessary screen time."
        )

        suggestions.append(
            "Increase revision frequency."
        )

        suggestions.append(
            "Improve sleep schedule and discipline."
        )

    # -------------------------
    # Weak Student
    # -------------------------

    else:

        performance_level = "NEEDS IMPROVEMENT"

        suggestions.append(
            "Increase daily study hours immediately."
        )

        suggestions.append(
            "Improve classroom attendance urgently."
        )

        suggestions.append(
            "Reduce excessive entertainment screen usage."
        )

        suggestions.append(
            "Seek academic mentoring and guidance."
        )

        suggestions.append(
            "Focus on rebuilding basic concepts."
        )

    # =========================
    # AI Summary System
    # =========================

    if prediction == 1 and confidence >= 90:

        summary = (

            f"The AI system predicts strong academic performance "
            f"with {confidence}% confidence. "

            "The student demonstrates highly positive academic indicators."
        )

    elif prediction == 1:

        summary = (

            f"The AI system predicts that the student is likely to pass "
            f"with {confidence}% confidence. "

            "Some areas can still be optimized for stronger consistency."
        )

    else:

        summary = (

            f"The AI system predicts possible academic struggle "
            f"with only {confidence}% confidence toward success. "

            "Several academic indicators require improvement."
        )

    # =========================
    # Return Response
    # =========================

    return {

        "prediction": int(prediction),

        "result": result,

        "confidence": confidence,

        "performance_level": performance_level,

        "summary": summary,

        "strengths": strengths,

        "weaknesses": weaknesses,

        "suggestions": suggestions,

        "analysis": analysis
    }


# =========================
# Feature Importance Route
# =========================

@app.get("/feature-importance")

def feature_importance():

    importance_values = model.feature_importances_

    importance_data = {}

    for feature, value in zip(feature_names, importance_values):

        importance_data[feature] = round(float(value), 4)

    return {

        "feature_importance": importance_data
    }


# =========================
# Model Stats Route
# =========================

@app.get("/model-stats")

def model_stats():

    return {

        "model_type": "Random Forest Classifier",

        "trees_used": 100,

        "status": "ACTIVE",

        "version": "2.0"
    }
