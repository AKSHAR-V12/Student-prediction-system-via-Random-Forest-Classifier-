# =========================
# Import Required Libraries
# =========================

import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================
# Load Dataset
# =========================

BASE_DIR = Path(__file__).resolve().parent

csv_path = BASE_DIR / "student_performance_large_dataset.csv"

df = pd.read_csv(csv_path)


# =========================
# Feature Selection
# =========================

features = [

    'study_hours',
    'attendance',
    'assignments_completed',
    'sleep_hours',
    'screen_time',
    'mock_test_score',
    'motivation_level',
    'internet_usage_hours'
]

# Input Features
X = df[features]

# Target Output
y = df['result']


# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)


# =========================
# Create Model
# =========================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42
)


# =========================
# Train Model
# =========================

model.fit(X_train, y_train)


# =========================
# Predict On Test Data
# =========================

predictions = model.predict(X_test)


# =========================
# Model Evaluation
# =========================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)

f1 = f1_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)

report = classification_report(y_test, predictions)


# =========================
# Print Evaluation Results
# =========================

print("\n=========================")
print("MODEL EVALUATION RESULTS")
print("=========================")

print(f"\nAccuracy Score: {accuracy}")

print(f"\nPrecision Score: {precision}")

print(f"\nRecall Score: {recall}")

print(f"\nF1 Score: {f1}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)


# =========================
# Feature Importance
# =========================

print("\n=========================")
print("FEATURE IMPORTANCE")
print("=========================")

importance_values = model.feature_importances_

for feature, importance in zip(features, importance_values):

    print(f"{feature}: {round(importance, 4)}")


# =========================
# Custom Student Prediction
# =========================

sample_student = pd.DataFrame([{

    'study_hours': 8,

    'attendance': 90,

    'assignments_completed': 9,

    'sleep_hours': 7,

    'screen_time': 2,

    'mock_test_score': 85,

    'motivation_level': 9,

    'internet_usage_hours': 3
}])


# =========================
# Predict Result
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


print("\n=========================")
print("CUSTOM STUDENT ANALYSIS")
print("=========================")

print(f"\nPrediction Value: {prediction}")

print(f"\nResult: {result}")

print(f"\nConfidence: {confidence}%")


# =========================
# Strength Analysis
# =========================

strengths = []

if sample_student['study_hours'][0] >= 7:

    strengths.append(
        "Excellent study discipline detected."
    )

if sample_student['attendance'][0] >= 85:

    strengths.append(
        "Strong attendance consistency."
    )

if sample_student['mock_test_score'][0] >= 75:

    strengths.append(
        "High mock test performance."
    )

if sample_student['assignments_completed'][0] >= 8:

    strengths.append(
        "Excellent assignment completion rate."
    )

if sample_student['motivation_level'][0] >= 8:

    strengths.append(
        "Strong motivation level detected."
    )


# =========================
# Weakness Analysis
# =========================

weaknesses = []

if sample_student['study_hours'][0] <= 3:

    weaknesses.append(
        "Low study hours may negatively affect performance."
    )

if sample_student['attendance'][0] <= 65:

    weaknesses.append(
        "Poor attendance may reduce classroom understanding."
    )

if sample_student['screen_time'][0] >= 7:

    weaknesses.append(
        "High screen time may reduce concentration."
    )

if sample_student['sleep_hours'][0] <= 5:

    weaknesses.append(
        "Poor sleep habits may affect learning ability."
    )

if sample_student['mock_test_score'][0] <= 45:

    weaknesses.append(
        "Low mock test performance detected."
    )


# =========================
# Performance Score System
# =========================

performance_score = 0

performance_score += sample_student['study_hours'][0]

performance_score += sample_student['attendance'][0] / 10

performance_score += sample_student['mock_test_score'][0] / 10

performance_score += sample_student['motivation_level'][0]


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
# Print Strengths
# =========================

print("\nStrengths:")

for item in strengths:

    print(f"- {item}")


# =========================
# Print Weaknesses
# =========================

print("\nWeaknesses:")

for item in weaknesses:

    print(f"- {item}")


# =========================
# Print Suggestions
# =========================

print("\nAI Suggestions:")

for item in suggestions:

    print(f"- {item}")


# =========================
# Print Performance Level
# =========================

print(f"\nPerformance Level: {performance_level}")


# =========================
# Save Model
# =========================

joblib.dump(model, 'student_model.pkl')

print("\n=========================")
print("MODEL SAVED SUCCESSFULLY")
print("=========================")