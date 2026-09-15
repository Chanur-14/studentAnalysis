# Import required libraries
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field
import pandas as pd
import numpy as np
import uvicorn


# Create the FastAPI application
app = FastAPI(
    title="Student Academic Risk Intelligence System API",
    description="API for analyzing student performance data",
    version="1.0.0"
)


def load_data():
    # Load the Maths.csv dataset from the data folder
    df = pd.read_csv("data/Maths.csv")

    # Create Result based on the final grade (G3)
    # G3 = 0: Dropout, 1-9: Fail, 10-20: Pass
    def get_result(g3):
        if g3 == 0:
            return "Dropout"
        elif 1 <= g3 <= 9:
            return "Fail"
        else:
            return "Pass"

    df["Result"] = df["G3"].apply(get_result)

    # Convert G3 into a percentage
    df["Percentage"] = (df["G3"] / 20) * 100

    # Calculate average alcohol consumption
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Calculate average education level of both parents
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Calculate the grade trend from G1 to G3
    df["grade_trend"] = df["G3"] - df["G1"]

    # Count "yes" values across support-related columns
    support_columns = ["schoolsup", "famsup", "paid"]
    df["total_support"] = (
        df[support_columns]
        .apply(lambda row: (row == "yes").sum(), axis=1)
    )

    # Calculate the academic risk score
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # Calculate the average of G1 and G2
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Return the prepared DataFrame
    return df


# Load and prepare the data when the application starts
df = load_data()

# Endpoint 1: Get overall student performance summary
@app.get("/summary")
def get_summary():
    # Exclude dropouts (G3 = 0) for average and pass rate calculations
    non_dropout = df[df["G3"] != 0]

    # Calculate total number of students
    total_students = len(df)

    # Calculate class average G3 among non-dropout students
    class_average_g3 = non_dropout["G3"].mean()

    # Calculate pass rate among non-dropout students
    pass_count = (non_dropout["G3"] >= 10).sum()
    pass_rate_percent = (pass_count / len(non_dropout)) * 100

    # Count at-risk students (G3 between 1 and 9)
    at_risk_count = ((df["G3"] >= 1) & (df["G3"] <= 9)).sum()

    # Count dropout students (G3 = 0)
    dropout_count = (df["G3"] == 0).sum()

    # Return the summary as JSON
    return {
        "total_students": int(total_students),
        "class_average_g3": round(float(class_average_g3), 2),
        "pass_rate_percent": round(float(pass_rate_percent), 2),
        "at_risk_count": int(at_risk_count),
        "dropout_count": int(dropout_count)
    }


# Endpoint 2: Get all at-risk students
@app.get("/at-risk")
def get_at_risk_students():
    # Filter students whose G3 is between 1 and 9
    at_risk = df[(df["G3"] >= 1) & (df["G3"] <= 9)]

    # Sort by G3 in ascending order, so worst-performing students come first
    at_risk = at_risk.sort_values("G3", ascending=True)

    # Build the response list
    students = []

    for index, row in at_risk.iterrows():
        students.append({
            "student_index": int(index),
            "G1": int(row["G1"]),
            "G2": int(row["G2"]),
            "G3": int(row["G3"]),
            "absences": int(row["absences"])
        })

    # Return the list of at-risk students
    return students


# Endpoint 3: Get the top 5 students
@app.get("/top-students")
def get_top_students():
    # Exclude dropouts (G3 = 0)
    non_dropout = df[df["G3"] != 0]

    # Sort students by G3 in descending order
    top_students = non_dropout.sort_values("G3", ascending=False).head(5)

    # Build the response list
    students = []

    for index, row in top_students.iterrows():
        students.append({
            "student_index": int(index),
            "G1": int(row["G1"]),
            "G2": int(row["G2"]),
            "G3": int(row["G3"])
        })

    # Return the top 5 students
    return students

    # Pydantic model for validating student input
class StudentInput(BaseModel):
    # G1 must be between 0 and 20
    G1: float = Field(
        ...,
        ge=0,
        le=20,
        description="G1 must be between 0 and 20"
    )

    # G2 must be between 0 and 20
    G2: float = Field(
        ...,
        ge=0,
        le=20,
        description="G2 must be between 0 and 20"
    )

    # Study time must be between 1 and 4
    studytime: int = Field(
        ...,
        ge=1,
        le=4,
        description="Studytime must be between 1 and 4"
    )

    # Absences must be between 0 and 100
    absences: int = Field(
        ...,
        ge=0,
        le=100,
        description="Absences must be between 0 and 100"
    )

    # Failures must be between 0 and 4
    failures: int = Field(
        ...,
        ge=0,
        le=4,
        description="Failures must be between 0 and 4"
    )


# Endpoint: Predict the student's final result
@app.post("/predict-result")
def predict_result(student: StudentInput):
    # Calculate the estimated G3 using the given formula
    estimated_g3 = (
        (student.G1 * 0.3)
        + (student.G2 * 0.6)
        + (student.studytime * 0.3)
        - (student.failures * 1.5)
        - (student.absences * 0.05)
    )

    # Clamp estimated G3 between 0 and 20
    estimated_g3 = max(0, min(20, estimated_g3))

    # Determine the predicted result
    if estimated_g3 == 0:
        prediction = "Dropout Risk"
    elif estimated_g3 < 10:
        prediction = "Fail"
    else:
        prediction = "Pass"

    # Determine prediction confidence
    if (student.G1 > 12 and student.G2 > 12) or (
        student.G1 < 8 and student.G2 < 8
    ):
        confidence = "High"
    else:
        confidence = "Medium"

    # Return the prediction details
    return {
        "estimated_g3": round(estimated_g3, 2),
        "prediction": prediction,
        "confidence": confidence
    }


    # Root endpoint: Provides basic API information
@app.get("/")
def root():
    # Return API name, documentation path, and version
    return {
        "message": "Student Academic Risk Intelligence System API",
        "docs": "Visit /docs for full API documentation",
        "version": "1.0.0"
    }


# Main block to start the FastAPI server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )