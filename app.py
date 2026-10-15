
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("student_model.joblib")


@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        sleep_hours = float(request.form["sleep_hours"])

        if not (0 <= study_hours <= 24):
            raise ValueError("Study hours must be between 0 and 24.")

        if not (0 <= attendance <= 100):
            raise ValueError("Attendance must be between 0 and 100.")

        if not (0 <= sleep_hours <= 24):
            raise ValueError("Sleep hours must be between 0 and 24.")

        features = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "sleep_hours": sleep_hours
        }])

        score = float(model.predict(features)[0])
        prediction = round(max(0, min(100, score)), 2)

        return render_template(
            "index.html", prediction=prediction
        )

    except (ValueError, KeyError):
        return "Invalid input. Please enter valid values.", 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Send a JSON object."}), 400

        study_hours = float(data["study_hours"])
        attendance = float(data["attendance"])
        sleep_hours = float(data["sleep_hours"])

        if not (0 <= study_hours <= 24
                and 0 <= attendance <= 100
                and 0 <= sleep_hours <= 24):
            return jsonify({"error": "Input values are out of range."}), 400

        features = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "sleep_hours": sleep_hours
        }])

        score = float(model.predict(features)[0])
        prediction = round(max(0, min(100, score)), 2)

        return jsonify({"predicted_score": prediction})

    except (ValueError, TypeError, KeyError):
        return jsonify({"error": "Invalid or missing input."}), 400


if __name__ == "__main__":
    app.run(debug=True)
