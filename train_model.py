
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Starting model training...")

data = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5,
                    6, 3, 7, 4, 8, 1, 9, 6, 3, 7],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 90, 60, 85,
                   78, 58, 88, 68, 95, 45, 98, 82, 62, 90],
    "sleep_hours": [5, 6, 7, 6, 8, 7, 8, 7, 5, 7,
                    6, 8, 7, 6, 8, 5, 8, 7, 6, 8],
    "score": [35, 40, 48, 55, 65, 72, 80, 92, 42, 75,
              74, 47, 85, 58, 96, 30, 98, 78, 50, 88]
})

X = data[["study_hours", "attendance", "sleep_hours"]]
y = data["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    LinearRegression()
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 score:", round(r2_score(y_test, predictions), 2))

joblib.dump(model, "student_model.joblib")

print("Model saved successfully!")
print("Training completed.")
