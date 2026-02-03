from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import sklearn
print(sklearn.__version__)

app = Flask(__name__)


lr_model = joblib.load("logistic_model.pkl")

nn_preprocessor = joblib.load("nn_preprocessor.pkl")
nn_model = load_model("nn_model.keras")
dt_model = joblib.load("decision_tree_model.pkl") 
rf_model = joblib.load("random_forest_model.pkl")
rf_columns = joblib.load("rf_columns.pkl")


def yes_no_to_int(value):
    return 1 if str(value).strip().lower() == "yes" else 0



def predict_logistic(age):
    X = pd.DataFrame([[age]], columns=["Age"])
    prediction = lr_model.predict(X)[0]
    return int(prediction)



def predict_neural_network(data):
    X = pd.DataFrame([{
        "Gender": data["gender"],                 # "F" or "M"
        "Age": int(data["age"]),
        "Neighbourhood": data["neighbourhood"],

        "Scholarship": yes_no_to_int(data["scholarship"]),
        "Hipertension": yes_no_to_int(data["hypertension"]),
        "Diabetes": yes_no_to_int(data["diabetes"]),
        "Alcoholism": yes_no_to_int(data["alcoholism"]),
        "Handcap": yes_no_to_int(data["handcap"]),
        "SMS_received": yes_no_to_int(data["sms_received"])
    }])

    X_processed = nn_preprocessor.transform(X)

    prob = nn_model.predict(X_processed, verbose=0)[0][0]

    return 1 if prob >= 0.5 else 0
def predict_decision_tree(data):
    X = pd.DataFrame([{
        "Gender": data["gender"],
        "Age": int(data["age"]),
        "Neighbourhood": data["neighbourhood"],
        "Hipertension": yes_no_to_int(data["hypertension"]),
        "Diabetes": yes_no_to_int(data["diabetes"]),
        "Alcoholism": yes_no_to_int(data["alcoholism"]),
        "Handcap": yes_no_to_int(data["handcap"]),
        "SMS_received": yes_no_to_int(data["sms_received"])
    }])

    prediction = dt_model.predict(X)[0]
    return 1 if str(prediction).strip().lower() == "yes" else 0
def predict_random_forest(data):
    X = pd.DataFrame([{
        "Gender": data["gender"],
        "Age": int(data["age"]),
        "Neighbourhood": data["neighbourhood"],
        "Hipertension": yes_no_to_int(data["hypertension"]),
        "Diabetes": yes_no_to_int(data["diabetes"]),
        "Alcoholism": yes_no_to_int(data["alcoholism"]),
        "Handcap": yes_no_to_int(data["handcap"]),
        "SMS_received": yes_no_to_int(data["sms_received"])
    }])

    
    X = pd.get_dummies(X, columns=["Gender", "Neighbourhood"], drop_first=True)

    X = X.reindex(columns=rf_columns, fill_value=0)

    prediction = rf_model.predict(X)[0]   # "Yes" or "No"

    return 1 if str(prediction).strip().lower() == "yes" else 0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/<model>", methods=["POST"])
def predict(model):
    data = request.json

    if model == "logistic_regression":
        result = predict_logistic(data["age"])

    elif model == "neural_network":
        result = predict_neural_network(data)
    elif model == "decision_tree":
        result = predict_decision_tree(data)
    elif model == "random_forest":
        result = predict_random_forest(data)

    else:
        return jsonify({"error": "Unknown model"}), 400

    return jsonify({
        "prediction": "Yes" if result == 1 else "No"
    })



if __name__ == "__main__":
    app.run(debug=True)