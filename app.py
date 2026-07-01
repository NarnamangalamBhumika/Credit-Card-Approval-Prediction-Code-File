from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# --------------------------------------------------
# Load Model, Scaler and Encoders
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "credit_card_approval_model.pkl"), "rb"))

print("Loading model from:", os.path.join(BASE_DIR, "credit_card_approval_model.pkl"))
print("Model expects:", model.n_features_in_)

scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

gender_encoder = pickle.load(open(os.path.join(BASE_DIR, "gender_encoder.pkl"), "rb"))
car_encoder = pickle.load(open(os.path.join(BASE_DIR, "car_encoder.pkl"), "rb"))
realty_encoder = pickle.load(open(os.path.join(BASE_DIR, "realty_encoder.pkl"), "rb"))
income_encoder = pickle.load(open(os.path.join(BASE_DIR, "income_encoder.pkl"), "rb"))
education_encoder = pickle.load(open(os.path.join(BASE_DIR, "education_encoder.pkl"), "rb"))
family_encoder = pickle.load(open(os.path.join(BASE_DIR, "family_encoder.pkl"), "rb"))
housing_encoder = pickle.load(open(os.path.join(BASE_DIR, "housing_encoder.pkl"), "rb"))
occupation_encoder = pickle.load(open(os.path.join(BASE_DIR, "occupation_encoder.pkl"), "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------
        # Read Inputs
        # -------------------------------

        gender = gender_encoder.transform([request.form["gender"]])[0]

        own_car = car_encoder.transform([request.form["own_car"]])[0]

        own_realty = realty_encoder.transform([request.form["own_realty"]])[0]

        income = float(request.form["income"])

        income_type = income_encoder.transform(
            [request.form["income_type"]])[0]

        education = education_encoder.transform(
            [request.form["education"]])[0]

        family_status = family_encoder.transform(
            [request.form["family_status"]])[0]

        housing = housing_encoder.transform(
            [request.form["housing"]])[0]

        occupation = occupation_encoder.transform(
            [request.form["occupation"]])[0]

        family_members = float(request.form["family_members"])

        age = int(request.form["age"])

        years_employed = float(request.form["years_employed"])

        # -------------------------------
        # Create DataFrame
        # -------------------------------

        data = pd.DataFrame([[
            gender,
            own_car,
            own_realty,
            income,
            income_type,
            education,
            family_status,
            housing,
            occupation,
            family_members,
            age,
            years_employed

        ]], columns=[

            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "AMT_INCOME_TOTAL",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
            "CNT_FAM_MEMBERS",
            "AGE",
            "YEARS_EMPLOYED"

        ])

        # -------------------------------
        # Scale Features
        # -------------------------------

        data_scaled = scaler.transform(data)

        # -------------------------------
        # Prediction
        # -------------------------------

        prediction = model.predict(data_scaled)

        if prediction[0] == 1:
            result = "Credit Card Approved ✅"
            color = "green"
        else:
            result = "Credit Card Rejected ❌"
            color = "red"

        return render_template(
            "index.html",
            prediction_text=result,
            color=color
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=f"Error : {str(e)}",
            color="red"
        )


if __name__ == "__main__":
    app.run(debug=True)