# Credit Card Approval Prediction

## Project Overview

This project is a Machine Learning-based Credit Card Approval Prediction system developed using Python, Flask, and XGBoost. The application predicts whether a customer's credit card application will be approved or rejected based on applicant details.

---

## Problem Statement

Financial institutions receive numerous credit card applications every day. Manually evaluating every application is time-consuming and may lead to inconsistencies. This project automates the approval prediction process using Machine Learning.

---

## Features

- Data Cleaning and Preprocessing
- Feature Engineering (Age & Years Employed)
- Label Encoding
- Feature Scaling using StandardScaler
- XGBoost Machine Learning Model
- Flask Web Application
- User-Friendly Interface
- Real-Time Prediction
- Model Persistence using Pickle

---

## Technologies Used

- Python
- Flask
- HTML
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Pickle

---

## Installation

1. Download or clone the repository from girhub

2. Navigate to the project folder:

```bash
cd Credit-Card-Approval-Prediction
```

3. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

4. Run the Flask application:

```bash
python app.py
```

5. Open your browser and visit:

```
http://127.0.0.1:5000
```

6. Enter the applicant details and click **Predict** to view the credit card approval result.

---

## Project Structure

```
Credit-Card-Approval-Prediction-Code-File/
│
├── app.py
├── Credit_Card_Approval_Prediction.ipynb
├── README.md
├── requirements.txt
├── credit_card_approval_model.pkl
├── scaler.pkl
├── gender_encoder.pkl
├── car_encoder.pkl
├── realty_encoder.pkl
├── income_encoder.pkl
├── education_encoder.pkl
├── family_encoder.pkl
├── housing_encoder.pkl
├── occupation_encoder.pkl
│
├── templates/
│   └── index.html
│
├── dataset/
│   ├── application_record.csv
│   └── credit_record.csv
│
└── screenshots/
    ├── home_page.png
    ├── prediction_result.png
```
---

## Dataset

The dataset used for this project exceeds GitHub's web upload size limit (25 MB), so it is not included in this repository.

The dataset can be downloaded from Kaggle:

Credit Card Approval Prediction:
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction