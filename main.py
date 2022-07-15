import os
import joblib


import pandas as pd

from fastapi import FastAPI
from data_model import ModelInput
from starter.ml.data import process_data

file_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(file_path, './model/model.joblib'))
encoder = joblib.load(os.path.join(file_path, './model/encoder.joblib'))
lb = joblib.load(os.path.join(file_path, './model/lb.joblib'))
cat_features = [
"workclass",
"education",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"native-country",
]






app = FastAPI()


@app.get("/")
def read_root():
    return {"Greeting": "Hello"}


@app.post('/inference/')
def inference(model_input: ModelInput):
    df = pd.DataFrame(model_input.dict(by_alias=True), index=[0])

    X, _, _, _ = process_data(
            df,
            categorical_features=cat_features,
            training=False,
            encoder=encoder,
            lb=lb)

    prediction = model.predict(X)
    
    if prediction[0] == 1:
        prediction = "Salary > 50k"
    else:
        prediction = "Salary <= 50k"
    return {"prediction": prediction}

if __name__ == '__main__':
    example = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
    }
    r = inference(example)
    print(r)
