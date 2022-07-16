import joblib
import pytest
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from starter.ml.model import (train_model, compute_model_metrics)
from starter.ml.data import process_data


@pytest.fixture()
def train_test():
    file_path = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(file_path, '../../data/census.csv'))
    train, test = train_test_split(df, test_size=0.20, random_state=42)
    return train


def test_train_model(train_test):
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

    X_train, y_train, encoder, lb = process_data(
        train_test, categorical_features=cat_features, label="salary", training=True)

    trained_model = train_model(X_train, y_train)

    assert isinstance(trained_model, RandomForestClassifier)


def test_compute_model_metrics():
    y = np.array([0, 1, 0])
    preds = np.array([0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference():
    file_path = os.path.dirname(__file__)
    model = joblib.load(os.path.join(file_path, '../../model/model.joblib'))
    encoder = joblib.load(
        os.path.join(
            file_path,
            '../../model/encoder.joblib'))
    lb = joblib.load(os.path.join(file_path, '../../model/lb.joblib'))
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

    df = pd.DataFrame(example, index=[0])

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb)

    preds = model.predict(X)
    print(preds)
    assert isinstance(preds, np.ndarray)
