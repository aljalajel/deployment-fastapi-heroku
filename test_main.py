import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_greeting():
    r = client.get('/')
    print(r.json())
    assert r.status_code == 200
    assert r.json() == {'Greeting': 'Hello'}


def test_post_less_than_50():
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
    data = json.dumps(example)
    r = client.post("/inference/", data=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json()["Salary"] == '<= 50k'


def test_post_greater_than_50():
    example = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    data = json.dumps(example)
    r = client.post("/inference/", data=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json()["Salary"] == '> 50k'
