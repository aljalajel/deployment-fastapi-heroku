# Script to train machine learning model.
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics



if __name__ == '__main__':
    
    file_path = os.path.dirname(__file__)
    # Add code to load in the data.
    data = pd.read_csv(os.path.join(file_path, "../data/census.csv"))
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=42)

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
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train and save a model.
    trained_model = train_model(X_train, y_train)
    joblib.dump(trained_model, os.path.join(file_path, '../model/model.joblib'))
    joblib.dump(encoder, os.path.join(file_path,'../model/encoder.joblib'))
    joblib.dump(lb, os.path.join(file_path,'../model/lb.joblib'))