import os
import joblib
import pandas as pd
from sklearn import metrics

from data import process_data
from model import inference, compute_model_metrics

def compute_slice_metrics(df,
                        categorical_features,
                        category, 
                        model,
                        label,
                        encoder,
                        lb,
                        ):
    slice_metrics: list = []
    for slice in df[category].unique():
        temp_df = df[df[category] == slice]
        X, y, _, _ = process_data(temp_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
        )
        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)
        num_records = len(temp_df)
        metrics_row = [category, slice,  num_records, precision, recall, fbeta]
        slice_metrics.append(metrics_row)
    return pd.DataFrame(slice_metrics, columns=['Category', 'Slice', 'No. Records', 'Precesion', 'Recall', 'FBeta' ])
    



if __name__ == '__main__':
    file_path = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(file_path, '../../data/census.csv'))
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
    category = 'education'
    model = joblib.load(os.path.join(file_path, '../../model/model.joblib'))
    encoder = joblib.load(os.path.join(file_path, '../../model/encoder.joblib'))
    lb = joblib.load(os.path.join(file_path, '../../model/lb.joblib'))
    slice_metrics = compute_slice_metrics(
        data,
        cat_features,
        category,
        model,
        label='salary',
        encoder=encoder,
        lb=lb,
    )
    slice_metrics.to_csv(os.path.join(file_path, '../../model/slice_output.csv'))