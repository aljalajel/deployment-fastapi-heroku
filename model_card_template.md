# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Developer: Majed Aljalajel
- Type: Random Forest Classifier, default sciket-learn hyperparameters


## Intended Use
- Udacity project (Deploying a Machine Learning Model on Heroku with FastAPI)

## Training Data
- [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income)
- 80% of this data is used for training

## Evaluation Data
- 20% of the [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income) is used for testing

## Metrics
- Precision: 75%
- Recall: 64%
- FBeta: 69%

## Ethical Considerations
- Native country, race and sex in the training dataset could bias the towards some groups. 

## Caveats and Recommendations
- Old dataset (1994 census data)