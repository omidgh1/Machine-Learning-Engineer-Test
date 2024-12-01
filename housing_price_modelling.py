import logging
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import numpy as np
import pandas as pd
import yaml
from app.preprocessing import DataPreprocessor

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def train_model(X_train, y_train, type, hyperparameters):
    """
    Train model
    """
    model = eval(type)()
    model.set_params(**hyperparameters)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Perform model evaluation , R2 and MAE
    """
    metrics = {}

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics['train_r2'] = r2_score(y_train, y_pred_train)
    metrics['test_r2'] = r2_score(y_test, y_pred_test)
    metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
    metrics['test_mae'] = mean_absolute_error(y_test, y_pred_test)

    return metrics

def hyperparameter_tuning(X_train, y_train, cv=5, config=None):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    model = eval(config.get('model')['type'])()
    grid_search = GridSearchCV(model, config.get('parameters'), cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def save_model(model, filepath):
    """
    Save a model to a file.
    """
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")

def main(config_path):
    config = load_config(config_path)

    dataset_path = config.get('data')['path']
    preprocessing_config = config.get('preprocessing')
    test_size = config.get('training')['test_size']
    random_state = config.get('training')['random_state']
    hyperparameter = config.get('model')['hyperparameters']
    model_type = config.get('model')['type']
    tuning = config.get('tuning')

    df = pd.read_csv(dataset_path)
    X = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
            'population', 'households', 'median_income', 'ocean_proximity']]
    y = df['median_house_value']

    # Preprocess data
    preprocessor = DataPreprocessor(**preprocessing_config)
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=test_size, random_state=random_state)

    model = train_model(X_train, y_train, model_type, hyperparameter)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    logging.info(f"Model evaluation metrics: {metrics}")

    if tuning.get('tuning'):
        best_params = hyperparameter_tuning(X_train, y_train, cv=5, config=tuning.get('parameters'))
        model = train_model(X_train, y_train, model_type, **best_params)
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        logging.info(f"Model evaluation metrics after tuning: {metrics}")

    save_model(model, 'random_forest_model.pkl')

if __name__ == "__main__":
    main("config.yaml")