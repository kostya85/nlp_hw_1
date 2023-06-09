
import logging

import joblib
import optuna
import pandas as pd
from dvc.api import make_checkpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data(file_path):
       return pd.read_csv(file_path, delimiter=';')

def preprocess_data(data_frame):
    data_frame['isHate'] = data_frame['isHate'].astype('int')

    X_train, X_test, y_train, y_test = train_test_split(data_frame['comment'], data_frame['isHate'], test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(stop_words='english')
    X_train_transformed = tfidf.fit_transform(X_train)
    X_test_transformed = tfidf.transform(X_test)

    return X_train_transformed, X_test_transformed, y_train, y_test

def train_model(X_train, y_train, trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, 100)
    max_depth = trial.suggest_int('max_depth', 1, 10, 1)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10, 1)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10, 1)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
       y_pred = model.predict(X_test)

       score = accuracy_score(y_test, y_pred)

       return score

def objective(trial, X_train, X_test, y_train, y_test):
    model = train_model(X_train, y_train, trial)
    score = evaluate_model(model, X_test, y_test)
    return score

def run():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("output.log")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info("Loading data")
    data = load_data("Ethos_Dataset_Binary.csv")
    logger.debug(f"data shape: {data.shape}")

    logger.info("Preprocessing data")
    X_train, X_test, y_train, y_test = preprocess_data(data)

    logger.info("Optimizing hyperparameters")
    study = optuna.create_study(direction='maximize')
    objective_function = lambda trial: objective(trial, X_train, X_test, y_train, y_test)
    study.optimize(objective_function, n_trials=50)

    logger.debug(f"Best hyperparameters: {study.best_params}")
    logger.debug(f"Best score: {study.best_value}")

    logger.info("Training model with optimal hyperparameters")
    model = train_model(X_train, y_train, study.best_trial)
    logger.debug(f"Model score on training data: {model.score(X_train, y_train)}")
    logger.debug(f"Model score on test data: {model.score(X_test, y_test)}")

    logger.info("Saving model")
    model_version = 'v1'
    output_path = f'models_output/model_{model_version}.joblib'
    joblib.dump(model, output_path)

    logger.info("Creating model checkpoint")
    checkpoint_message = "Model training and optimization completed."
    make_checkpoint()

if __name__ == '__main__':
    run()