# Functionality like setting up mongodb, database connection
#  We can call the functionality within the components.

import os,sys
import numpy as np
import pandas as pd
from src.exception_handler import CustomException
import dill
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        logging.info(f"{file_path} object saved successfully...")
    except Exception as e:
        raise CustomException(e,sys)
# used in data transformation to save the pickle file of preprocessor

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        reports = {}
        for model_name, model in models.items():
            grid_search = GridSearchCV(model, params[model_name], cv=5)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            model_train_score = r2_score(y_train, y_train_pred)
            model_test_score = r2_score(y_test, y_test_pred)

            reports[model_name] = {
                "best_params": grid_search.best_params_,
                "train_score": model_train_score,
                "test_score": model_test_score
            }

        return reports

    except Exception as e:
        raise CustomException(e, sys)
    

def load_obj(filepath):
    try:
        with open(filepath, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

