# Functionality like setting up mongodb, database connection
#  We can call the functionality within the components.

import os,sys
import numpy as np
import pandas as pd
from src.exception_handler import CustomException
import dill
from src.logger import logging
from sklearn.metrics import r2_score


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        logging.info("Preprocessor object saved successfully...")
    except Exception as e:
        raise CustomException(e,sys)
# used in data transformation to save the pickle file of preprocessor

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        reports={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train) #model training

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            model_train_score=r2_score(y_train_pred,y_train)
            model_test_score=r2_score(y_test_pred,y_test)
            reports[list(models.keys())[i]]=model_test_score
        return reports


    except Exception as e:
        raise CustomException(e,sys)

