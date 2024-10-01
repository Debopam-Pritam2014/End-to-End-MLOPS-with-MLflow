# Functionality like setting up mongodb, database connection
#  We can call the functionality within the components.

import os,sys
import numpy as np
import pandas as pd
from src.exception_handler import CustomException
import dill
from src.logger import logging


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

