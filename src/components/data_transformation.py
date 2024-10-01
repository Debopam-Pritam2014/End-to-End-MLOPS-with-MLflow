from src.logger import logging
from src.exception_handler import CustomException
import os,sys
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from src.utils import save_object


from dataclasses import dataclass

# In the DataTransformationConfig we basically provide the input file path 
@dataclass
class DataTransformationConfig:
    data_preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self) :
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            # Create the pipeline for numerical and categorical features
            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed.....")
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(sparse_output=False)),
                    ("scalar",StandardScaler())
                ]
            )
            logging.info("Categorical columns encoding completed.....")
            # now we will add transformation of both numerical and categorical
            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,categorical_columns)
                ]
            )
        except Exception as e:
            raise CustomException(e,sys)
        return preprocessor
    
    def initiate_data_transformation(self,train_data_path,test_data_path):
        logging.info("Data transformation initiated....")

        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            logging.info("Read train and test data completed......")
            logging.info("Obtaining preprocessor object......")
            preprocessor_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing to training and testing dataframes....")
            input_feature_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_features_test_df)
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object....")

            save_object(
                file_path=self.data_transformation_config.data_preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.data_preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e,sys)


