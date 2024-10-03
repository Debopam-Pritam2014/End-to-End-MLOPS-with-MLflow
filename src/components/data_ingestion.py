import os,sys
from src.exception_handler import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #used to create class variables

# testing the data_transformation.py
from src.components.data_transformation import DataTransformation,DataTransformationConfig

# testing the model_trainer.py
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig


# where i have to save the raw data, train and tesr data

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","complete_data.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started...")
        try:
            #Read the data
            df=pd.read_csv("data/complete_data.csv")
            logging.info("Read dataset as dataframe..")
            # make directory for train, test and raw
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path),exist_ok=True)

            logging.info(f"raw data stored in: {self.data_ingestion_config.raw_data_path}")
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            logging.info(f"Train test split initiated...")
            train_data,test_data=train_test_split(df,test_size=0.2,random_state=1)
            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info(f"train and test data ingested succesfully.")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_df,test_df=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path=train_df,test_data_path=test_df)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr))


    