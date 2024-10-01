import os,sys
from src.exception_handler import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #used to create class variables

# where i have to save the raw data, train and tesr data

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw_data.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started...")
        try:
            #Read the data
            df=pd.read_csv("data/raw_data.csv")
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
    obj.initiate_data_ingestion()

    