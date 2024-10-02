import os,sys
from src.logger import logging
from src.exception_handler import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,ExtraTreesRegressor,
                              AdaBoostRegressor,GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object,evaluate_model
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge":Ridge(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Ada Boost":AdaBoostRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
                "Extra Tree":ExtraTreesRegressor(),
                "K-Neighbors Regressor":KNeighborsRegressor()
            }
            # eval_model function is present inside utils
            models_report:str=evaluate_model(X_train,y_train,X_test,y_test,models)

            #to get the best model score
            best_model_score=max(sorted(models_report.values()))

            # to get the best model name
            best_model_name=list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info("Best model found and saving.....")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            prediction=best_model.predict(X_test)
            score=r2_score(y_test,prediction)
            # return score
            return (score,best_model_name)


        
        except Exception as e:
            raise CustomException(e,sys)
