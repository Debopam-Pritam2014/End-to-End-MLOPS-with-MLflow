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

            params = {
                    "Linear Regression": {
                        "fit_intercept": [True, False],
                        # "normalize": [True, False]
                    },
                    "Lasso": {
                        "alpha": [0.1, 0.5, 1, 5],
                        # "max_iter": [1000, 5000, 10000],
                        # "tol": [0.001, 0.01, 0.1],
                        # "random_state": [42]
                    },
                    "Ridge": {
                        "alpha": [0.1, 0.5, 1,5],
                        # "max_iter": [20,50,100,200,300,500],
                        # "tol": [0.001, 0.002,0.0001,0.0008, 0.1],
                        # "random_state": [42]
                    },
                    "Decision Tree": {
                        "max_depth": [None, 3, 5, 10],
                        "min_samples_split": [2, 5],
                        # "min_samples_leaf": [1, 5, 10],
                        # "random_state": [42]
                    },
                    "Random Forest": {
                        "n_estimators": [ 50, 100],
                        "max_depth": [None, 3, 5],
                        # "min_samples_split": [2, 5, 10],
                        # "min_samples_leaf": [1, 5, 10],
                        # "random_state": [42]
                    },
                    "Ada Boost": {
                        "n_estimators": [ 50, 100],
                        "learning_rate": [0.1, 0.2],
                        # "loss": ["linear", "square", "exponential"],
                        # "random_state": [42]
                    },
                    "Gradient Boost": {
                        "n_estimators": [50, 100],
                        # "learning_rate": [0.1, 0.5, 1],
                        # "max_depth": [3, 5, 10],
                        # "min_samples_split": [2, 5, 10],
                        # "min_samples_leaf": [1, 5, 10],
                        # "random_state": [42]
                    },
                    "Extra Tree": {
                        "n_estimators": [10, 50, 100, 200],
                        # "max_depth": [None, 3, 5, 10],
                        # "min_samples_split": [2, 5, 10],
                        # "min_samples_leaf": [1, 5, 10],
                        # "random_state": [42]
                    },
                    "K-Neighbors Regressor": {
                        "n_neighbors": [3, 5, 10],
                        # "weights": ["uniform", "distance"],
                        # "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                        # "leaf_size": [10, 30, 50]
                    }
                }
            # eval_model function is present inside utils
            models_report = evaluate_model(X_train, y_train, X_test, y_test, models, params)


            # to get the best model and its score
            best_model_info = max(models_report.items(), key=lambda item: item[1]['test_score'])
            best_model_name = best_model_info[0]
            best_model_score = best_model_info[1]['test_score']
            best_model = models[best_model_name]
            best_model_params = best_model_info[1]['best_params']
            logging.info(f"Best model:{best_model_name}, score: {best_model_score}, Params: {best_model_params}")

            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info("Best model found and saving.....")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # prediction=best_model.predict(X_test)
            # score=r2_score(y_test,prediction)
            # return score
            return (best_model_score,best_model_name,best_model_params)


        
        except Exception as e:
            raise CustomException(e,sys)
