
import os
import sys

from src.exception.exception import LaptopPriceException 
from src.logging.logger import logging

from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig

from src.utils.main_utils import LaptopModel
from src.utils.main_utils import save_object,load_object
from src.utils.main_utils import load_numpy_array_data, evaluate_models
from src.utils.main_utils import get_metric_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from catboost import CatBoostRegressor
from sklearn.ensemble import ( AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

import mlflow

import dagshub
dagshub_token = os.getenv("DAGSHUB_TOKEN")

dagshub.init(repo_owner='PATRICK079', repo_name='laptopprice', mlflow=True)


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise LaptopPriceException(e,sys)
        
           
    def track_mlflow(self,best_model,classificationmetric):
       with mlflow.start_run():
              root_mean_squared_error =classificationmetric.root_mean_squared_error
              r2_score=classificationmetric.r2_score


              mlflow.log_metric("root_mean_squared_error",root_mean_squared_error)
              mlflow.log_metric("r2_score",  r2_score)
              mlflow.sklearn.log_model(best_model,"model")
        

    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "ElasticNet Regression": ElasticNet(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(verbose=1),
            "Gradient Boosting": GradientBoostingRegressor(verbose=1),
            "AdaBoost": AdaBoostRegressor(),
            "XGBoost": XGBRegressor(verbose=1),
            "CatBoost": CatBoostRegressor(verbose=1),
            "K-Nearest Neighbors": KNeighborsRegressor()
        }

        params = {
            "Linear Regression": {},  
            "Ridge Regression": {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            "Lasso Regression": {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            },
            "ElasticNet Regression": {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.2, 0.5, 0.7, 1.0]
            },
            "Decision Tree": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                'max_depth': [3, 5, 10, 20]
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 64, 128, 256],
                'max_depth': [5, 10, 20],
                'criterion': ['squared_error', 'absolute_error']
            },
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'n_estimators': [50, 100, 150, 200],
                'subsample': [0.6, 0.7, 0.8, 0.9]
            },
            "AdaBoost": {
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'n_estimators': [50, 100, 150]
            },
            "XGBoost": {
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10]
            },
            "CatBoost": {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 6, 10]
            },
            "K-Nearest Neighbors": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
    }

        # Training and evaluation logic goes here if needed

        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test, 
                                          models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        logging.info(f"The best model is: {best_model_name}")
        print(f"The best model is: {best_model_name}")
        
        best_model = models[best_model_name]

        y_train_pred=best_model.predict(X_train)

        train_metric=get_metric_score(y_true = y_train,y_pred=y_train_pred)
        
        ## Track the experiements with mlflow

        self.track_mlflow(best_model,train_metric)


        y_test_pred=best_model.predict(x_test)

        test_metric=get_metric_score(y_true=y_test,y_pred=y_test_pred)

       
        self.track_mlflow(best_model,test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Laptop_Model=LaptopModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=LaptopModel)

        save_object("final_model/laptop_model.pkl",best_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=train_metric,
                             test_metric_artifact=test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise LaptopPriceException(e,sys)


                        