import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import LaptopPriceException 
from src.logging.logger import logging
from src.utils.main_utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise LaptopPriceException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Reads CSV data from a file path."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise LaptopPriceException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """Creates and returns a preprocessor object that performs one-hot encoding and MinMax scaling."""
        try:
            logging.info("Creating data transformation pipeline")

            # Define column types based on your dataset
            categorical_features = ["status", "storage_type", "touch"]
            numerical_features = [ "storage", "screen"]

            # Transformation pipeline
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")
            numerical_transformer = MinMaxScaler()

            # Column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, numerical_features),
                    ("cat", categorical_transformer, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise LaptopPriceException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            
            # Read training and testing datasets
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Splitting input and target features
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Get the preprocessing pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit and transform the training and testing data
            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)

            # Concatenate transformed features with the target variable
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save transformed data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            
            # Save preprocessor object
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_object("artifacts/preprocessors/data_preprocessor.pkl", preprocessor)

            # Prepare artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise LaptopPriceException(e, sys)
