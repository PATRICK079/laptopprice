import os
import sys
import numpy as np
import pandas as pd

"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN = "final_price"
PIPELINE_NAME: str = "LaptopPrice"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "Laptop.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

#SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

#SAVED_MODEL_DIR =os.path.join("saved_models")
#MODEL_FILE_NAME = "laptop_model.pkl"




"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "LaptopPriceData" 
DATA_INGESTION_DATABASE_NAME: str = "Patrick079"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2
