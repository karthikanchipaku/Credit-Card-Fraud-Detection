import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from src.logger.logger import logging
from src.exception.exception import CustomException
from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'models', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a ColumnTransformer to scale Time and Amount using RobustScaler.
        The PCA features (V1-V28) are bypassed and remain untouched.
        """
        try:
            logging.info("Building the data transformation pipeline.")
            
            # Features that need scaling
            features_to_scale = ['Time', 'Amount']
            
            # RobustScaler is less prone to outliers than StandardScaler
            scale_pipeline = Pipeline(steps=[
                ("scaler", RobustScaler())
            ])

            # remainder='passthrough' ensures V1-V28 are kept as is
            preprocessor = ColumnTransformer(
                transformers=[
                    ("scale_pipeline", scale_pipeline, features_to_scale)
                ],
                remainder='passthrough'
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies the transformation pipeline to the train and test sets and saves the object.
        """
        try:
            logging.info("Reading train and test data.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "Class"
            
            # Splitting independent and dependent features
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            logging.info("Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object to train and test data.")
            X_train_scaled = preprocessing_obj.fit_transform(X_train)
            X_test_scaled = preprocessing_obj.transform(X_test)

            # Re-combine the scaled features with the target variable as a numpy array
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            logging.info("Saving preprocessing object to artifacts directory.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data Transformation completed successfully.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)