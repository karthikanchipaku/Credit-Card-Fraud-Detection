import os
import sys
import pickle
from src.logger.logger import logging
from src.exception.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a python object (like a scikit-learn pipeline) to a specified file path.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a saved python object from a specified file path.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)