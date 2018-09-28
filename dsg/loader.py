import pandas as pd
import pickle

class Util():
    TRAIN_SESSION = "./data/train_session.csv"
    TRAIN_TRACKING = "./data/train_tracking.csv"
    RANDOM_SUBMISSION = "./data/random_submission.csv"
    TEST_TRACKING = "./data/test_tracking.csv"
    PRODUCT_CATEGORY = "./data/productid_category.csv"

    FEATURES = "./data/Xtrain.p"
    LABELS = "./data/Ytrain.p"

    AFTER_PREPROCESSING = "./data/after_preprocessing.pkl"

    @staticmethod
    def generate_submission_file(model, preprocessor):
        X_test = preprocessor.transform()
        y_pred = model.transform(X_test)

        # generate the submission file
        # substitute the column
        return

class DataLoader(object):    
    
    @staticmethod
    def load_created_data():
        with open(Util.FEATURES, "rb") as f:
            X_train = pickle.load(f)
        with open(Util.LABELS, "rb") as f:
            y_train = pickle.load(f)
        print("DATA LOADED")
        return [X_train, y_train]
    
    @staticmethod
    def save_into_pickle(file_path, file):
        with open(file_path, 'wb') as f:
            pickle.dump(file, f)
        return
    
    @staticmethod
    def load_from_pickle(file_path):
        with open(file_path, "rb") as f:
            file = pickle.load(f)
        return file

    
