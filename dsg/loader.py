import pandas as pd

class Util():
    TRAIN_SESSION = "./data/train_session.csv"
    TRAIN_TRACKING = "./data/train_tracking.csv"
    RANDOM_SUBMISSION = "./data/random_submission.csv"
    TEST_TRACKING = "./data/test_tracking.csv"
    PRODUCT_CATEGORY = "./data/productid_category.csv"

class DataLoader(object):
    @staticmethod
    def load_train_data(file_path):
        df_train_target = pd.read_csv(Util.TRAIN_SESSION)
        df_train_features = pd.read_csv(Util.TRAIN_TRACKING)
        df_train = df_train_target.merge(df_train_features, on="sid")
        df_train = df_train.sort_values(by=['duration'])
        return df_train
    
