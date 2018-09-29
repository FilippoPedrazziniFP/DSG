import time
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from dsg.preprocessor import Preprocessor
from dsg.loader import Util, DataLoader
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

class SeqPreprocessor(Preprocessor):
    def __init__(self, train_samples=10000, 
		test_samples=1000, val_samples=1000, max_len=11):
        super(SeqPreprocessor, self).__init__()
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.val_samples = val_samples
        self.max_len = max_len

    def load_data(self, is_train=True):
        if is_train == True:            
            try:
                with open(Util.TRAIN_LSTM, "rb") as f:
                    df_train = pickle.load(f)
            except:
                print("train File not found")
                exit()
            
            X_train = df_train["features"].tolist()
            y_train = list(np.asarray(df_train["target"].tolist()).astype(int))
            return X_train, y_train
        if is_train == False:
            try:
                with open(Util.TEST_LSTM, "rb") as f:
                    df_train = pickle.load(f)
            except:
                print("test File not found")
                exit()
            X_test = df_train["features"].tolist()
            return X_test, None
        return
    
    def transform(self, pad=True):
        X, _ = self.load_data(is_train=False)
        
        if pad == True:
            X = pad_sequences(X, maxlen=self.max_len)
            X = np.reshape(X, newshape=(X.shape[0], -1))
        
        return X
    
    def fit_transform(self, pad=True):
        X, y = self.load_data(is_train=True)
        
        if pad == True:
            X = pad_sequences(X, maxlen=self.max_len)
            X = np.reshape(X, newshape=(X.shape[0], -1))

        # train, test, validation
        X_train, y_train, X_test, y_test, X_val, y_val = \
        self.train_test_validation_split(X, y)

        return X_train, y_train, X_test, y_test, X_val, y_val, X, y

def get_type_feature(type_str):
    """
	input : type field of df_train_tracking
	output : 2 one-hot-encoded vectors (page, event)

	"""
    page_type = ['PA', 'LP', 'LR', 'CAROUSEL', 'SHOW_CASE']
    event_type = ['ADD_TO_BASKET', 'PURCHASE_PRODUCT', 'PRODUCT']

    page_vec = [0] * len(page_type)
    event_vec = [0] * len(event_type)

    indeces_page = [i for i, elem in enumerate(page_type) if elem in type_str]
    indeces_event = [i for i, elem in enumerate(event_type) if elem in type_str]

    if indeces_page:
        page_vec[indeces_page[0]] = 1

    if indeces_event:
        event_vec[indeces_event[0]] = 1
    
    page_vec.extend(event_vec)
   
    return page_vec, event_vec


def process_string(s):
    s = s.replace("SEARCH", "LR")
    s = s.replace("LIST_PRODUCT", "LP")
    return s


def encode_train(df):
    """
    Dataframe grouped by id. Type feature is encoded in two vectors page_vec, event_vec
    Resulting dataframe saved in df_encoded.pkl
    """
    df_grouped = df.groupby('sid').apply(lambda x: x.sort_values(["duration"]))
    # df_grouped['page_vec'] = df_grouped["type"].apply(lambda x: get_type_feature(process_string(x))[0])
    # df_grouped['event_vec'] = df_grouped["type"].apply(lambda x: get_type_feature(process_string(x))[1])
    df_grouped['concat_vec'] = df_grouped["type"].apply(lambda x: get_type_feature(process_string(x)))
    # df_grouped.to_pickle("./df_encoded.pkl")
    return df_grouped

def columns_df_to_list(df):
    """
    Returns a dataframe sid, [list of actions features]
    """
    df["list"] = df.apply(lambda x: list(x[['sid', 'concat_vec']]), axis=1)
    listed_final = df["list"]
    listed_final = listed_final.reset_index().groupby('sid')['list'].apply(list).reset_index()
    
    return listed_final



def transform_train_tracking(df):
    df = encode_train(df)
    df = columns_df_to_list(df)

    return df


def get_dataset(transformed_df, label_df):
    merged_df = label_df.merge(transformed_df, on=['sid'], how='left')
    merged_df['label'] = merged_df["target"].apply(lambda x: 0 if x is False else 1)
    x = merged_df['list'].tolist()
    y = merged_df['label'].tolist()

    return x, y

