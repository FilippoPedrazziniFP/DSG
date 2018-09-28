from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from dsg.loader import Util, DataLoader
import pandas as pd



class Preprocessor():

    def __init__(self):
        super(Preprocessor, self).__init__()

    def fit_transform(self):
        raise NotImplementedError

    def fit(self, data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError


class SeqPreprocessor():
    def fit(self):
        return

    def fit_transform(self, train=True):
        if train:
            try:
                print("Loading pickle data...")
                df_transformed = DataLoader.load_from_pickle("./df_transformed_train.pkl")
                df_train = pd.read_csv(Util.TRAIN_SESSION)
                print("loading completed")
            except:
                print("loading failed, constructing data...")
                df_train_tracking = pd.read_csv(Util.TRAIN_TRACKING)
                df_transformed = transform_train_tracking(df_train_tracking)
                df_transformed.to_pickle("./df_transformed_train.pkl")
                print("data constructed")
            x, y = get_dataset(df_transformed, df_train)
            return x, y

        else:
            try:
                print("Loading pickle data...")
                df_transformed = DataLoader.load_from_pickle("./df_transformed_test.pkl")
                print("loading completed")
            except:
                print("loading failed, constructing data...")
                df_test_tracking = pd.read_csv(Util.TEST_TRACKING)
                df_transformed = transform_train_tracking(df_test_tracking)
                df_transformed.to_pickle("./df_transformed_test.pkl")
                print("data constructed")
            x = df_transformed['list'].tolist()
            return x

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

    # return 2 hot encoded vectors
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
    df_grouped['page_vec'] = df_grouped["type"].apply(lambda x: get_type_feature(process_string(x))[0])
    df_grouped['event_vec'] = df_grouped["type"].apply(lambda x: get_type_feature(process_string(x))[1])
    df_grouped.to_pickle("./df_encoded.pkl")

    return df_grouped


def columns_df_to_list(df):
    """
    Returns a dataframe sid, [list of actions features]
    """
    df["list"] = df.apply(lambda x: list(x[:]), axis=1)
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

