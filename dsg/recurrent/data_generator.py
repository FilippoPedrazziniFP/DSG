import pandas as pd
import matplotlib
import time
import pickle
import progressbar
import numpy as np

MAX_DATE = 632
STARTING_DATE = 0
ENDING_DATE = 100
SEQ_LABEL = 7
SEQ_FEATURES = 56
TRADE_DATA = "../../data/Trade.csv"

def generate_labels(positive_samples, negative_samples):
    
    # Concatanate Negative and Positive Samples
    test_set = pd.concat([positive_samples, negative_samples])
    test_set = test_set.drop(["TradeDateKey"], axis=1)

    # Unique Values
    test_set = test_set.groupby(['CustomerIdx', 'IsinIdx']).sum()
    test_set = test_set.reset_index(level=['CustomerIdx', 'IsinIdx'])
    test_set["CustomerInterest"] = test_set["CustomerInterest"].apply(lambda x: 1 if x > 1 else x)
    
    return test_set

def generate_customer_features(date, customer_id):
    # Take the customer history
    df = df_tra[df_tra["CustomerIdx"] == customer_id]
    # Take One Week
    df = df[(df["TradeDateKey"] >= date) & (df["TradeDateKey"] < date + seq_label)]
    # Number of Interactions during the period
    num_int = len(df.index)
    return num_int

def generate_bond_features(date, bond_id):
    # Take the customer history
    df = df_tra[df_tra["IsinIdx"] == bond_id]
    # Take One Week
    df = df[(df["TradeDateKey"] >= date) & (df["TradeDateKey"] < date + seq_label)]
    # Number of Interactions during the period
    num_int = len(df.index)
    return num_int

def generate_customer_bond_features(date, customer_id, bond_id):
    # Take the customer-bond history
    df = df_tra[(df_tra["CustomerIdx"] == customer_id) & (df_tra["IsinIdx"] == bond_id)]
    # Take One Week
    df = df[(df["TradeDateKey"] >= date) & (df["TradeDateKey"] < date + seq_label)]
    # Number of Interactions during the period
    num_int = len(df.index)
    return num_int

def generate_sequence(date, sample):
    """
        generate a single sequence of shape
        (8, feeatures)
    """
    customer_id = sample[0]
    bond_id = sample[1]
    sequence = []
    for i in range(date-seq_features, date, seq_label):
        cust_feature = generate_customer_features(i, customer_id)
        bond_feature = generate_bond_features(i, bond_id)
        cust_bond_feature = generate_customer_bond_features(i, customer_id, bond_id)
        sample = np.array([cust_feature, bond_feature, cust_bond_feature])
        sequence.append(sample)
        
    sequence = np.asarray(sequence)
    return sequence
    
def generate_set(date, df):
    """
        generate a numpy array with shape:
        (samples, 8, features)
    """
    features = []
    labels = []
    for sample in df:
        feature = generate_sequence(date, sample)
        features.append(feature)
        labels.append(sample[-1])
   
    features = np.asarray(features)
    labels = np.asarray(labels)
    return features, labels

def generate_negative_samples(df):
    # Negative Samples
    negative_samples = df[df["TradeDateKey"] >= 0]
    negative_samples = negative_samples.groupby(["CustomerIdx", "IsinIdx"]).count()
    negative_samples = negative_samples[negative_samples["CustomerInterest"] <= 1]
    negative_samples["CustomerInterest"] = negative_samples["CustomerInterest"].apply(lambda x: 0 if x > 0 else x)
    negative_samples = negative_samples.reset_index(level=['CustomerIdx', 'IsinIdx'])
    return negative_samples

def create_weekly_data(df):
    data = []
    for i in progressbar.progressbar(range(STARTING_DATE + SEQ_FEATURES, ENDING_DATE - SEQ_LABEL)):
        positive_samples = df[(df["TradeDateKey"] >= i) & (df["TradeDateKey"] < i + SEQ_LABEL)]
        labels = generate_labels(positive_samples, negative_samples)
        data.append((i, labels))

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    return data

def train_test_validation_split(data):
    train = data[0:5]
    val = data[-2]
    test = data[-1]
    return train, test, val

def generate_set(data):
    train_samples = []
    for i, df in progressbar.progressbar(train):
        X, y = generate_set(i, df.values)
        print(X.shape)
        print(y.shape)
        train_samples.append((X, y))
    return train_samples

def main():
    
    # Load Data    
    df = pd.read_csv(TRADE_DATA)

    # Use just part of the data as training
    df = df[df["TradeDateKey"] >= TRAINING_DATE]

    # Delete Holding Values
    df = df[df["TradeStatus"] != "Holding"]

    # Drop useless columns
    df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)

    # Get Dummies for BuySell feature
    df = pd.get_dummies(df, columns=["BuySell"])

    # Reorder Trade by Date
    df = df.sort_values("TradeDateKey", ascending=True)

    print("CREATING DICTIONARY")
    print(df.head(5))
    print(df.describe())
    
    # Load the dictionary {date : value}
    dictionary_date = pickle.load(open( "date_dict.p", "rb" ))

    # Transform DateKey into a column from 0 to 1000
    df["TradeDateKey"] = df["TradeDateKey"].apply(lambda x: dictionary_date[x])

    print("FIRST PREPROCESSING DONE: ")
    print(df.head(5))
    print(df.describe())

    negative_samples = generate_negative_samples(df)
    print("GENERATED GENERAL NEGATIVE SAMPLES")

    print("CREATING WEEKLY LABELS")
    data = create_weekly_data(df)
    print("NUMBER OF WEEKS GENERATED: ", len(data))

    train, test, val = train_test_validation_split(data)
    print("SPLITTED TRAIN TEST VAL")

    print("Some Statistics About The Weekly interactions.")
    for i, df in train:
        print(df.shape)

    print("GENERATING TRAIN SET...")
    train_samples = generate_set(train)
    print("GENERATED TRAIN SET")
    
    return

main()

