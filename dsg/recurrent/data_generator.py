import pandas as pd
import matplotlib
import time
from itertools import islice
import pickle
import progressbar
import numpy as np

TRADE_DATA = "../../data/Trade.csv"
CHALLENGE_DATA = "../../data/Challenge_20180423.csv"
WEEK_LEN = 5
MONTH_LEN = 6
WEEK_MONTH = 4
FEATURES_MONTH = 2
NUM_WEEKS = 4

class DataGenerator(object):
    def __init__(self):
        super(DataGenerator, self).__init__()
        self.customer_dictionary = pickle.load(open("cust_dict.pkl", "rb"))
        self.bond_dictionary = pickle.load(open("bond_dict.pkl", "rb"))
        self.cus_bond_dictionary = pickle.load(open("cus_bond_dict.pkl", "rb"))
        self.debug_dictionaries()

    def take(self, n, iterable):
        return list(islice(iterable, n))

    def print_dictionary(self, dictionary=None):
        n_items = self.take(100, dictionary.items())
        for key, val in n_items:
            print(key, val)
        return

    def debug_dictionaries(self):
        # self.print_dictionary(self.customer_dictionary)
        feat = self.customer_dictionary[(0, 1)]
        # print(feat)
        # exit()
        return

    def generate_customer_features(self, date, customer_id):
        try:
            customer_features = self.customer_dictionary[(date, customer_id)]
            customer_features = np.asarray(customer_features)
        except KeyError:
            customer_features = np.array([0.0, 0.0, 0.0])

        return customer_features

    def generate_bond_features(self, date, bond_id):
        try:
            bond_features = self.bond_dictionary[(date, bond_id)]
            bond_features = np.asarray(bond_features)
        except KeyError:
            bond_features = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return bond_features

    def generate_customer_bond_features(self, date, customer_id, bond_id):
        try:
            cus_bond_features = self.cus_bond_dictionary[(date, customer_id, bond_id)]
            cus_bond_features = np.asarray(cus_bond_features)
        except KeyError:
            cus_bond_features = np.array([0.0, 0.0, 0.0])
        return cus_bond_features

    def generate_sequence(self, date, sample):
        """
            generate a single sequence of shape
            (8, feeatures)
        """
        customer_id = sample[0]
        bond_id = sample[1]
        sequence = []
        for i in range(date - FEATURES_MONTH*WEEK_LEN*WEEK_MONTH, date, WEEK_LEN):
            sample_seq = []
            cust_feature = self.generate_customer_features(i, customer_id)
            sample_seq.extend(cust_feature)
            bond_feature = self.generate_bond_features(i, bond_id)
            sample_seq.extend(bond_feature)
            cust_bond_feature = self.generate_customer_bond_features(i, customer_id, bond_id)
            sample_seq.extend(cust_bond_feature)
            sample_seq.extend([sample[2].astype("float32")])
            sample_seq.extend([sample[3].astype("float32")])
            sample_seq = np.asarray(sample_seq)
            sequence.append(sample_seq)
            
        sequence = np.asarray(sequence)
        return sequence
        
    def generate_samples(self, date, df):
        """
            generate a numpy array with shape:
            ((samples, 8, features), (labels))
        """
        features = []
        labels = []
        for sample in df:
            feature = self.generate_sequence(date, sample)
            features.append(feature)
            labels.append(sample[-1])
       
        features = np.asarray(features)
        labels = np.asarray(labels)
        return features, labels

    def generate_labels_classification(self, df, from_date, from_date_label, to_date=None):
        """
            The method creates a dataframe for testing purposes similar to the one 
            of the competition. It uses the last 2 years/6 months interactions as negative labels
            x 2 based on sell and buy.

            @args
                df : DataFrame -> entire Trade Table.
                from_date : int -> corresponding tot the date in which 
                    we start the week of test.
                from_date_label : int -> representing the starting date from
                    which we start to collect the negative samples.
                to_date : int -> date representing the end of the week.
            @return
                test_set : DataFrame -> with 1 week of positive and negative 
                    samples from the week considered and the previous 6 months.
        """

        if to_date is None:
            positive_samples = df[df["TradeDateKey"] >= from_date]
            positive_samples_neg = df[df["TradeDateKey"] >= from_date]
        else:
            positive_samples = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]
            positive_samples_neg = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]

        positive_samples_neg["BuySell_Buy"] = positive_samples["BuySell_Sell"]
        positive_samples_neg["BuySell_Sell"] = positive_samples["BuySell_Buy"]
        positive_samples_neg["CustomerInterest"] = positive_samples_neg["CustomerInterest"]\
            .apply(lambda x: 0 if x > 0 else x)
        
        # Negative Samples
        negative_samples = df[(df["TradeDateKey"] >= from_date_label) & (df["TradeDateKey"] < from_date)]
        negative_samples_neg = df[(df["TradeDateKey"] >= from_date_label) & (df["TradeDateKey"] < from_date)]

        # Opposite Positive
        positive_samples_neg["BuySell_Buy"] = negative_samples["BuySell_Sell"]
        positive_samples_neg["BuySell_Sell"] = negative_samples["BuySell_Buy"]

        # Double Negative Samples
        negative_samples_neg = df[(df["TradeDateKey"] >= from_date_label) & (df["TradeDateKey"] < from_date)]
        negative_samples_neg["BuySell_Buy"] = negative_samples["BuySell_Sell"]
        negative_samples_neg["BuySell_Sell"] = negative_samples["BuySell_Buy"]

        # Put to zero all the negative
        negative_samples["CustomerInterest"] = negative_samples["CustomerInterest"]\
            .apply(lambda x: 0 if x > 0 else x)
        negative_samples_neg["CustomerInterest"] = negative_samples_neg["CustomerInterest"]\
            .apply(lambda x: 0 if x > 0 else x)

        # Concatanate Negative and Positive Samples
        labels = pd.concat([positive_samples, positive_samples_neg, negative_samples_neg, negative_samples])
        labels = labels.drop(["TradeDateKey"], axis=1)

        # Unique Values
        labels = labels.groupby(['CustomerIdx', 'IsinIdx', "BuySell_Sell", "BuySell_Buy"]).sum()
        labels = labels.reset_index(level=['CustomerIdx', 'IsinIdx', "BuySell_Sell", "BuySell_Buy"])
        labels["CustomerInterest"] = labels["CustomerInterest"].apply(lambda x: 1 if x > 1 else x)

        # Reorder the columns
        labels = labels[['CustomerIdx', 'IsinIdx', "BuySell_Buy", "BuySell_Sell", 'CustomerInterest']]

        return labels

    def create_weekly_data(self, df, max_date):
        data = []
        for i in progressbar.progressbar(range(max_date - WEEK_LEN*NUM_WEEKS, max_date, WEEK_LEN)):
            week = self.generate_labels_classification(
                df=df,
                from_date=i,
                to_date=i+WEEK_LEN,
                from_date_label=i-MONTH_LEN*WEEK_LEN*WEEK_MONTH
                )
            data.append((i, week))
        with open('data.pkl', 'wb') as f:
            pickle.dump(data, f)
        return data

    def train_test_validation_split(self, data):
        train = data[0:-2]
        test = data[-1]
        val = data[-2]
        return train, test, val, data

    def generate_set(self, data, kind="train"):
        """
            The method generates a training data starting 
            from a list of dataframes which represent the
            weekly labels.
            
            train_samples : numpy array (samples, 8, features)
        """
        features = []
        labels = []

        for i, df in progressbar.progressbar(data):
            X, y = self.generate_samples(i, df.values)
            features.append(X)
            labels.append(y)

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        with open('features_' + kind + '.pkl', 'wb') as f:
            pickle.dump(features, f)
        with open('labels_' + kind + '.pkl', 'wb') as f:
            pickle.dump(labels, f)
        return features, labels

    def generate_entire_set(self, data):
        """
            The method generates a training data starting 
            from a list of dataframes which represent the
            weekly labels.
            
            train_samples : numpy array (samples, 8, features)
        """
        features = []
        labels = []

        for i, df in progressbar.progressbar(data):
            X, y = self.generate_samples(i, df.values)
            features.append(X)
            labels.append(y)
            with open('entire_data/features_' + str(i) + '_entire.pkl', 'wb') as f:
                pickle.dump(X, f)
            with open('entire_data/labels_' + str(i) + '_entire.pkl', 'wb') as f:
                pickle.dump(y, f)
        
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels

    def generate_single_set(self, data, kind="test"):        
        X, y = self.generate_samples(data[0], data[1].values)
        
        with open('features_' + kind + '.pkl', 'wb') as f:
            pickle.dump(X, f)
        with open('labels_' + kind + '.pkl', 'wb') as f:
            pickle.dump(y, f)
        return X, y

    def challenge_transform(self, df):
        df = df.drop(["PredictionIdx", "DateKey"], axis=1)
        df = pd.get_dummies(df, columns=["BuySell"])
        df = df[['CustomerIdx', 'IsinIdx', "BuySell_Buy", 
            "BuySell_Sell", 'CustomerInterest']]
        return df

def main():
        
    # Load the Data
    df = pd.read_csv(TRADE_DATA)

    # Reorder Trade by Date
    df = df.sort_values("TradeDateKey", ascending=True)

    # Load the dictionary {date : value}
    dictionary_date = {}
    i = 0
    for row in df["TradeDateKey"].unique():
        dictionary_date[row]=i
        i = i+1

    # Delete Holding Values
    df = df[df["TradeStatus"] != "Holding"]

    # Drop useless columns
    df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)

    # Get Dummies for BuySell feature
    df = pd.get_dummies(df, columns=["BuySell"])

    # Converting Dates
    max_date = dictionary_date[20180422]
    print("MAX_DATE: ", max_date)
    max_date = max_date + 2
    print("CLIPPED MAX DATE: ", max_date)

    # Transform DateKey into a column from 0 to 1000
    df["TradeDateKey"] = df["TradeDateKey"].apply(lambda x: dictionary_date[x])

    print("CREATING WEEKLY LABELS...")
    # Create DataGenerator object
    generator = DataGenerator()
    try:
        data = pickle.load(open('data.pkl', 'rb'))
    except:
        data = generator.create_weekly_data(df, max_date)
    print("NUMBER OF WEEKS GENERATED: ", len(data))

    train, test, val, data = generator.train_test_validation_split(data)
    print("SPLITTED TRAIN TEST VAL")

    print("Some Statistics About The Weekly interactions.")
    for i, df in train:
        print(df.shape)

    print("GENERATING TRAIN SET...")
    t = time.clock()
    try:
        X_train = pickle.load(open('features_train.pkl', 'rb'))
        y_train = pickle.load(open('labels_train.pkl', 'rb'))
    except:
        X_train, y_train = generator.generate_set(train)
        print(X_train.shape)
        print(y_train.shape)
    print("GENERATED TRAIN SET in: ", time.clock() - t)

    print("GENERATING TEST SET...")
    t = time.clock()
    try:
        X_test = pickle.load(open('features_test.pkl', 'rb'))
        y_test = pickle.load(open('labels_test.pkl', 'rb'))
    except:
        X_test, y_test = generator.generate_single_set(test, kind="test")
        print(X_test.shape)
        print(y_test.shape)
    print("GENERATED TEST SET in: ", time.clock() - t)

    print("GENERATING VAL SET...")
    t = time.clock()
    try:
        X_val = pickle.load(open('features_val.pkl', 'rb'))
        y_val = pickle.load(open('labels_val.pkl', 'rb'))
    except:
        X_val, y_val = generator.generate_single_set(val, kind="val")
        print(X_val.shape)
        print(y_val.shape)
    print("GENERATED VAL SET in: ", time.clock() - t)

    print("GENERATING ENTIRE SET...")
    t = time.clock()
    try: 
        X = []
        y = []
        for i, df in data:
            features = pickle.load(open('./entire_data/features_' + str(i) + '_entire.pkl', 'rb'))
            labels = pickle.load(open('./entire_data/labels_' + str(i) + '_entire.pkl', 'rb'))
            X.append(features)
            y.append(labels)
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
    except:
        X, y = generator.generate_entire_set(data)
        print(X.shape)
        print(y.shape)
    print("GENERATED ENTIRE SET in: ", time.clock() - t)

    # Load and Transform Challenge
    print("GENERATING CHALLENGE SET...")
    t = time.clock()
    try:
        X_challenge = pickle.load(open('features_challenge.pkl', 'rb'))
    except:
        challenge = pd.read_csv(CHALLENGE_DATA)
        challenge = generator.challenge_transform(challenge)
        X_challenge = generator.generate_single_set((max_date, challenge), kind="challenge")
    print("GENERATED CHALLENGE SET in: ", time.clock() - t)
    
    return

main()

