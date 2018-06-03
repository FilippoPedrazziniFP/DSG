import pandas as pd
import numpy as np
import hashlib
from sklearn.utils import shuffle


class DataGenerator(object):
    def __init__(self):
        super(DataGenerator, self).__init__()

    def hash(self, row):
        return hashlib.md5((str(row["CustomerIdx"]) + str(row["IsinIdx"]) + row["BuySell"]).encode()).hexdigest()

    def generate_test_dataset(self, df, day, imb_perc = 0.015, consider_holding_1 = False):

        trade = df

        # Select the transactions to use in the test set
        trades_on_eval_day = trade[trade["TradeDateKey"] == day]

        trades_on_eval_day = trades_on_eval_day[trades_on_eval_day["CustomerInterest"] == 1]

        real_interactions = trades_on_eval_day[["CustomerIdx", "IsinIdx", "BuySell", "CustomerInterest"]]

        if consider_holding_1:
            real_interactions["CustomerInterest"] = 1

        # Calculate how many rows to add
        positive_samples = real_interactions[real_interactions["CustomerInterest"] == 1].shape[0]
        out_rows = int(positive_samples / imb_perc)
        rows_to_add = out_rows - real_interactions.shape[0]


        # get how many trades have been done in the considered period
        tot_trades = trade.shape[0]

        # Generate the CustomerIdx column for the added interactions
        trades_per_cust = trade.groupby(["CustomerIdx"]).size().reset_index(name='counts')
        count = trades_per_cust["counts"].as_matrix().flatten()
        cust_idx = trades_per_cust["CustomerIdx"].as_matrix().flatten()
        probability_of_being_chosen = np.divide(count, tot_trades)

        fake_cust = np.random.choice(cust_idx, rows_to_add, p=probability_of_being_chosen)

        # Generate the Isin column for the added interactions
        trades_per_bond = trade.groupby(["IsinIdx"]).size().reset_index(name='counts')
        count = trades_per_bond["counts"].as_matrix().flatten()
        bond_idx = trades_per_bond["IsinIdx"].as_matrix().flatten()
        probability = np.divide(count, tot_trades)

        fake_bond = np.random.choice(bond_idx, rows_to_add, p=probability)

        # Generate the BuySell column for the added interactions
        fake_buy_sell = np.random.choice(["Buy", "Sell"], rows_to_add)

        # Zip them in one single data frame
        fake_interactions = pd.DataFrame(data=np.matrix([fake_cust, fake_bond, fake_buy_sell]).T,
                                         columns=["CustomerIdx", "IsinIdx", "BuySell"],
                                         index=np.arange(fake_cust.shape[0]))
        fake_interactions["CustomerInterest"] = 0



        # Merge ral and fake in one dataset
        test_set = real_interactions.append(fake_interactions)


        test_set["DateKey"] = day

        test_set["PredictionIdx"] = test_set.apply(self.hash, axis=1)

        # Eliminate duplicates, some duplicates exists because the same customer
        # buys/sells the same bond on the same day
        test_set = test_set[~test_set.duplicated(["PredictionIdx"])]

        # Sort by hashvalue
        test_set = test_set.sort_values("PredictionIdx")

        return test_set


    def subsample_negatives(self, df, num_samples):
        pos = df[df["CustomerInterest"] == 1]
        neg = df[df["CustomerInterest"] == 0]
        neg = neg.sample(num_samples)

        return pos.append(neg)

    def generate_train_dataset(self, df, till_date, consider_holding_1 = False, remove_holding=False):
        trade = df

        # Select the transactions to use in the test set
        trades_to_use = trade[trade["TradeDateKey"] < till_date]

        real_interactions = trades_to_use[["CustomerIdx", "IsinIdx", "BuySell", "CustomerInterest", "TradeDateKey"]]

        if consider_holding_1:
            real_interactions["CustomerInterest"] = 1
            rows_to_add = trades_to_use.shape[0]

        else:
            positive_samples = real_interactions[real_interactions["CustomerInterest"] == 1].shape[0]
            negative_samples = real_interactions.shape[0] - positive_samples

            if not remove_holding:
                if negative_samples > positive_samples:
                    real_interactions = self.subsample_negatives(real_interactions, positive_samples)
                    rows_to_add = 0
                else:
                    rows_to_add = positive_samples - negative_samples
            else:
                rows_to_add = positive_samples
                real_interactions = real_interactions[real_interactions["CustomerInterest"] == 1]


        if rows_to_add > 0:
            # get how many trades have been done in the considered period
            tot_trades = trades_to_use.shape[0]

            # Generate the CustomerIdx column for the added interactions
            trades_per_cust = trades_to_use.groupby(["CustomerIdx"]).size().reset_index(name='counts')
            count = trades_per_cust["counts"].as_matrix().flatten()
            cust_idx = trades_per_cust["CustomerIdx"].as_matrix().flatten()
            probability_of_being_chosen = np.divide(count, tot_trades)

            fake_cust = np.random.choice(cust_idx, rows_to_add, p=probability_of_being_chosen)

            # Generate the Isin column for the added interactions
            trades_per_bond = trades_to_use.groupby(["IsinIdx"]).size().reset_index(name='counts')
            count = trades_per_bond["counts"].as_matrix().flatten()
            bond_idx = trades_per_bond["IsinIdx"].as_matrix().flatten()
            probability = np.divide(count, tot_trades)

            fake_bond = np.random.choice(bond_idx, rows_to_add, p=probability)

            # Generate the BuySell column for the added interactions
            fake_buy_sell = np.random.choice(["Buy", "Sell"], rows_to_add)

            # Zip them in one single data frame
            fake_interactions = pd.DataFrame(data=np.matrix([fake_cust, fake_bond, fake_buy_sell]).T,
                                             columns=["CustomerIdx", "IsinIdx", "BuySell"],
                                             index=np.arange(fake_cust.shape[0]))
            fake_interactions["CustomerInterest"] = 0


            fake_interactions["TradeDateKey"] = trades_to_use["TradeDateKey"].copy()

            # Merge real and fake in one dataset
            train_set = real_interactions.append(fake_interactions)

        else:
            train_set = real_interactions

        # Eliminate duplicates, some duplicates exists because the same customer
        # buys/sells the same bond on the same day
        train_set = train_set[~train_set.duplicated(["CustomerIdx", "IsinIdx", "BuySell", "TradeDateKey"])]

        train_set = shuffle(train_set, random_state = 0)

        return train_set
