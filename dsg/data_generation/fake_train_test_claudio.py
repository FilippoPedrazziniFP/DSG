import pandas as pd
import numpy as np
import hashlib
from sklearn.utils import shuffle

class FakeDataGeneratorClaudio(object):
    def __init__(self):
        super(FakeDataGeneratorClaudio, self).__init__()

    def hash(self, row):
        return hashlib.md5((str(row["CustomerIdx"]) + str(row["IsinIdx"]) + row["BuySell"]).encode()).hexdigest()

    def generate_train_dataset(self, trade_df, from_day=20171008, till_day=20180405, out_rows=None, imb_perc=None,
                               remove_holding=True, remove_price=True, remove_not_eur=True, remove_trade_status=True):

        trade_df = trade_df.rename(index=str, columns={"TradeDateKey": "DateKey"})

        trade_df = self.remove_columns(trade_df, remove_holding, remove_not_eur, remove_price, remove_trade_status)



        if from_day is not None:
            trade_df = trade_df[trade_df["DateKey"] >= from_day]
        if till_day is not None:
            trade_df = trade_df[trade_df["DateKey"] <= till_day]

        negative_samples_distribution = trade_df
        positive_samples = trade_df

        rows_to_add = self.get_rows_to_add(imb_perc, out_rows, positive_samples)

        if rows_to_add <= 0:
            return positive_samples


        tot_trades = trade_df.shape[0]
        trades_per_day = trade_df.groupby(["DateKey"]).size().reset_index(name='counts')
        count = trades_per_day["counts"].as_matrix().flatten()
        days = trades_per_day["DateKey"].as_matrix().flatten()
        probability_of_being_chosen = np.divide(count, tot_trades)

        negative_samples = pd.DataFrame()

        while negative_samples.shape[0] < rows_to_add:
            still_to_add = int((rows_to_add - negative_samples.shape[0]))

            if still_to_add >= negative_samples_distribution.shape[0]:
                random_interactions = negative_samples_distribution
            else:
                random_interactions = negative_samples_distribution.sample(still_to_add)

            sampled_days = np.random.choice(days, still_to_add, p=probability_of_being_chosen)
            random_interactions = random_interactions.drop(columns=["DateKey"])
            random_interactions = random_interactions.assign(DateKey=sampled_days)

            random_interactions = self.exclude(random_interactions,
                                               positive_samples,
                                               columns=["CustomerIdx", "IsinIdx", "BuySell", "DateKey"])

            negative_samples = negative_samples.append(random_interactions)

        negative_samples["CustomerInterest"] = 0

        merged = positive_samples.append(negative_samples)

        shuffled = shuffle(merged).reset_index(drop=True)

        return shuffled

    def generate_test_dataset(self, trade_df, from_day=20180414, till_day=20180420, out_rows=500000, imb_perc=None,
                              remove_holding=True, remove_price=True, remove_not_eur=True, remove_trade_status=True,
                              set_date=20180416, months_of_trades=6):

        trade_df = trade_df.rename(index=str, columns={"TradeDateKey": "DateKey"})

        trade_df = self.remove_columns(trade_df, remove_holding, remove_not_eur, remove_price, remove_trade_status)

        positive_samples = trade_df

        if from_day is not None:
            positive_samples = positive_samples[positive_samples["DateKey"] >= from_day]

        if till_day is not None:
            positive_samples = positive_samples[positive_samples["DateKey"] <= till_day]

        positive_samples["DateKey"] = set_date

        rows_to_add = self.get_rows_to_add(imb_perc, out_rows, positive_samples)

        if rows_to_add <= 0:
            return positive_samples

        # sample the negative samples from the previous months
        months_before_from_day = self.monthsbefore(from_day, months_of_trades)
        negative_samples_distribution = trade_df[trade_df["DateKey"] >= months_before_from_day]
        negative_samples_distribution = negative_samples_distribution[negative_samples_distribution["DateKey"] <= from_day]


        negative_samples = pd.DataFrame()

        while negative_samples.shape[0] < rows_to_add:
            still_to_add = int((rows_to_add - negative_samples.shape[0]))
            random_interactions = negative_samples_distribution.sample(still_to_add)

            random_interactions["DateKey"] = set_date

            random_interactions = self.exclude(random_interactions,
                                               positive_samples,
                                               columns=["CustomerIdx", "IsinIdx", "BuySell", "DateKey"])

            negative_samples = negative_samples.append(random_interactions)

        negative_samples["CustomerInterest"] = 0

        merged = positive_samples.append(negative_samples)

        shuffled = shuffle(merged).reset_index(drop=True)

        return shuffled

    def get_rows_to_add(self, imb_perc, out_rows, positive_samples):
        if out_rows is None:
            if imb_perc is None:
                rows_to_add = positive_samples.shape[0]
            else:
                rows_to_add = int((positive_samples.shape[0] / imb_perc) * (1 - imb_perc))
        else:
            rows_to_add = out_rows - positive_samples.shape[0]
        return rows_to_add


    def generate_submission_dataset(self, trade_df, from_day=20171022, till_day=None, out_rows=None, imb_perc=None,
                               remove_holding=True, remove_price=True, remove_not_eur=True, remove_trade_status=True,):

        return self.generate_train_dataset(trade_df, from_day, till_day, out_rows, imb_perc,
                                     remove_holding, remove_price, remove_not_eur, remove_trade_status)

    def generate_validation_dataset(self, trade_df, from_day=20180407, till_day=20180413, out_rows=500000,
                                    imb_perc=None,
                                    remove_holding=True, remove_price=True, remove_not_eur=True,
                                    remove_trade_status=True,
                                    set_date=20180409, months_of_trades=6):

        return self.generate_test_dataset(trade_df, from_day, till_day, out_rows, imb_perc,
                                     remove_holding, remove_price, remove_not_eur, remove_trade_status, set_date,
                                     months_of_trades)

    def remove_columns(self, trade_df, remove_holding=True, remove_not_eur=True, remove_price=True,
                       remove_trade_status=True):

        if remove_price and "Price" in trade_df.columns:
            trade_df = trade_df.drop(columns=["Price"])
        if remove_not_eur and "NotionalEUR" in trade_df.columns:
            trade_df = trade_df.drop(columns=["NotionalEUR"])
        if remove_holding and "TradeStatus" in trade_df.columns:
            trade_df = trade_df[trade_df["TradeStatus"] != "Holding"]
        if remove_trade_status and "TradeStatus" in trade_df.columns:
            trade_df = trade_df.drop(columns=["TradeStatus"])

        return trade_df

    def exclude(self, df1, df2, columns=None):
        if columns is None:
            return df1[~df1.isin(df2).all(1)]
        else:
            return df1[~df1[columns].isin(df2[columns]).all(1)]

    def monthsbefore(self, from_day, months_to_remove):
        year = int(str(from_day)[:4])
        month = int(str(from_day)[4:6])
        day = int(str(from_day)[6:])

        min_years = int(months_to_remove / 12)
        year -= min_years
        months_to_remove = months_to_remove % 12
        if month <= months_to_remove:
            year -= 1
            month = 12 - months_to_remove + month
        else:
            month -= months_to_remove

        return int(str(year) + "{0:02d}".format(month) + "{0:02d}".format(day))