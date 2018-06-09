import pandas as pd
import numpy as np
import hashlib
from sklearn.utils import shuffle


class DataGeneratorClaudio(object):
    def __init__(self):
        super(DataGeneratorClaudio, self).__init__()

    def hash(self, row):
        return hashlib.md5((str(row["CustomerIdx"]) + str(row["IsinIdx"]) + row["BuySell"]).encode()).hexdigest()

    @DeprecationWarning
    def generate_train_dataset(self, trade_df, from_day=20171008, till_day=20180405, out_rows=None, imb_perc=None,
                               remove_holding=True, remove_price=True, remove_not_eur=True, remove_trade_status=True):

        trade_df = trade_df.rename(index=str, columns={"TradeDateKey": "DateKey"})
        trade_df = trade_df[trade_df["TradeStatus"] != "Holding"]
        trade_df = trade_df[["CustomerIdx","IsinIdx","BuySell", "DateKey"]]
        trade_df = self.remove_columns(trade_df, remove_holding, remove_not_eur, remove_price, remove_trade_status)



        if from_day is not None:
            trade_df = trade_df[trade_df["DateKey"] >= from_day]
        if till_day is not None:
            trade_df = trade_df[trade_df["DateKey"] <= till_day]

        negative_samples_distribution = trade_df
        positive_samples = trade_df.drop_duplicates(["CustomerIdx","IsinIdx","BuySell", "DateKey"])

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
            still_to_add = int((rows_to_add - negative_samples.shape[0]) * 1.3)

            if still_to_add >= negative_samples_distribution.shape[0]:
                random_interactions = negative_samples_distribution
            else:
                random_interactions = negative_samples_distribution.sample(still_to_add)

            sampled_days = np.random.choice(days, random_interactions.shape[0], p=probability_of_being_chosen)
            random_interactions = random_interactions.drop(columns=["DateKey"])
            random_interactions = random_interactions.assign(DateKey=sampled_days)

            random_interactions = self.exclude(random_interactions,
                                               positive_samples,
                                               columns=["CustomerIdx", "IsinIdx", "BuySell", "DateKey"])

            negative_samples = negative_samples.append(random_interactions)
            negative_samples = negative_samples.drop_duplicates(["CustomerIdx","IsinIdx","BuySell","DateKey"])

        negative_samples = negative_samples.sample(rows_to_add)
        negative_samples["CustomerInterest"] = 0

        merged = positive_samples.append(negative_samples)

        shuffled = shuffle(merged).reset_index(drop=True)

        return shuffled

    def generate_test_dataset(self, trade_df, from_day=20180416, till_day=None, set_date=20180417, key_date = 20171022):

        trade_df = trade_df.rename(index=str, columns={"TradeDateKey": "DateKey"})
        trade_df = trade_df[trade_df["TradeStatus"] != "Holding"]
        trade_df = trade_df[["CustomerIdx","IsinIdx","BuySell", "DateKey"]]

        positive_samples = trade_df

        if from_day is not None:
            positive_samples = positive_samples[positive_samples["DateKey"] >= from_day]

        if till_day is not None:
            positive_samples = positive_samples[positive_samples["DateKey"] <= till_day]

        positive_samples = positive_samples.drop_duplicates(["CustomerIdx","IsinIdx","BuySell"])
        positive_samples["CustomerInterest"] = 1

        negative_samples_new = trade_df[trade_df["DateKey"] >= key_date].sample(frac=0.95)

        # Here we have to have a bigger sample of old negative samples (15% instead of 8% as we said before)
        # because the drop duplicates
        # will keep only the first occurrence, and the append will put the old negatives as second occurrence
        # so quite a bit of old negatives samples will be removed.
        # 15 % is a good value because it gives an output test with more or less the same size of challenge
        negative_samples_old = trade_df[trade_df["DateKey"] < key_date].sample(frac=0.15)

        negative_samples = negative_samples_new.append(negative_samples_old).drop_duplicates(["CustomerIdx","IsinIdx","BuySell"])

        negative_samples["CustomerInterest"] = 0

        merged = positive_samples.append(negative_samples).drop_duplicates(["CustomerIdx","IsinIdx","BuySell"])

        merged["DateKey"] = set_date

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

    def generate_validation_dataset(self, trade_df, from_day=20180409, till_day=20180414, set_date=20180410, key_date = 20171022):

        return self.generate_test_dataset(trade_df, from_day, till_day, set_date, key_date)


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
