import pandas as pd
import numpy as np
import hashlib
from sklearn.utils import shuffle
import datetime

class DataGenerator(object):
    def __init__(self):
        super(DataGenerator, self).__init__()


    def generate_train_dataset(self, trade_df, from_day=20180000, till_day=20180409):

        trade_df = trade_df.rename(index=str, columns={"TradeDateKey": "DateKey"})
        trade_df = trade_df[trade_df["TradeStatus"] != "Holding"]
        trade_df = trade_df[["CustomerIdx", "IsinIdx", "BuySell", "DateKey"]]
        trade_df["DateKey"] = trade_df["DateKey"].apply(lambda x: self.monday_date(x))

        positive_samples = trade_df

        if from_day is not None:
            positive_samples = positive_samples[positive_samples["DateKey"] >= from_day]

        if till_day is not None:
            positive_samples = positive_samples[positive_samples["DateKey"] < till_day]


        positive_samples = positive_samples.drop_duplicates(["CustomerIdx", "IsinIdx", "BuySell","DateKey"]).sample(frac=0.95)
        positive_samples["CustomerInterest"] = 1

        key_date = self.monthsbefore(from_day,6)

        negative_samples_new = trade_df[trade_df["DateKey"] >= key_date].sample(frac=0.95)

        negative_samples_old = trade_df[trade_df["DateKey"] < key_date].sample(frac=0.10)


        negative_samples = negative_samples_new.append(negative_samples_old).sample(int(positive_samples.shape[0]*1.2)).reset_index(drop=True)

        negative_samples["DateKey"] = positive_samples["DateKey"].sample(negative_samples.shape[0], replace=True).reset_index(drop=True)

        negative_samples["CustomerInterest"] = 0

        merged = positive_samples.append(negative_samples).drop_duplicates(["CustomerIdx", "IsinIdx", "BuySell", "DateKey"])



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

        negative_samples_new = trade_df[trade_df["DateKey"] >= key_date].sample(frac=0.95).drop_duplicates(["CustomerIdx","IsinIdx","BuySell"])

        negative_samples_old = trade_df[trade_df["DateKey"] < key_date].sample(frac=0.10).drop_duplicates(["CustomerIdx","IsinIdx","BuySell"])

        negative_samples = negative_samples_new.append(negative_samples_old).drop_duplicates(["CustomerIdx","IsinIdx","BuySell"])
        negative_samples["CustomerInterest"] = 0

        merged = positive_samples.append(negative_samples).drop_duplicates(["CustomerIdx","IsinIdx","BuySell"])

        merged["DateKey"] = set_date

        shuffled = shuffle(merged).reset_index(drop=True)
        return shuffled



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



    def monday_date(self, date):
        year = int(str(date)[:4])
        month = int(str(date)[4:6])
        day = int(str(date)[6:])

        date = datetime.datetime(year, month, day)
        time_delta = datetime.timedelta(days=date.weekday())
        monday_date = date - time_delta

        year = monday_date.year
        month = monday_date.month
        day = monday_date.day

        return int(str(year) + "{0:02d}".format(month) + "{0:02d}".format(day))







if __name__ == "__main__":
    trade_df = pd.read_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/Trade.csv")
    train = DataGenerator().generate_train_dataset(trade_df)
    print(train.shape)