import pandas as pd
import datetime
import time

class FeatureExtractor(object):
    def __init__(self, trade: pd.DataFrame, chall: pd.DataFrame):
        super(FeatureExtractor, self).__init__()

        self.chall = chall
        self.trade = trade.rename(index=str, columns={"TradeDateKey": "DateKey"})


    def get_customer_frequency_features(self):

        keys = self.trade[["CustomerIdx","DateKey"]]
        keys.loc[:,"DateKey"] = keys["DateKey"].apply(lambda x: self.monday_date(x))
        keys = keys.append(self.chall[["CustomerIdx","DateKey"]])
        keys = keys.drop_duplicates(["CustomerIdx","DateKey"])

        values = self.trade[["CustomerIdx","DateKey","BuySell"]]
        values.loc[:,"Date_BuySell"] = values.apply(lambda x: (x["DateKey"], x["BuySell"]), axis=1)
        values = values.groupby("CustomerIdx")["Date_BuySell"].apply(list)

        key_values = keys.join(values, on="CustomerIdx")
        key_values = key_values.fillna(value={"Date_BuySell": 0})

        key_values.loc[:, "CustBuy1w"] = key_values\
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
                            else sum(1 for t in x["Date_BuySell"][:100]
                                        if t[0] >= self.days_before(x["DateKey"], 7)
                                        and t[1] == 'Buy'),
                                        axis = 1)

        key_values.loc[:, "CustBuy2w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 14)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "CustBuy1m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 30)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "CustBuy2m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 60)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "CustTotBuy"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "CustSell1w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 7)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "CustSell2w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 14)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "CustSell1m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 30)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "CustSell2m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 60)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "CustTotSell"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "CustLastBuy"] = key_values \
            .apply(lambda x: 100 if x["Date_BuySell"] is 0
        else min([100] + [self.days_dist(x["DateKey"], t[0]) for t in x["Date_BuySell"]
                          if t[1] == 'Buy']),
                   axis=1)

        key_values.loc[:, "CustLastSell"] = key_values \
            .apply(lambda x: 100 if x["Date_BuySell"] is 0
        else min([100] + [self.days_dist(x["DateKey"], t[0]) for t in x["Date_BuySell"]
                          if t[1] == 'Sell']),
                   axis=1)


        return key_values

    def get_bond_frequency_features(self):

        keys = self.trade[["IsinIdx", "DateKey"]]
        keys.loc[:, "DateKey"] = keys["DateKey"].apply(lambda x: self.monday_date(x))
        keys = keys.append(self.chall[["IsinIdx","DateKey"]])
        keys = keys.drop_duplicates(["IsinIdx", "DateKey"])

        values = self.trade[["IsinIdx", "DateKey", "BuySell"]]
        values.loc[:, "Date_BuySell"] = values.apply(lambda x: (x["DateKey"], x["BuySell"]), axis=1)
        values = values.groupby("IsinIdx")["Date_BuySell"].apply(list)

        key_values = keys.join(values, on="IsinIdx")
        key_values = key_values.fillna(value={"Date_BuySell": 0})

        key_values.loc[:, "BondBuy1w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 7)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "BondBuy2w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 14)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "BondBuy1m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 30)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "BondBuy2m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 60)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "BondTotBuy"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "BondSell1w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 7)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "BondSell2w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 14)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "BondSell1m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 30)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "BondSell2m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 60)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "BondTotSell"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "BondLastBuy"] = key_values \
            .apply(lambda x: 100 if x["Date_BuySell"] is 0
        else min([100] + [self.days_dist(x["DateKey"], t[0]) for t in x["Date_BuySell"]
                          if t[1] == 'Buy']),
                   axis=1)

        key_values.loc[:, "BondLastSell"] = key_values \
            .apply(lambda x: 100 if x["Date_BuySell"] is 0
        else min([100] + [self.days_dist(x["DateKey"], t[0]) for t in x["Date_BuySell"]
                          if t[1] == 'Sell']),
                   axis=1)

        return key_values

    def get_bond_cust_frequency_features(self):

        keys = self.trade[["CustomerIdx","IsinIdx", "DateKey"]]
        keys.loc[:, "DateKey"] = keys["DateKey"].apply(lambda x: self.monday_date(x))
        keys = keys.append(self.chall[["CustomerIdx","IsinIdx","DateKey"]])
        keys = keys.drop_duplicates(["CustomerIdx","IsinIdx", "DateKey"])

        values = self.trade[["CustomerIdx","IsinIdx", "DateKey", "BuySell"]]
        values.loc[:, "Date_BuySell"] = values.apply(lambda x: (x["DateKey"], x["BuySell"]), axis=1)
        values = values.groupby(["CustomerIdx","IsinIdx"])["Date_BuySell"].apply(list)

        key_values = keys.join(values, on=["CustomerIdx","IsinIdx"])
        key_values = key_values.fillna(value={"Date_BuySell": 0})

        key_values.loc[:, "BondCustBuy1w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 7)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "BondCustBuy2w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 14)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "BondCustBuy1m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 30)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "BondCustBuy2m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 60)
                 and t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "BondCustTotBuy"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[1] == 'Buy'),
                   axis=1)

        key_values.loc[:, "BondCustSell1w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 7)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "BondCustSell2w"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 14)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "BondCustSell1m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 30)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "BondCustSell2m"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[0] >= self.days_before(x["DateKey"], 60)
                 and t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "BondCustTotSell"] = key_values \
            .apply(lambda x: 0 if x["Date_BuySell"] is 0
        else sum(1 for t in x["Date_BuySell"][:100]
                 if t[1] == 'Sell'),
                   axis=1)

        key_values.loc[:, "BondCustLastBuy"] = key_values \
            .apply(lambda x: 100 if x["Date_BuySell"] is 0
        else min([100] + [self.days_dist(x["DateKey"], t[0]) for t in x["Date_BuySell"]
                 if t[1] == 'Buy']),
                   axis=1)

        key_values.loc[:, "BondCustLastSell"] = key_values \
            .apply(lambda x: 100 if x["Date_BuySell"] is 0
        else min([100] + [self.days_dist(x["DateKey"], t[0]) for t in x["Date_BuySell"]
                          if t[1] == 'Sell']),
                   axis=1)

        return key_values

    def monday_date(self, date):
        date = int(date)
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

    def days_before(self, date, days):
        date=int(date)
        year = int(str(date)[:4])
        month = int(str(date)[4:6])
        day = int(str(date)[6:])

        date = datetime.datetime(year, month, day)
        time_delta = datetime.timedelta(days=days)
        date_days_before = date - time_delta

        year = date_days_before.year
        month = date_days_before.month
        day = date_days_before.day

        return int(str(year) + "{0:02d}".format(month) + "{0:02d}".format(day))

    def days_after(self, date, days):

        date=int(date)
        year = int(str(date)[:4])
        month = int(str(date)[4:6])
        day = int(str(date)[6:])

        date = datetime.datetime(year, month, day)
        time_delta = datetime.timedelta(days=days)
        date_days_before = date + time_delta

        year = date_days_before.year
        month = date_days_before.month
        day = date_days_before.day
        return int(str(year) + "{0:02d}".format(month) + "{0:02d}".format(day))


    def days_dist(self, date1, date2):

        date1 = int(date1)
        date2 = int(date2)

        year1 = int(str(date1)[:4])
        month1 = int(str(date1)[4:6])
        day1 = int(str(date1)[6:])

        year2 = int(str(date2)[:4])
        month2 = int(str(date2)[4:6])
        day2 = int(str(date2)[6:])

        date1 = datetime.datetime(year1, month1, day1)
        date2 = datetime.datetime(year2, month2, day2)

        delta = date1 - date2

        if delta.days < 0:
            return 100

        else:
            return delta.days

if __name__ == "__main__" :

    trade_df = pd.read_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/Trade.csv")
    chall_df = pd.read_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/Challenge_20180423.csv")


    feature_extractor = FeatureExtractor(trade_df[(trade_df["TradeDateKey"] > 20171100) & (trade_df["TradeStatus"] != 'Holding')], chall_df)

    t = time.time()
    print("Extracting Customer Features")
    cust_frequency = feature_extractor.get_customer_frequency_features()
    cust_frequency.to_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/cust_frequency_features2018.csv", index=False)
    print("extracted in {} seconds".format(time.time() - t))

    t = time.time()
    print("Extracting Bond Features")
    bond_frequency = feature_extractor.get_bond_frequency_features()
    bond_frequency.to_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/bond_frequency_features2018.csv", index=False)
    print("extracted in {} seconds".format(time.time() - t))

    t = time.time()
    print("Extracting Customer-Bond Features")
    cust_bond_frequency = feature_extractor.get_bond_cust_frequency_features()
    cust_bond_frequency.to_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/bondcust_frequency_features2018.csv", index=False)
    print("extracted in {} seconds".format(time.time() - t))
