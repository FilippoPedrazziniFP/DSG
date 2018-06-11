import pandas as pd
import datetime
import time

class FeatureExtractor(object):
    def __init__(self, trade: pd.DataFrame, chall: pd.DataFrame):
        super(FeatureExtractor, self).__init__()

        self.chall = chall
        self.trade = trade.rename(index=str, columns={"TradeDateKey": "DateKey"})

    def get_customer_frequency_features(self, col_suf='Cust', key=["CustomerIdx"], n_weeks = 8):

        return self.exract_features(col_suf, key, n_weeks)

    def get_bond_frequency_features(self, col_suf='Bond', key=["IsinIdx"], n_weeks = 8):

        return self.exract_features(col_suf, key, n_weeks)


    def get_customer_bond_frequency_features(self, col_suf='CustBond', key=["CustomerIdx", "IsinIdx"], n_weeks = 8):

        return self.exract_features(col_suf, key, n_weeks)



    def exract_features(self, col_suf, key, n_weeks):

        trade = self.trade
        trade.loc[:, "DateKey"] = self.trade["DateKey"].apply(lambda x: self.monday_date(x))

        chall = self.chall
        chall.loc[:, "DateKey"] = 20180423

        keys = trade[key + ["DateKey"]]
        keys = keys.append(chall[key + ["DateKey"]])
        keys = keys.drop_duplicates(key + ["DateKey"])


        shifted_trade = trade[key + ["DateKey", "BuySell"]]
        shifted_trade.loc[:, "DateKey"] = shifted_trade["DateKey"].apply(lambda x: self.days_after(x, 7))

        featureBuy = shifted_trade[shifted_trade["BuySell"] == "Buy"].groupby(key + ["DateKey"]).size().reset_index(
            name=(
                '%sBuy1w' % col_suf))

        featureSell = shifted_trade[shifted_trade["BuySell"] == "Sell"].groupby(key + ["DateKey"]).size().reset_index(
            name=(
                '%sSell1w' % col_suf))

        featureBuy = featureBuy[key + ["DateKey", ('%sBuy1w' % col_suf)]]
        featureSell = featureSell[key + ["DateKey", ('%sSell1w' % col_suf)]]

        seq = keys.merge(featureBuy, on=key + ["DateKey"], how='left')
        seq = seq.merge(featureSell, on=key + ["DateKey"], how='left')

        for i in range(1, n_weeks + 1):

            seq2 = seq[key + ["DateKey", ("%sBuy1w" % col_suf), ("%sSell1w" % col_suf)]]
            seq2["DateKey"] = seq2["DateKey"].apply(lambda x: self.days_before(x, i * 7))
            seq2 = seq2.rename(columns={("%sBuy1w" % col_suf): col_suf + "Buy" + str(i) + "WeeksBefore", (
                "%sSell1w" % col_suf): col_suf + "Sell" + str(i) + "WeeksBefore"})
            seq = seq.merge(seq2, on=key + ["DateKey"], how="left")

        seq = seq.fillna(0)

        trade = self.trade
        trade.loc[:, "Relative_Week"] = self.trade["DateKey"].apply(lambda x: self.days_after(self.monday_date(x), 7))
        trade = trade.loc[trade.groupby(key + ["Relative_Week", "BuySell"])["DateKey"].idxmin()]
        trade["Days_Before"] = trade.apply(lambda x: self.days_dist(x["Relative_Week"], x["DateKey"]), axis=1)

        featureBuy = trade[trade["BuySell"] == "Buy"]
        featureBuy = featureBuy.rename(index=str, columns={"Days_Before": ("%sLastBuy" % col_suf)})
        featureBuy = featureBuy[key + ["DateKey", ('%sLastBuy' % col_suf)]]

        featureSell = trade[trade["BuySell"] == "Sell"]
        featureSell = featureSell.rename(index=str, columns={"Days_Before": ("%sLastSell" % col_suf)})
        featureSell = featureSell[key + ["DateKey", ('%sLastSell' % col_suf)]]

        seq = seq.merge(featureBuy, on=key + ["DateKey"], how="left")
        seq = seq.merge(featureSell, on=key + ["DateKey"], how="left")
        seq = seq.fillna(100)

        seq.loc[seq["DateKey"] == 20180423, "DateKey"] = 20180424

        return seq


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


    feature_extractor = FeatureExtractor(trade_df[(trade_df["TradeStatus"] != 'Holding')], chall_df)

    t = time.time()
    print("Extracting Customer Features")
    cust_frequency = feature_extractor.get_customer_frequency_features()
    cust_frequency.to_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/cust_frequency_features2017.csv", index=False)
    print("extracted in {} seconds".format(time.time() - t))


    t = time.time()
    print("Extracting Bond Features")
    bond_frequency = feature_extractor.get_bond_frequency_features()
    bond_frequency.to_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/bond_frequency_features2017.csv", index=False)
    print("extracted in {} seconds".format(time.time() - t))

    t = time.time()
    print("Extracting Customer-Bond Features")
    cust_bond_frequency = feature_extractor.get_customer_bond_frequency_features()
    cust_bond_frequency.to_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/bondcust_frequency_features2017.csv", index=False)
    print("extracted in {} seconds".format(time.time() - t))
