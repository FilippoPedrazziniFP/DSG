import pandas as pd
import numpy as np
import datetime
class SequenceGenerator(object):

    def __init__(self, cust_frequency: pd.DataFrame, bond_frequency: pd.DataFrame):
        super(SequenceGenerator, self).__init__()

        self.cust_frequency = cust_frequency[["DateKey", "CustomerIdx", "CustBuy1w", "CustSell1w"]]
        self.bond_frequency = bond_frequency[["DateKey", "IsinIdx", "BondBuy1w", "BondSell1w"]]


    def get_cust_sequence_features(self, n_weeks = 8):
        seq = self.cust_frequency

        for i in range(1, n_weeks +1) :
            seq2 = seq[["DateKey", "CustomerIdx", "CustBuy1w", "CustSell1w"]]
            seq2["DateKey"] = seq2["DateKey"].apply(lambda x: self.days_before(x, i*7))
            seq2 = seq2.rename(columns={"CustBuy1w": "CustBuy"+str(i)+"WeeksBefore", "CustSell1w":"CustSell"+str(i)+"WeeksBefore"})
            seq = seq.merge(seq2, on=["DateKey", "CustomerIdx"], how = "left")

    def get_bond_sequence_features(self, n_weeks = 8):
        seq = self.bond_frequency

        for i in range(1, n_weeks +1) :

            seq2 = seq[["DateKey", "IsinIdx", "BondBuy1w", "BondSell1w"]]
            seq2["DateKey"] = seq2["DateKey"].apply(lambda x: self.days_before(x, i*7))
            seq2 = seq2.rename(columns={"BondBuy1w": "BondBuy"+str(i)+"WeeksBefore", "BondSell1w":"BondSell"+str(i)+"WeeksBefore"})
            seq = seq.merge(seq2, on=["DateKey", "IsinIdx"], how = "left")

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

if __name__ == "__main__":
    cust_frequency = pd.read_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/cust_frequency_features2018.csv")
    bond_frequency = pd.read_csv("/Users/claudioarcidiacono/PycharmProjects/DSG/data/bond_frequency_features2018.csv")
    seq_gen = SequenceGenerator(cust_frequency,bond_frequency)
    seq = seq_gen.get_bond_sequence_features()