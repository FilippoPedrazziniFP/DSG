import pandas as pd
import matplotlib
import time
import pickle
import progressbar
import numpy as np

MAX_DATE = 632
SEQ_LABEL = 7
TRADE_DATA = "../../data/Trade.csv"

def create_customer_dict(df_trade):
	print("GENERATING CUSTOMER DICTIONARY")
	cust_dict = {}
	for d in progressbar.progressbar(range(0, MAX_DATE - SEQ_LABEL, SEQ_LABEL)):
		for c in df_trade["CustomerIdx"].unique():
			df = df_trade[(df_trade["CustomerIdx"] == c) & (df_trade["TradeDateKey"] >= d) 
				& (df_trade["TradeDateKey"] < d + SEQ_LABEL)]
			interactions = len(df.index)
			df_buy = df[df["BuySell_Buy"] == 1]
			buy = len(df_buy.index)
			df_sell = df[df["BuySell_Sell"] == 1]
			sell = len(df_sell.index)
			cust_dict[(d, c)] = [interactions, sell, buy]
	with open('cust_dict.pkl', 'wb') as f:
		pickle.dump(cust_dict, f)
	return

def create_bond_dict(df_trade):
	print("GENERATING BOND DICTIONARY")
	bond_dict = {}
	for d in progressbar.progressbar(range(0, MAX_DATE - SEQ_LABEL, SEQ_LABEL)):
		for b in df_trade["IsinIdx"].unique():
			df = df_trade[(df_trade["IsinIdx"] == b) & (df_trade["TradeDateKey"] >= d) 
				& (df_trade["TradeDateKey"] < d + SEQ_LABEL)]
			interactions = len(df.index)
			df_buy = df[df["BuySell_Buy"] == 1]
			buy = len(df_buy.index)
			df_sell = df[df["BuySell_Sell"] == 1]
			sell = len(df_sell.index)
			bond_dict[(d, b)] = [interactions, sell, buy]
	
	with open('bond_dict.pkl', 'wb') as f:
		pickle.dump(bond_dict, f)
	return

def create_cus_bond_dict(df_trade):
	print("GENERATING CUS-BOND DICTIONARY")
	cus_bond_dict = {}
	for d in progressbar.progressbar(range(0, MAX_DATE - SEQ_LABEL, SEQ_LABEL)):
		df = df_trade[(df_trade["TradeDateKey"] >= d) & (df_trade["TradeDateKey"] < d + SEQ_LABEL)]
		df_ = df[["IsinIdx", "CustomerIdx"]].drop_duplicates()
		for index , row in df_.iterrows():
			df_b_c = df[(df["CustomerIdx"] == row["CustomerIdx"]) & (df["IsinIdx"] == row["IsinIdx"])]
			interactions = len(df_b_c.index)
			df_buy = df_b_c[df_b_c["BuySell_Buy"] == 1]
			buy = len(df_buy.index)
			df_sell = df_b_c[df_b_c["BuySell_Sell"] == 1]
			sell = len(df_sell.index)
			cus_bond_dict[(d, row["CustomerIdx"], row["IsinIdx"])] = [interactions, sell, buy]
	with open('cus_bond_dict.pkl', 'wb') as f:
		pickle.dump(cus_bond_dict, f)
	return

def main():

	# Read CSV
	df = pd.read_csv(TRADE_DATA)

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
	
	# Create a dictionary to store {date : value}
	dictionary_date = {}
	i = 0
	for row in df["TradeDateKey"].unique():
	    dictionary_date[row]=i
	    i = i+1
    
	with open('date_dict.pkl', 'wb') as f:
		pickle.dump(dictionary_date, f)
	print("MAXIMUM VALUE: ", max(zip(dictionary_date.values(), dictionary_date.keys())))

	# Transform DateKey into a column from 0 to 1000
	df["TradeDateKey"] = df["TradeDateKey"].apply(lambda x: dictionary_date[x])

	print("FIRST PREPROCESSING DONE: ")
	print(df.head(5))
	print(df.describe())

	# create_customer_dict(df)
	# create_bond_dict(df)
	# create_cus_bond_dict(df)

	return

main()





