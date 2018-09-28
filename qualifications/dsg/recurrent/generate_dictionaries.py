import pandas as pd
import matplotlib
import time
import pickle
import progressbar
import numpy as np

SEQ_LABEL = 5
TRAINING_DATE = 20170101
TRADE_DATA = "../../data/Trade.csv"
MARKET_DATA = "../../data/Market.csv"

def create_customer_dict(df_trade, max_date):
	print("GENERATING CUSTOMER DICTIONARY")
	cust_dict = {}
	for d in progressbar.progressbar(range(0, max_date - SEQ_LABEL, SEQ_LABEL)):
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

def create_bond_dict(df_trade, max_date):
	print("GENERATING BOND DICTIONARY")
	bond_dict = {}
	for d in progressbar.progressbar(range(0, max_date - SEQ_LABEL, SEQ_LABEL)):
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

def create_bond_dict_temporal_info(df_trade, df_market, max_date):
	print("GENERATING BOND DICTIONARY WITH MARKET INFO")
	bond_dict = {}
	for d in progressbar.progressbar(range(0, max_date - SEQ_LABEL, SEQ_LABEL)):
		for b in df_trade["IsinIdx"].unique():
			
			# Trade features
			df = df_trade[(df_trade["IsinIdx"] == b) & (df_trade["TradeDateKey"] >= d) 
				& (df_trade["TradeDateKey"] < d + SEQ_LABEL)]
			interactions = len(df.index)

			df_buy = df[df["BuySell_Buy"] == 1]
			buy = len(df_buy.index)
			df_sell = df[df["BuySell_Sell"] == 1]
			sell = len(df_sell.index)

			# Market Features
			df_m = df_market[(df_market["IsinIdx"] == b) & (df_market["DateKey"] >= d) 
				& (df_market["DateKey"] < d + SEQ_LABEL)]
			row = np.squeeze(df_m.groupby("IsinIdx").mean().values)
			try: 
				price = row[1]
				yield_ = row[2]
				zspread = row[3]
			except IndexError:
				price = 0
				yield_ = 0
				zspread = 0
			bond_dict[(d, b)] = [interactions, buy, sell, price, yield_, zspread]
	
	with open('bond_dict.pkl', 'wb') as f:
		pickle.dump(bond_dict, f)
	return

def create_cus_bond_dict(df_trade, max_date):
	print("GENERATING CUS-BOND DICTIONARY")
	cus_bond_dict = {}
	for d in progressbar.progressbar(range(0, max_date - SEQ_LABEL, SEQ_LABEL)):
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

def create_date_dictionary(df):
	dictionary_date = {}
	i = 0
	for row in df["TradeDateKey"].unique():
	    dictionary_date[row]=i
	    i = i+1
	return dictionary_date

def main():

	# Read CSV
	df_trade = pd.read_csv(TRADE_DATA)
	df_market = pd.read_csv(MARKET_DATA)

	# Reorder Trade by Date
	df_trade = df_trade.sort_values("TradeDateKey", ascending=True)
	df_market = df_market.sort_values("DateKey", ascending=True)

	# Create a dictionary to store {date : value}
	dictionary_date = create_date_dictionary(df_trade)
	print("MAXIMUM VALUE: ", max(zip(dictionary_date.values(), dictionary_date.keys())))

	# From a closer date
	df_trade = df_trade[df_trade["TradeDateKey"] >= TRAINING_DATE]
	df_market = df_market[df_market["DateKey"] >= TRAINING_DATE]

	# Delete Holding Values
	df_trade = df_trade[df_trade["TradeStatus"] != "Holding"]

	# Drop useless columns
	df_trade = df_trade.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)

	# Get Dummies for BuySell feature
	df_trade = pd.get_dummies(df_trade, columns=["BuySell"])

	print("CREATING DICTIONARY")
	print(df_trade.head(5))
	print(df_trade.describe())
	
	# Transform DateKey into a column from 0 to 632
	df_trade["TradeDateKey"] = df_trade["TradeDateKey"].apply(lambda x: dictionary_date[x])
	df_market["DateKey"] = df_market["DateKey"].apply(lambda x: dictionary_date[x])

	# Converting Dates
	max_date = dictionary_date[20180422]
	print("MAX_DATE: ", max_date)

	print("FIRST PREPROCESSING DONE: ")
	print(df_trade.head(5))
	print(df_trade.describe())

	# create_customer_dict(df_trade, max_date)
	# create_bond_dict(df_trade, max_date)
	# create_cus_bond_dict(df_trade, max_date)
	create_bond_dict_temporal_info(df_trade, df_market, max_date)

	return

main()





