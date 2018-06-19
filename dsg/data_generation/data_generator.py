import pandas as pd
import numpy as np
from dsg.data_loader import DataLoader
import pickle
import random
from sklearn.utils import shuffle

class FakeGeneratorFilo(object):
	def __init__(self):
		super(FakeGeneratorFilo, self).__init__()

	def generate_train_set_linear(self, df, from_date, to_date=None, from_date_features=20180101):
		"""	
			The method generates the Train set for the linear model;
			It uses 1 week as label in which CustomerInterest represents 
			the label to predict.

			@args
				df : DataFrame
				from_date : int -> representing the date from which 
					you want to start the week representing the training
				to_date : int -> representing the end of the week.

			@return
				features : DataFrame -> corresponfing to all the trades 
					before from_date
				labels : DataFrame -> representing the week of trading 
					to consider as label.
		"""
		# Delete Holding Values
		df = df[df["TradeStatus"] != "Holding"]

		# Drop Useless Columns
		df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)
		df = df.sort_values("TradeDateKey", ascending=True)

		# From Date Features
		df = df[df["TradeDateKey"] >= from_date_features]
		
		if to_date is None:
			labels = df[df["TradeDateKey"] >= from_date]
		else:
			labels = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]

		# Group By Customer and Bond to have the sum of interests during the period.
		labels = labels.groupby(["CustomerIdx", "IsinIdx"]).count()["CustomerInterest"].reset_index(level=['CustomerIdx', 'IsinIdx'])		

		# The Train is composed of all the values before the Date
		features = df[df["TradeDateKey"] <= from_date]		
		
		return features, labels

	def generate_train_set_svd(self, df, to_date=None, max_rating=5, from_date_features=20180101):
		"""
			The method generates the training data for Collaborative Filtering
			models; CustomerInterest is clipped to a maximum value of 5; in order 
			to represent a customer-bond rating.
			@args
				df : DataFrame 
				to_date : int -> representing the date in which we finish
					to consider the data.
				max_rating : int -> max rate possible

			@return
				features : DataFrame
				labels : DataFrame [CustomerIdx, IsinIdx, CustomerInterest]
			
		"""
		# Delete Holding Values
		df = df[df["TradeStatus"] != "Holding"]

		# Drop Useless Columns
		df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)
		df = df.sort_values("TradeDateKey", ascending=True)

		# From Date Features
		df = df[df["TradeDateKey"] >= from_date_features]

		if to_date is None:
			data = df
		else:
			data = df[df["TradeDateKey"] <= to_date]
		
		# Group By Customer and Bond to have the sum of interests during the period.
		data = data.groupby(["CustomerIdx", "IsinIdx"]).count()["CustomerInterest"].reset_index(level=['CustomerIdx', 'IsinIdx'])
		
		# Clip to a maximum of 5 Interactions
		data["CustomerInterest"] = data["CustomerInterest"].apply(lambda x: max_rating if x > max_rating else x)				
		
		return data

	def generate_train_set_classification(self, df, from_date, to_date=None, 
			from_date_label=20160101, from_date_features=20180101, clip=1):
		"""
			The method generates the training set for classification purposes.
		"""
		# Delete Holding Values
		df = df[df["TradeStatus"] != "Holding"]

		# Drop Useless Columns
		df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)
		df = df.sort_values("TradeDateKey", ascending=True)

		# Get Positive Values
		if to_date is None:
			positive_samples = df[df["TradeDateKey"] >= from_date]
		else:
			positive_samples = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]

		# Negative Samples
		negative_samples = df[df["TradeDateKey"] >= from_date_label]
		negative_samples = negative_samples.groupby(["CustomerIdx", "IsinIdx", "BuySell"]).count()
		negative_samples = negative_samples[negative_samples["CustomerInterest"] <= clip]
		negative_samples["CustomerInterest"] = negative_samples["CustomerInterest"].apply(lambda x: 0 if x > 0 else x)
		negative_samples = negative_samples.reset_index(level=['CustomerIdx', 'IsinIdx', "BuySell"])

		# Concatanate Negative and Positive Samples
		labels = pd.concat([positive_samples, negative_samples])
		labels = labels.drop(["TradeDateKey"], axis=1)
		
		# The Train is composed of all the values before the Date
		features = df[df["TradeDateKey"] >= from_date_features]
		features = features[features["TradeDateKey"] <= from_date]

		# Reorder Labels
		labels = labels[['CustomerIdx', 'IsinIdx', 'BuySell', 'CustomerInterest']]
		
		# Unique Values
		labels = labels.groupby(['CustomerIdx', 'IsinIdx', 'BuySell']).sum()
		labels = labels.reset_index(level=['CustomerIdx', 'IsinIdx', 'BuySell'])
		labels["CustomerInterest"] = labels["CustomerInterest"].apply(lambda x: 1 if x > 1 else x)

		# One Hot Encoding for Sell and Buy
		labels = pd.get_dummies(labels, columns=['BuySell'])
		labels = labels[['CustomerIdx', 'IsinIdx', "BuySell_Buy", "BuySell_Sell", 'CustomerInterest']]
		return features, labels

	def generate_test_claudio(self, trade_df, from_day=20180415, till_day=20180422, set_date=20180417, key_date=20171022):

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
		negative_samples_old = trade_df[trade_df["DateKey"] < key_date].sample(frac=0.15)

		negative_samples = negative_samples_new.append(negative_samples_old).drop_duplicates(["CustomerIdx","IsinIdx","BuySell"])

		negative_samples["CustomerInterest"] = 0

		merged = positive_samples.append(negative_samples).drop_duplicates(["CustomerIdx","IsinIdx","BuySell"])

		merged["DateKey"] = set_date

		shuffled = shuffle(merged).reset_index(drop=True)
		return shuffled

	def generate_test_set(self, df, from_date, to_date=None, from_date_label=20171122):
		"""
			The method creates a dataframe for testing purposes similar to the one 
			of the competition. It uses the last 6 months interactions as negative labels.

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
		# Delete Holding Values
		df = df[df["TradeStatus"] != "Holding"]

		# Drop Useless Columns
		df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)

		# One Hot Encoding for Sell and Buy
		df = pd.get_dummies(df, columns=['BuySell'])
		
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
		test_set = pd.concat([positive_samples, negative_samples, negative_samples_neg, positive_samples_neg])
		test_set = test_set.drop(["TradeDateKey"], axis=1)

		# Unique Values
		test_set = test_set.groupby(['CustomerIdx', 'IsinIdx', "BuySell_Sell", "BuySell_Buy"]).sum()
		test_set = test_set.reset_index(level=['CustomerIdx', 'IsinIdx', "BuySell_Sell", "BuySell_Buy"])
		test_set["CustomerInterest"] = test_set["CustomerInterest"].apply(lambda x: 1 if x > 1 else x)

		# Reorder the columns
		test_set = test_set[['CustomerIdx', 'IsinIdx', "BuySell_Buy", "BuySell_Sell", 'CustomerInterest']]

		return test_set

	def generate_features_labels_classification(self, df, from_date, from_date_feat_label, to_date=None):
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

		# One Hot Encoding for Sell and Buy
		df = pd.get_dummies(df, columns=['BuySell'])
		
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
		negative_samples = df[(df["TradeDateKey"] >= from_date_feat_label) & (df["TradeDateKey"] < from_date)]
		negative_samples_neg = df[(df["TradeDateKey"] >= from_date_feat_label) & (df["TradeDateKey"] < from_date)]

		# Opposite Positive
		positive_samples_neg["BuySell_Buy"] = negative_samples["BuySell_Sell"]
		positive_samples_neg["BuySell_Sell"] = negative_samples["BuySell_Buy"]

		# Double Negative Samples
		negative_samples_neg = df[(df["TradeDateKey"] >= from_date_feat_label) & (df["TradeDateKey"] < from_date)]
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

		# The Train is composed of all the values before the Date
		features = df[df["TradeDateKey"] >= from_date_feat_label]
		features = features[features["TradeDateKey"] <= from_date]

		return features, labels

	def generate_features_labels_regression(self, df, from_date, from_date_feat_label, to_date=None):
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

		# One Hot Encoding for Sell and Buy
		df = pd.get_dummies(df, columns=['BuySell'])
		
		if to_date is None:
			positive_samples = df[df["TradeDateKey"] >= from_date]
		else:
			positive_samples = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]

		positive_samples = positive_samples.groupby(["CustomerIdx", "IsinIdx", "BuySell_Buy", "BuySell_Sell"])\
			.count().reset_index(level=['CustomerIdx', 'IsinIdx', "BuySell_Buy", "BuySell_Sell"])
		positive_samples_neg = positive_samples.groupby(["CustomerIdx", "IsinIdx", "BuySell_Buy", "BuySell_Sell"])\
			.count().reset_index(level=['CustomerIdx', 'IsinIdx', "BuySell_Buy", "BuySell_Sell"])

		positive_samples_neg["BuySell_Buy"] = positive_samples["BuySell_Sell"]
		positive_samples_neg["BuySell_Sell"] = positive_samples["BuySell_Buy"]
		positive_samples_neg["CustomerInterest"] = positive_samples_neg["CustomerInterest"]\
            .apply(lambda x: 0 if x > 0 else x)

		positive_samples = positive_samples.drop(["TradeDateKey"], axis=1)
		positive_samples_neg = positive_samples_neg.drop(["TradeDateKey"], axis=1)
		
		# Negative Samples
		negative_samples = df[(df["TradeDateKey"] >= from_date_feat_label) & (df["TradeDateKey"] < from_date)]
		negative_samples_neg = df[(df["TradeDateKey"] >= from_date_feat_label) & (df["TradeDateKey"] < from_date)]

		# Double Negative Samples
		negative_samples_neg = df[(df["TradeDateKey"] >= from_date_feat_label) & (df["TradeDateKey"] < from_date)]
		negative_samples_neg["BuySell_Buy"] = negative_samples["BuySell_Sell"]
		negative_samples_neg["BuySell_Sell"] = negative_samples["BuySell_Buy"]

		# Put to zero all the negative
		negative_samples["CustomerInterest"] = negative_samples["CustomerInterest"]\
			.apply(lambda x: 0 if x > 0 else x)
		negative_samples_neg["CustomerInterest"] = negative_samples_neg["CustomerInterest"]\
			.apply(lambda x: 0 if x > 0 else x)

		negative_samples = negative_samples.drop(["TradeDateKey"], axis=1)
		negative_samples_neg = negative_samples_neg.drop(["TradeDateKey"], axis=1)
		negative_samples_neg = negative_samples_neg[['CustomerIdx', 'IsinIdx', "BuySell_Buy", "BuySell_Sell", 'CustomerInterest']]
		negative_samples = negative_samples[['CustomerIdx', 'IsinIdx', "BuySell_Buy", "BuySell_Sell", 'CustomerInterest']]

		# Concatanate Negative and Positive Samples
		labels = pd.concat([positive_samples, positive_samples_neg, negative_samples_neg, negative_samples])

		# Unique Values
		labels = labels.drop_duplicates(subset=['CustomerIdx', 'IsinIdx', "BuySell_Buy", "BuySell_Sell"])

		# Reorder the columns
		labels = labels[['CustomerIdx', 'IsinIdx', "BuySell_Buy", "BuySell_Sell", 'CustomerInterest']]
		
		# Clip to a maximum of 5 Interactions or normalize
		# max_value = labels["CustomerInterest"].max()
		labels["CustomerInterest"] = labels["CustomerInterest"].apply(lambda x: 5 if x > 5 else x)	

		# The Train is composed of all the values before the Date
		features = df[df["TradeDateKey"] >= from_date_feat_label]
		features = features[features["TradeDateKey"] <= from_date]

		return features, labels

	def generate_dictionaries(self, df):

		cust_dict = self.create_customer_features_dict(df)
		bond_dict = self.create_bonds_features_dict(df)
		cus_bond_dict = self.create_cus_bond_features_dict(df)
		
		return cust_dict, bond_dict, cus_bond_dict

	def create_date_dictionary(self, df):
		dictionary_date = {}
		i = 0
		for row in df["TradeDateKey"].unique():
		    dictionary_date[row]=i
		    i = i+1
		return dictionary_date

	def train_test_val_split(self, data):
		train = data[0:-2]
		val = data[-2]
		test = data[-1]
		return train, test, val, data

	def generate_train_test_val(self, weeks=6):
		"""
			The method generates the train/test/validation
			set to train and test the model.
		"""
		week_len = 5
		month_len = 6
		week_month = 4

		# Load the Data
		loader = DataLoader()
		df = loader.load_trade_data()
		self.df_bond = loader.load_market_data()

		# Reorder the data
		df = df.sort_values("TradeDateKey", ascending=True)
		self.df_bond = self.df_bond.sort_values("DateKey", ascending=True)

		# Date Dictionary to Transform the Dates
		self.date_dict = self.create_date_dictionary(df)

		# Delete Holding Values
		df = df[df["TradeStatus"] != "Holding"]

		# Drop Useless Columns
		df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)

		# Transform the Dates
		df["TradeDateKey"] = df["TradeDateKey"].apply(lambda x: self.date_dict[x])
		self.df_bond["DateKey"] = self.df_bond["DateKey"].apply(lambda x: self.date_dict[x])

		# Converting Dates
		max_date = self.date_dict[20180422]
		print("MAX_DATE: ", max_date)

		# Generate Weeks
		data_weeks = []
		for i in range(max_date - week_len*weeks, max_date, week_len):
			print("WEEK: ", i)
			features, labels = self.generate_features_labels_regression(
				df=df, 
				from_date=i, 
				to_date=i+week_len,
				from_date_feat_label=i-month_len*week_len*week_month
				)
			print(labels.shape)
			data_weeks.append((features, labels))

		# Splitting into Train/Test and Validation
		train, test, val, data = self.train_test_val_split(data_weeks)

		train_samples = []
		for t in train:
			week = self.create_set(t)
			print(week.shape)
			train_samples.append(week)

		# Concatanate + Features/Labels Split
		train_samples = np.concatenate(train_samples, axis=0)
		print(train_samples.shape)
		X_train, y_train = self.features_labels_split(train_samples)
		
		with open('X_train.pkl', 'wb') as f:
			pickle.dump(X_train, f)	

		with open('y_train.pkl', 'wb') as f:
			pickle.dump(y_train, f)		
		
		print(X_train.shape)
		print(y_train.shape)

		data_samples = []
		for t in data:
			week = self.create_set(t)
			print(week.shape)
			data_samples.append(week)

		# Concatanate + Features/Labels Split
		data_samples = np.concatenate(data_samples, axis=0)
		print(data_samples.shape)
		X, y = self.features_labels_split(data_samples)

		with open('X.pkl', 'wb') as f:
			pickle.dump(X, f)	

		with open('y.pkl', 'wb') as f:
			pickle.dump(y, f)	

		print(X.shape)
		print(y.shape)

		# Test set
		test = self.create_set(test)
		X_test, y_test = self.features_labels_split(test)

		with open('X_test.pkl', 'wb') as f:
			pickle.dump(X_test, f)	

		with open('y_test.pkl', 'wb') as f:
			pickle.dump(y_test, f)

		print(X_test.shape)
		print(y_test.shape)

		# Val Set
		val = self.create_set(val)
		X_val, y_val = self.features_labels_split(val)

		with open('X_val.pkl', 'wb') as f:
			pickle.dump(X_val, f)	

		with open('y_val.pkl', 'wb') as f:
			pickle.dump(y_val, f)	

		# Load challenge data
		challenge = loader.load_challenge_data()
		challenge = self.challenge_transform(challenge)
		X_challenge = self.create_challenge_set(challenge, df)

		with open('X_challenge.pkl', 'wb') as f:
			pickle.dump(X_challenge, f)	

		return X_train, y_train, X_test, y_test, X_val, y_val, X, y, X_challenge

	def create_challenge_set(self, challenge, df):
		
		# OHE of the buy sell-column
		df = pd.get_dummies(df, columns=["BuySell"])

		# Generate Dictionary from Features
		cust_dict, bond_dict, cus_bond_dict = self.generate_dictionaries(df)

		train = challenge.values
		train_set = []
		for sample in train:			
			try:
				customer_features = cust_dict[sample[0]]
				customer_features = np.asarray(customer_features)
			except KeyError:
				customer_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
			
			try: 
				bond_features = bond_dict[sample[1]]
				bond_features = np.asarray(bond_features)
			except KeyError:
				bond_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

			try: 
				cus_bond_features = cus_bond_dict[(sample[0], sample[1])]
				cus_bond_features = np.asarray(cus_bond_features)
			except KeyError:
				cus_bond_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
			
			row = np.append(customer_features, bond_features)
			row = np.append(row, cus_bond_features)
			sell_buy = [sample[2], sample[3]]
			row = np.append(row, sell_buy)
			train_set.append(row)
		train_set = np.asarray(train_set)
		return train_set


	def create_set(self, data):
		"""
			The method creates a matrix with 2 columns:
			[CustomerIdx, NormalizedFrequency] given a dictionary
			of CustomersIdx : Normalized Values.
		"""
		features = data[0]
		labels = data[1]

		# Generate Dictionary from Features
		cust_dict, bond_dict, cus_bond_dict = self.generate_dictionaries(features)

		train = labels.values
		train_set = []
		for sample in train:			
			try:
				customer_features = cust_dict[sample[0]]
				customer_features = np.asarray(customer_features)
			except KeyError:
				customer_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
			
			try: 
				bond_features = bond_dict[sample[1]]
				bond_features = np.asarray(bond_features)
			except KeyError:
				bond_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

			try: 
				cus_bond_features = cus_bond_dict[(sample[0], sample[1])]
				cus_bond_features = np.asarray(cus_bond_features)
			except KeyError:
				cus_bond_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
			
			label = sample[-1]
			row = np.append(customer_features, bond_features)
			row = np.append(row, cus_bond_features)
			sell_buy = [sample[2], sample[3]]
			row = np.append(row, sell_buy)
			row = np.append(row, label)
			train_set.append(row)
		train_set = np.asarray(train_set)
		return train_set

	def challenge_transform(self, df):
		
		df = df.drop(["PredictionIdx", "DateKey"], axis=1)
		df = pd.get_dummies(df, columns=["BuySell"])
		df = df[['CustomerIdx', 'IsinIdx', "BuySell_Buy", 
			"BuySell_Sell", 'CustomerInterest']]
		return df

	def features_labels_split(self, X):
		"""
			The method splits the data into Features and Labels
			returning a numpy array
			@args
				X : numpy array
			@return
				features : numpy array
				labels : numpy array
		"""
		features = X[:,0:-1]
		labels = X[:,-1]
		return features, labels

	def create_bonds_features_dict(self, df_train):
		"""
			The method creates a dictionary with
			BondIdx : features.
		"""
		max_date = df_train["TradeDateKey"].max()

		# Number of Interactions during the period
		num_int = df_train.groupby('IsinIdx').count()
		num_int["NumInt"] = num_int["TradeDateKey"]
		num_int = num_int.reset_index()
		num_int = num_int[["IsinIdx", "NumInt"]]

		# Last interactions
		last_int = df_train.groupby('IsinIdx').max()
		last_int["LastInt"] = last_int["TradeDateKey"].apply(lambda x: max_date - x)
		last_int = last_int.reset_index()
		last_int = last_int[["IsinIdx", "LastInt"]]

		# First Merge
		df = pd.merge(num_int, last_int, on=['IsinIdx'], how='left')

		# Last month Interactions
		last_month_df = df_train[df_train["TradeDateKey"] >= max_date -20]
		last_month_df = last_month_df.groupby('IsinIdx').count()
		last_month_df["LastMonthInt"] = last_month_df["TradeDateKey"]
		last_month_df = last_month_df.reset_index()
		last_month_df = last_month_df[["IsinIdx", "LastMonthInt"]]

		# Second Merge
		df = pd.merge(df, last_month_df, on=['IsinIdx'], how='left')

		# Last Week Interactions
		last_week_df = df_train[df_train["TradeDateKey"] >= max_date -5]
		last_week_df = last_week_df.groupby('IsinIdx').count()
		last_week_df["LastWeekInt"] = last_week_df["TradeDateKey"]
		last_week_df = last_week_df.reset_index()
		last_week_df = last_week_df[["IsinIdx", "LastWeekInt"]]

		# Third Merge
		df = pd.merge(df, last_week_df, on=['IsinIdx'], how='left')

		# Last 2 Week Interactions
		last_2_week_df = df_train[df_train["TradeDateKey"] >= max_date -10]
		last_2_week_df = last_2_week_df.groupby('IsinIdx').count()
		last_2_week_df["Last2WeekInt"] = last_2_week_df["TradeDateKey"]
		last_2_week_df = last_2_week_df.reset_index()
		last_2_week_df = last_2_week_df[["IsinIdx", "Last2WeekInt"]]

		# Fourth Merge
		df = pd.merge(df, last_2_week_df, on=['IsinIdx'], how='left')

		# Last 2 Month Interactions
		last_2_month_df = df_train[df_train["TradeDateKey"] >= max_date -40]
		last_2_month_df = last_2_month_df.groupby('IsinIdx').count()
		last_2_month_df["Last2MonthInt"] = last_2_month_df["TradeDateKey"]
		last_2_month_df = last_2_month_df.reset_index()
		last_2_month_df = last_2_month_df[["IsinIdx", "Last2MonthInt"]]

		# Fifth Merge
		df = pd.merge(df, last_2_month_df, on=['IsinIdx'], how='left')

		# Number of sell interactions in the last week
		last_week_df_sell = df_train[(df_train["TradeDateKey"] >= max_date -5) & (df_train["BuySell_Sell"] == 1)]
		last_week_df_sell = last_week_df_sell.groupby(['IsinIdx']).count()
		last_week_df_sell["LastWeekSell"] = last_week_df_sell["TradeDateKey"]
		last_week_df_sell = last_week_df_sell.reset_index()
		last_week_df_sell = last_week_df_sell[["IsinIdx", "LastWeekSell"]]

		# Sixth Merge
		df = pd.merge(df, last_week_df_sell, on=['IsinIdx'], how='left')

		# Number of sell interactions in the last 2 weeks
		last_2_week_df_sell = df_train[(df_train["TradeDateKey"] >= max_date -10) & (df_train["BuySell_Sell"] == 1)]
		last_2_week_df_sell = last_2_week_df_sell.groupby(["IsinIdx"]).count()
		last_2_week_df_sell["Last2WeekSell"] = last_2_week_df_sell["TradeDateKey"]
		last_2_week_df_sell = last_2_week_df_sell.reset_index()
		last_2_week_df_sell = last_2_week_df_sell[["IsinIdx", "Last2WeekSell"]]

		# Seventh Merge
		df = pd.merge(df, last_2_week_df_sell, on=['IsinIdx'], how='left')

		# Number of sell interactions in the last month
		last_month_df_sell = df_train[(df_train["TradeDateKey"] >= max_date -20) & (df_train["BuySell_Sell"] == 1)]
		last_month_df_sell = last_month_df_sell.groupby(['IsinIdx']).count()
		last_month_df_sell["LastMonthSell"] = last_month_df_sell["TradeDateKey"]
		last_month_df_sell = last_month_df_sell.reset_index()
		last_month_df_sell = last_month_df_sell[["IsinIdx", "LastMonthSell"]]

		# Eigth Merge
		df = pd.merge(df, last_month_df_sell, on=['IsinIdx'], how='left')

		# Number of sell interactions since the beginning
		df_sell = df_train[df_train["BuySell_Sell"] == 1]
		df_sell = df_sell.groupby(['IsinIdx']).count()
		df_sell["SellInt"] = df_sell["TradeDateKey"]
		df_sell = df_sell.reset_index()
		df_sell = df_sell[["IsinIdx", "SellInt"]]

		# Ninth Merge
		df = pd.merge(df, df_sell, on=['IsinIdx'], how='left')

		# Number of sell interactions in the last week
		last_week_df_buy = df_train[(df_train["TradeDateKey"] >= max_date -5) & (df_train["BuySell_Buy"] == 1)]
		last_week_df_buy = last_week_df_buy.groupby(['IsinIdx']).count()
		last_week_df_buy["LastWeekBuy"] = last_week_df_buy["TradeDateKey"]
		last_week_df_buy = last_week_df_buy.reset_index()
		last_week_df_buy = last_week_df_buy[["IsinIdx", "LastWeekBuy"]]

		# Tenth Merge
		df = pd.merge(df, last_week_df_buy, on=['IsinIdx'], how='left')

		# Number of sell interactions in the last 2 weeks
		last_2_week_df_buy = df_train[(df_train["TradeDateKey"] >= max_date -10) & (df_train["BuySell_Buy"] == 1)]
		last_2_week_df_buy = last_2_week_df_buy.groupby(['IsinIdx']).count()
		last_2_week_df_buy["Last2WeekBuy"] = last_2_week_df_buy["TradeDateKey"]
		last_2_week_df_buy = last_2_week_df_buy.reset_index()
		last_2_week_df_buy = last_2_week_df_buy[["IsinIdx", "Last2WeekBuy"]]

		# Eleventh Merge
		df = pd.merge(df, last_2_week_df_buy, on=['IsinIdx'], how='left')

		# Number of sell interactions in the last month
		last_month_df_buy = df_train[(df_train["TradeDateKey"] >= max_date -20) & (df_train["BuySell_Buy"] == 1)]
		last_month_df_buy = last_month_df_buy.groupby(['IsinIdx']).count()
		last_month_df_buy["LastMonthBuy"] = last_month_df_buy["TradeDateKey"]
		last_month_df_buy = last_month_df_buy.reset_index()
		last_month_df_buy = last_month_df_buy[["IsinIdx", "LastMonthBuy"]]

		# Twelveth Merge
		df = pd.merge(df, last_month_df_buy, on=['IsinIdx'], how='left')

		# Number of sell interactions since the beginning
		df_buy = df_train[df_train["BuySell_Buy"] == 1]
		df_buy = df_buy.groupby(['IsinIdx']).count()
		df_buy["BuyInt"] = df_buy["TradeDateKey"]
		df_buy = df_buy.reset_index()
		df_buy = df_buy[["IsinIdx", "BuyInt"]]

		# Twelveth Merge
		df = pd.merge(df, df_buy, on=['IsinIdx'], how='left')

		""" Market Features """

		# AVG Spread in the last week
		last_week_spread_df = self.df_bond[self.df_bond["DateKey"] >= max_date -5]
		last_week_spread_df = last_week_spread_df.groupby(['IsinIdx']).mean()
		last_week_spread_df = last_week_spread_df.reset_index()
		last_week_spread_df["LastWeekZSpread"] = last_week_spread_df["ZSpread"]
		last_week_spread_df = last_week_spread_df[["IsinIdx", "LastWeekZSpread"]]

		# Twelveth Merge
		df = pd.merge(df, last_week_spread_df, on=['IsinIdx'], how='left')

		# AVG Spread in the last 2 week
		last_2_week_spread_df = self.df_bond[self.df_bond["DateKey"] >= max_date -10]
		last_2_week_spread_df = last_2_week_spread_df.groupby(['IsinIdx']).mean()
		last_2_week_spread_df = last_2_week_spread_df.reset_index()
		last_2_week_spread_df["Last2WeekZSpread"] = last_2_week_spread_df["ZSpread"]
		last_2_week_spread_df = last_2_week_spread_df[["IsinIdx", "Last2WeekZSpread"]]

		# Twelveth Merge
		df = pd.merge(df, last_2_week_spread_df, on=['IsinIdx'], how='left')

		# AVG Spread in the last month
		last_month_spread_df = self.df_bond[self.df_bond["DateKey"] >= max_date -20]
		last_month_spread_df = last_month_spread_df.groupby(['IsinIdx']).mean()
		last_month_spread_df = last_month_spread_df.reset_index()
		last_month_spread_df["LastMonthZSpread"] = last_month_spread_df["ZSpread"]
		last_month_spread_df = last_month_spread_df[["IsinIdx", "LastMonthZSpread"]]

		# Twelveth Merge
		df = pd.merge(df, last_month_spread_df, on=['IsinIdx'], how='left')

		# AVG Spread since the beginning
		spread_df = self.df_bond[self.df_bond["DateKey"] >= max_date -6*5*4]
		spread_df = spread_df.groupby(['IsinIdx']).mean()
		spread_df = spread_df.reset_index()
		spread_df["AVGZSpread"] = spread_df["ZSpread"]
		spread_df = spread_df[["IsinIdx", "AVGZSpread"]]

		# Twelveth Merge
		df = pd.merge(df, spread_df, on=['IsinIdx'], how='left')

		# AVG Yield in the last week
		last_week_yield_df = self.df_bond[self.df_bond["DateKey"] >= max_date -5]
		last_week_yield_df = last_week_yield_df.groupby(['IsinIdx']).mean()
		last_week_yield_df = last_week_yield_df.reset_index()
		last_week_yield_df["LastWeekYield"] = last_week_yield_df["Yield"]
		last_week_yield_df = last_week_yield_df[["IsinIdx", "LastWeekYield"]]

		# Twelveth Merge
		df = pd.merge(df, last_week_yield_df, on=['IsinIdx'], how='left')

		# AVG Yield in the last 2 week
		last_2_week_yield_df = self.df_bond[self.df_bond["DateKey"] >= max_date -10]
		last_2_week_yield_df = last_2_week_yield_df.groupby(['IsinIdx']).mean()
		last_2_week_yield_df = last_2_week_yield_df.reset_index()
		last_2_week_yield_df["Last2WeekYield"] = last_2_week_yield_df["Yield"]
		last_2_week_yield_df = last_2_week_yield_df[["IsinIdx", "Last2WeekYield"]]

		# Twelveth Merge
		df = pd.merge(df, last_2_week_yield_df, on=['IsinIdx'], how='left')

		# AVG Yield in the last month
		last_month_yield_df = self.df_bond[self.df_bond["DateKey"] >= max_date -20]
		last_month_yield_df = last_month_yield_df.groupby(['IsinIdx']).mean()
		last_month_yield_df = last_month_yield_df.reset_index()
		last_month_yield_df["LastMonthYield"] = last_month_yield_df["Yield"]
		last_month_yield_df = last_month_yield_df[["IsinIdx", "LastMonthYield"]]

		# Twelveth Merge
		df = pd.merge(df, last_month_yield_df, on=['IsinIdx'], how='left')

		# AVG Yield since the beginning
		yield_df = self.df_bond[self.df_bond["DateKey"] >= max_date -6*5*4]
		yield_df = yield_df.groupby(['IsinIdx']).mean()
		yield_df = yield_df.reset_index()
		yield_df["AVGYield"] = yield_df["Yield"]
		yield_df = yield_df[["IsinIdx", "AVGYield"]]

		# Twelveth Merge
		df = pd.merge(df, yield_df, on=['IsinIdx'], how='left')

		# AVG Yield in the last week
		last_week_price_df = self.df_bond[self.df_bond["DateKey"] >= max_date -5]
		last_week_price_df = last_week_price_df.groupby(['IsinIdx']).mean()
		last_week_price_df = last_week_price_df.reset_index()
		last_week_price_df["LastWeekPrice"] = last_week_price_df["Price"]
		last_week_price_df = last_week_price_df[["IsinIdx", "LastWeekPrice"]]

		# Twelveth Merge
		df = pd.merge(df, last_week_price_df, on=['IsinIdx'], how='left')

		# AVG Yield in the last 2 week
		last_2_week_price_df = self.df_bond[self.df_bond["DateKey"] >= max_date -10]
		last_2_week_price_df = last_2_week_price_df.groupby(['IsinIdx']).mean()
		last_2_week_price_df = last_2_week_price_df.reset_index()
		last_2_week_price_df["Last2WeekPrice"] = last_2_week_price_df["Price"]
		last_2_week_price_df = last_2_week_price_df[["IsinIdx", "Last2WeekPrice"]]

		# Twelveth Merge
		df = pd.merge(df, last_2_week_price_df, on=['IsinIdx'], how='left')

		# AVG Yield in the last month
		last_month_price_df = self.df_bond[self.df_bond["DateKey"] >= max_date -20]
		last_month_price_df = last_month_price_df.groupby(['IsinIdx']).mean()
		last_month_price_df = last_month_price_df.reset_index()
		last_month_price_df["LastMonthPrice"] = last_month_price_df["Price"]
		last_month_price_df = last_month_price_df[["IsinIdx", "LastMonthPrice"]]

		# Twelveth Merge
		df = pd.merge(df, last_month_price_df, on=['IsinIdx'], how='left')

		# AVG Yield since the beginning
		price_df = self.df_bond[self.df_bond["DateKey"] >= max_date -6*5*4]
		price_df = price_df.groupby(['IsinIdx']).mean()
		price_df = price_df.reset_index()
		price_df["AVGPrice"] = price_df["Price"]
		price_df = price_df[["IsinIdx", "AVGPrice"]]

		# Twelveth Merge
		df = pd.merge(df, price_df, on=['IsinIdx'], how='left')

		# Fill NAN with Zeros
		df.fillna(0, inplace=True)
		
		df['Features'] = list(zip(
			df['NumInt'], df['LastInt'], df["LastMonthInt"], df["LastWeekInt"], df["Last2WeekInt"], df["Last2MonthInt"], 
			df["LastWeekSell"], df["Last2WeekSell"], df["LastMonthSell"], df["SellInt"], 
			df["LastWeekBuy"], df["Last2WeekBuy"], df["LastMonthBuy"], df["BuyInt"], 
			df["LastWeekZSpread"], df["Last2WeekZSpread"], df["LastMonthZSpread"], df["AVGZSpread"],
			df["LastWeekYield"], df["Last2WeekYield"], df["LastMonthYield"], df["AVGYield"],
			df["LastWeekPrice"], df["Last2WeekPrice"], df["LastMonthPrice"], df["AVGPrice"]))
		df_dict = df.groupby("IsinIdx")["Features"].apply(list).to_dict()
		
		return df_dict

	def create_customer_features_dict(self, df_train):
		"""
			The method creates a dictionary where the key 
			represents the customer id and the value
			is a list with some features:

			CustomerIdx : Freq, Last Int, Last Month, Last Week, 
				Last 2 Week, Last 2 Months
		"""
		print(df_train.describe())
		print(df_train.head())

		max_date = df_train["TradeDateKey"].max()

		# Number of Interactions during the period
		num_int = df_train.groupby('CustomerIdx').count()
		num_int["NumInt"] = num_int["TradeDateKey"]
		num_int = num_int.reset_index()
		num_int = num_int[["CustomerIdx", "NumInt"]]

		# Last interactions
		last_int = df_train.groupby('CustomerIdx').max()
		last_int["LastInt"] = last_int["TradeDateKey"].apply(lambda x: max_date - x)
		last_int = last_int.reset_index()
		last_int = last_int[["CustomerIdx", "LastInt"]]

		# First Merge
		df = pd.merge(num_int, last_int, on=['CustomerIdx'], how='left')

		# Last month Interactions
		last_month_df = df_train[df_train["TradeDateKey"] >= max_date -20]
		last_month_df = last_month_df.groupby('CustomerIdx').count()
		last_month_df["LastMonthInt"] = last_month_df["TradeDateKey"]
		last_month_df = last_month_df.reset_index()
		last_month_df = last_month_df[["CustomerIdx", "LastMonthInt"]]

		# Second Merge
		df = pd.merge(df, last_month_df, on=['CustomerIdx'], how='left')

		# Last Week Interactions
		last_week_df = df_train[df_train["TradeDateKey"] >= max_date -5]
		last_week_df = last_week_df.groupby('CustomerIdx').count()
		last_week_df["LastWeekInt"] = last_week_df["TradeDateKey"]
		last_week_df = last_week_df.reset_index()
		last_week_df = last_week_df[["CustomerIdx", "LastWeekInt"]]

		# Third Merge
		df = pd.merge(df, last_week_df, on=['CustomerIdx'], how='left')

		# Last 2 Week Interactions
		last_2_week_df = df_train[df_train["TradeDateKey"] >= max_date -10]
		last_2_week_df = last_2_week_df.groupby('CustomerIdx').count()
		last_2_week_df["Last2WeekInt"] = last_2_week_df["TradeDateKey"]
		last_2_week_df = last_2_week_df.reset_index()
		last_2_week_df = last_2_week_df[["CustomerIdx", "Last2WeekInt"]]

		# Fourth Merge
		df = pd.merge(df, last_2_week_df, on=['CustomerIdx'], how='left')

		# Last 2 Month Interactions
		last_2_month_df = df_train[df_train["TradeDateKey"] >= max_date -40]
		last_2_month_df = last_2_month_df.groupby('CustomerIdx').count()
		last_2_month_df["Last2MonthInt"] = last_2_month_df["TradeDateKey"]
		last_2_month_df = last_2_month_df.reset_index()
		last_2_month_df = last_2_month_df[["CustomerIdx", "Last2MonthInt"]]

		# Fifth Merge
		df = pd.merge(df, last_2_month_df, on=['CustomerIdx'], how='left')

		# Number of sell interactions in the last week
		last_week_df_sell = df_train[(df_train["TradeDateKey"] >= max_date -5) & (df_train["BuySell_Sell"] == 1)]
		last_week_df_sell = last_week_df_sell.groupby('CustomerIdx').count()
		last_week_df_sell["LastWeekSell"] = last_week_df_sell["TradeDateKey"]
		last_week_df_sell = last_week_df_sell.reset_index()
		last_week_df_sell = last_week_df_sell[["CustomerIdx", "LastWeekSell"]]

		# Sixth Merge
		df = pd.merge(df, last_week_df_sell, on=['CustomerIdx'], how='left')

		# Number of sell interactions in the last 2 weeks
		last_2_week_df_sell = df_train[(df_train["TradeDateKey"] >= max_date -10) & (df_train["BuySell_Sell"] == 1)]
		last_2_week_df_sell = last_2_week_df_sell.groupby('CustomerIdx').count()
		last_2_week_df_sell["Last2WeekSell"] = last_2_week_df_sell["TradeDateKey"]
		last_2_week_df_sell = last_2_week_df_sell.reset_index()
		last_2_week_df_sell = last_2_week_df_sell[["CustomerIdx", "Last2WeekSell"]]

		# Seventh Merge
		df = pd.merge(df, last_2_week_df_sell, on=['CustomerIdx'], how='left')

		# Number of sell interactions in the last month
		last_month_df_sell = df_train[(df_train["TradeDateKey"] >= max_date -20) & (df_train["BuySell_Sell"] == 1)]
		last_month_df_sell = last_month_df_sell.groupby('CustomerIdx').count()
		last_month_df_sell["LastMonthSell"] = last_month_df_sell["TradeDateKey"]
		last_month_df_sell = last_month_df_sell.reset_index()
		last_month_df_sell = last_month_df_sell[["CustomerIdx", "LastMonthSell"]]

		# Eigth Merge
		df = pd.merge(df, last_month_df_sell, on=['CustomerIdx'], how='left')

		# Number of sell interactions since the beginning
		df_sell = df_train[df_train["BuySell_Sell"] == 1]
		df_sell = df_sell.groupby('CustomerIdx').count()
		df_sell["SellInt"] = df_sell["TradeDateKey"]
		df_sell = df_sell.reset_index()
		df_sell = df_sell[["CustomerIdx", "SellInt"]]

		# Ninth Merge
		df = pd.merge(df, df_sell, on=['CustomerIdx'], how='left')

		# Number of sell interactions in the last week
		last_week_df_buy = df_train[(df_train["TradeDateKey"] >= max_date -5) & (df_train["BuySell_Buy"] == 1)]
		last_week_df_buy = last_week_df_buy.groupby('CustomerIdx').count()
		last_week_df_buy["LastWeekBuy"] = last_week_df_buy["TradeDateKey"]
		last_week_df_buy = last_week_df_buy.reset_index()
		last_week_df_buy = last_week_df_buy[["CustomerIdx", "LastWeekBuy"]]

		# Tenth Merge
		df = pd.merge(df, last_week_df_buy, on=['CustomerIdx'], how='left')

		# Number of sell interactions in the last 2 weeks
		last_2_week_df_buy = df_train[(df_train["TradeDateKey"] >= max_date -10) & (df_train["BuySell_Buy"] == 1)]
		last_2_week_df_buy = last_2_week_df_buy.groupby('CustomerIdx').count()
		last_2_week_df_buy["Last2WeekBuy"] = last_2_week_df_buy["TradeDateKey"]
		last_2_week_df_buy = last_2_week_df_buy.reset_index()
		last_2_week_df_buy = last_2_week_df_buy[["CustomerIdx", "Last2WeekBuy"]]

		# Eleventh Merge
		df = pd.merge(df, last_2_week_df_buy, on=['CustomerIdx'], how='left')

		# Number of sell interactions in the last month
		last_month_df_buy = df_train[(df_train["TradeDateKey"] >= max_date -20) & (df_train["BuySell_Buy"] == 1)]
		last_month_df_buy = last_month_df_buy.groupby('CustomerIdx').count()
		last_month_df_buy["LastMonthBuy"] = last_month_df_buy["TradeDateKey"]
		last_month_df_buy = last_month_df_buy.reset_index()
		last_month_df_buy = last_month_df_buy[["CustomerIdx", "LastMonthBuy"]]

		# Twelveth Merge
		df = pd.merge(df, last_month_df_buy, on=['CustomerIdx'], how='left')

		# Number of sell interactions since the beginning
		df_buy = df_train[df_train["BuySell_Buy"] == 1]
		df_buy = df_buy.groupby('CustomerIdx').count()
		df_buy["BuyInt"] = df_buy["TradeDateKey"]
		df_buy = df_buy.reset_index()
		df_buy = df_buy[["CustomerIdx", "BuyInt"]]

		# Twelveth Merge
		df = pd.merge(df, df_buy, on=['CustomerIdx'], how='left')

		# Fill NAN with Zeros
		df.fillna(0, inplace=True)
		
		df['Features'] = list(zip(df['NumInt'], df['LastInt'], df["LastMonthInt"], 
			df["LastWeekInt"], df["Last2WeekInt"], df["Last2MonthInt"], 
			df["LastWeekSell"], df["Last2WeekSell"], df["LastMonthSell"], df["SellInt"], 
			df["LastWeekBuy"], df["Last2WeekBuy"], df["LastMonthBuy"], df["BuyInt"]))
		df_dict = df.groupby("CustomerIdx")["Features"].apply(list).to_dict()
		
		return df_dict

	def create_cus_bond_features_dict(self, df_train):
		"""
			The method creates the dictionary with all the features related 
			to the pair custonmer - bond.
		"""
		max_date = df_train["TradeDateKey"].max()

		# Number of Interactions during the period
		num_int = df_train.groupby(['CustomerIdx', 'IsinIdx']).count()
		num_int["NumInt"] = num_int["TradeDateKey"]
		num_int = num_int.reset_index(level=['CustomerIdx', 'IsinIdx'])
		num_int = num_int[["CustomerIdx", 'IsinIdx', "NumInt"]]

		# Last interactions
		last_int = df_train.groupby(['CustomerIdx', 'IsinIdx']).max()
		last_int["LastInt"] = last_int["TradeDateKey"].apply(lambda x: max_date - x)
		last_int = last_int.reset_index(level=['CustomerIdx', 'IsinIdx'])
		last_int = last_int[["CustomerIdx", 'IsinIdx', "LastInt"]]

		# First Merge
		df = pd.merge(num_int, last_int, on=['CustomerIdx', "IsinIdx"], how='left')

		# Fill NAN with Zeros
		df.fillna(100, inplace=True)

		# Last month Interactions
		last_month_df = df_train[df_train["TradeDateKey"] >= max_date -20]
		last_month_df = last_month_df.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_month_df["LastMonthInt"] = last_month_df["TradeDateKey"]
		last_month_df = last_month_df.reset_index()
		last_month_df = last_month_df[["CustomerIdx", 'IsinIdx', "LastMonthInt"]]

		# Second Merge
		df = pd.merge(df, last_month_df, on=['CustomerIdx', "IsinIdx"], how='left')

		# Last Week Interactions
		last_week_df = df_train[df_train["TradeDateKey"] >= max_date -5]
		last_week_df = last_week_df.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_week_df["LastWeekInt"] = last_week_df["TradeDateKey"]
		last_week_df = last_week_df.reset_index()
		last_week_df = last_week_df[["CustomerIdx", 'IsinIdx', "LastWeekInt"]]

		# Third Merge
		df = pd.merge(df, last_week_df, on=['CustomerIdx', "IsinIdx"], how='left')

		# Last 2 Week Interactions
		last_2_week_df = df_train[df_train["TradeDateKey"] >= max_date -10]
		last_2_week_df = last_2_week_df.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_2_week_df["Last2WeekInt"] = last_2_week_df["TradeDateKey"]
		last_2_week_df = last_2_week_df.reset_index()
		last_2_week_df = last_2_week_df[["CustomerIdx", 'IsinIdx', "Last2WeekInt"]]

		# Fourth Merge
		df = pd.merge(df, last_2_week_df, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Last 2 Month Interactions
		last_2_month_df = df_train[df_train["TradeDateKey"] >= max_date -40]
		last_2_month_df = last_2_month_df.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_2_month_df["Last2MonthInt"] = last_2_month_df["TradeDateKey"]
		last_2_month_df = last_2_month_df.reset_index()
		last_2_month_df = last_2_month_df[["CustomerIdx", 'IsinIdx', "Last2MonthInt"]]

		# Fifth Merge
		df = pd.merge(df, last_2_month_df, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Number of sell interactions in the last week
		last_week_df_sell = df_train[(df_train["TradeDateKey"] >= max_date -5) & (df_train["BuySell_Sell"] == 1)]
		last_week_df_sell = last_week_df_sell.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_week_df_sell["LastWeekSell"] = last_week_df_sell["TradeDateKey"]
		last_week_df_sell = last_week_df_sell.reset_index()
		last_week_df_sell = last_week_df_sell[["CustomerIdx", "IsinIdx", "LastWeekSell"]]

		# Sixth Merge
		df = pd.merge(df, last_week_df_sell, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Number of sell interactions in the last 2 weeks
		last_2_week_df_sell = df_train[(df_train["TradeDateKey"] >= max_date -10) & (df_train["BuySell_Sell"] == 1)]
		last_2_week_df_sell = last_2_week_df_sell.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_2_week_df_sell["Last2WeekSell"] = last_2_week_df_sell["TradeDateKey"]
		last_2_week_df_sell = last_2_week_df_sell.reset_index()
		last_2_week_df_sell = last_2_week_df_sell[["CustomerIdx", "IsinIdx", "Last2WeekSell"]]

		# Seventh Merge
		df = pd.merge(df, last_2_week_df_sell, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Number of sell interactions in the last month
		last_month_df_sell = df_train[(df_train["TradeDateKey"] >= max_date -20) & (df_train["BuySell_Sell"] == 1)]
		last_month_df_sell = last_month_df_sell.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_month_df_sell["LastMonthSell"] = last_month_df_sell["TradeDateKey"]
		last_month_df_sell = last_month_df_sell.reset_index()
		last_month_df_sell = last_month_df_sell[["CustomerIdx", "IsinIdx", "LastMonthSell"]]

		# Eigth Merge
		df = pd.merge(df, last_month_df_sell, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Number of sell interactions since the beginning
		df_sell = df_train[df_train["BuySell_Sell"] == 1]
		df_sell = df_sell.groupby(['CustomerIdx', 'IsinIdx']).count()
		df_sell["SellInt"] = df_sell["TradeDateKey"]
		df_sell = df_sell.reset_index()
		df_sell = df_sell[["CustomerIdx", "IsinIdx", "SellInt"]]

		# Ninth Merge
		df = pd.merge(df, df_sell, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Number of sell interactions in the last week
		last_week_df_buy = df_train[(df_train["TradeDateKey"] >= max_date -5) & (df_train["BuySell_Buy"] == 1)]
		last_week_df_buy = last_week_df_buy.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_week_df_buy["LastWeekBuy"] = last_week_df_buy["TradeDateKey"]
		last_week_df_buy = last_week_df_buy.reset_index()
		last_week_df_buy = last_week_df_buy[["CustomerIdx", "IsinIdx", "LastWeekBuy"]]

		# Tenth Merge
		df = pd.merge(df, last_week_df_buy, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Number of sell interactions in the last 2 weeks
		last_2_week_df_buy = df_train[(df_train["TradeDateKey"] >= max_date -10) & (df_train["BuySell_Buy"] == 1)]
		last_2_week_df_buy = last_2_week_df_buy.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_2_week_df_buy["Last2WeekBuy"] = last_2_week_df_buy["TradeDateKey"]
		last_2_week_df_buy = last_2_week_df_buy.reset_index()
		last_2_week_df_buy = last_2_week_df_buy[["CustomerIdx", "IsinIdx", "Last2WeekBuy"]]

		# Eleventh Merge
		df = pd.merge(df, last_2_week_df_buy, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Number of sell interactions in the last month
		last_month_df_buy = df_train[(df_train["TradeDateKey"] >= max_date -20) & (df_train["BuySell_Buy"] == 1)]
		last_month_df_buy = last_month_df_buy.groupby(['CustomerIdx', 'IsinIdx']).count()
		last_month_df_buy["LastMonthBuy"] = last_month_df_buy["TradeDateKey"]
		last_month_df_buy = last_month_df_buy.reset_index()
		last_month_df_buy = last_month_df_buy[["CustomerIdx", "IsinIdx", "LastMonthBuy"]]

		# Twelveth Merge
		df = pd.merge(df, last_month_df_buy, on=['CustomerIdx', 'IsinIdx'], how='left')

		# Number of sell interactions since the beginning
		df_buy = df_train[df_train["BuySell_Buy"] == 1]
		df_buy = df_buy.groupby(['CustomerIdx', 'IsinIdx']).count()
		df_buy["BuyInt"] = df_buy["TradeDateKey"]
		df_buy = df_buy.reset_index()
		df_buy = df_buy[["CustomerIdx", "IsinIdx", "BuyInt"]]

		# Twelveth Merge
		df = pd.merge(df, df_buy, on=['CustomerIdx', 'IsinIdx'], how='left')

		print("NEW LAST INT TYPE IMPLE.")
		print(df.head())
		print(df.describe())
		# exit()

		# Type of the last interaction
		df_int_type = df_train.sort_values("TradeDateKey", ascending=True)
		df_int_type = df_int_type.drop_duplicates(["CustomerIdx", "IsinIdx"], keep="last")
		df_int_type["TypeLastInt"] = df_int_type["BuySell_Buy"]
		df_int_type = df_int_type[["CustomerIdx", "IsinIdx", "TypeLastInt"]]

		# Twelveth Merge
		df = pd.merge(df, df_int_type, on=['CustomerIdx', 'IsinIdx'], how='left')
		print(df.groupby(["CustomerIdx", "IsinIdx"]).count().describe())

		# Fill NAN with Zeros
		df.fillna(0, inplace=True)
		
		df['Features'] = list(zip(df['NumInt'], df['LastInt'], df["LastMonthInt"], 
			df["LastWeekInt"], df["Last2WeekInt"], df["Last2MonthInt"], 
			df["LastWeekSell"], df["Last2WeekSell"], df["LastMonthSell"], 
			df["SellInt"], df["LastWeekBuy"], df["Last2WeekBuy"], 
			df["LastMonthBuy"], df["BuyInt"], df["TypeLastInt"]))
		df_dict = df.groupby(['CustomerIdx', "IsinIdx"])["Features"].apply(list).to_dict()
		
		return df_dict

