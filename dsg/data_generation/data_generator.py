import pandas as pd
from dsg.data_generation.fake_train_test_claudio import DataGeneratorClaudio

"""
	TODO: CLEAN CODE -- A LOT OF DUPLICATED CODE

"""

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

	def generate_test_val_set_claudio(self, df):
		"""
			The method calls Claudio class to generate Test and train.
		"""
		generator = DataGeneratorClaudio()
		test = generator.generate_test_dataset(df)
		val = generator.generate_validation_dataset(df)

		test = test.drop(["DateKey", "BuySell"], axis=1)
		val = val.drop(["DateKey", "BuySell"], axis=1)
		return test, val


	def generate_test_set(self, df, from_date, to_date=None, from_date_label=20160101, clip=2):
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

		# Generate Random New Negative Samples
		"""

			To implement - two ways:
				- random from the list of customer and bonds that never interacted
				with each other.
				- use sell and buy to generate them. if CUS BOND SELL = 1 then CUS BOND BUY = 0
		"""

		# Concatanate Negative and Positive Samples
		test_set = pd.concat([positive_samples, negative_samples])
		test_set = test_set.drop(["TradeDateKey"], axis=1)

		# Unique Values
		test_set = test_set.groupby(['CustomerIdx', 'IsinIdx', "BuySell"]).sum()
		test_set = test_set.reset_index(level=['CustomerIdx', 'IsinIdx', "BuySell"])
		test_set["CustomerInterest"] = test_set["CustomerInterest"].apply(lambda x: 1 if x > 1 else x)

		# One Hot Encoding for Sell and Buy
		test_set = pd.get_dummies(test_set, columns=['BuySell'])
		test_set = test_set[['CustomerIdx', 'IsinIdx', "BuySell_Buy", "BuySell_Sell", 'CustomerInterest']]		
		return test_set

