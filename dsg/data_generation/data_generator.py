import pandas as pd
from dsg.data_generation.fake_train_test_claudio import FakeDataGeneratorClaudio

class FakeGeneratorFilo(object):
	def __init__(self):
		super(FakeGeneratorFilo, self).__init__()

	def generate_train_set_linear(self, df, from_date, to_date=None):
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
		
		if to_date is None:
			labels = df[df["TradeDateKey"] >= from_date]
		else:
			labels = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]

		# Group By Customer and Bond to have the sum of interests during the period.
		labels = labels.groupby(["CustomerIdx", "IsinIdx"]).count()["CustomerInterest"].reset_index(level=['CustomerIdx', 'IsinIdx'])		
		
		# The Train is composed of all the values before the Date
		features = df[df["TradeDateKey"] <= from_date]		
		
		return features, labels

	def generate_train_set_svd(self, df, to_date=None, max_rating=5):
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
				data : DataFrame
			
		"""
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
			from_date_label=20171008, from_date_features=20180101):
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
		negative_samples = negative_samples.groupby(["CustomerIdx", "IsinIdx"]).count()
		negative_samples = negative_samples[negative_samples["CustomerInterest"] <= 2]
		negative_samples["CustomerInterest"] = negative_samples["CustomerInterest"].apply(lambda x: 0 if x > 0 else x)
		negative_samples = negative_samples.reset_index(level=['CustomerIdx', 'IsinIdx'])

		# Concatanate Negative and Positive Samples
		labels = pd.concat([positive_samples, negative_samples])
		labels = labels.drop(["TradeDateKey", "BuySell"], axis=1)
		
		# The Train is composed of all the values before the Date
		features = df[df["TradeDateKey"] >= from_date_features]
		features = features[features["TradeDateKey"] <= from_date]

		return features, labels


	def generate_test_set(self, df, from_date, to_date=None, from_date_label=20171008):

		from_date_label = 20170420
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
		if to_date is None:
			positive_samples = df[df["TradeDateKey"] >= from_date]
		else:
			positive_samples = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]
		
		# Negative Samples
		negative_samples = df[df["TradeDateKey"] >= from_date_label]
		negative_samples = negative_samples.groupby(["CustomerIdx", "IsinIdx"]).count()
		negative_samples = negative_samples[negative_samples["CustomerInterest"] <= 2]
		negative_samples["CustomerInterest"] = negative_samples["CustomerInterest"].apply(lambda x: 0 if x > 0 else x)
		negative_samples = negative_samples.reset_index(level=['CustomerIdx', 'IsinIdx'])

		# Concatanate Negative and Positive Samples
		test_set = pd.concat([positive_samples, negative_samples])
		test_set = test_set.drop(["TradeDateKey", "BuySell"], axis=1)
		test_set = test_set[['CustomerIdx', 'IsinIdx', 'CustomerInterest']]
		
		return test_set
		