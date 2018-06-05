class DataGenerator(object):
	def __init__(self):
		super(DataGenerator, self).__init__()

	def generate_train_set_regression(self, df, from_date, to_date=None):
		if to_date is None:
			label = df[df["TradeDateKey"] >= from_date]
		else:
			label = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]

		# Group By Customer and Bond to have the sum of interests during the period.
		label = label.groupby(["CustomerIdx", "IsinIdx"]).count()["CustomerInterest"].reset_index(level=['CustomerIdx', 'IsinIdx'])		
		features = df[df["TradeDateKey"] <= from_date]		
		return features, label

	def generate_train_set_classification(self, df, from_date, to_date=None):
		if to_date is None:
			label = df[df["TradeDateKey"] >= from_date]
		else:
			label = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]	
		features = df[df["TradeDateKey"] <= from_date]	
		label = label.drop(["TradeDateKey", "BuySell"], axis=1)
		return features, label

	def generate_test_set(self, df, from_date, to_date=None):
		"""
			The method returns the data from a date to a date 
			for testing purposes.

			@args
				df : DF
				from_date : int -> corresponding to a date
				to_date : int -> corresponding to a date

			@return
				label : DF
		"""
		if to_date is None:
			label = df[df["TradeDateKey"] >= from_date]
		else:
			label = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]
		
		return label
		