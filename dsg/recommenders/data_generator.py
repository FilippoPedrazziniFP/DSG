
class DataGenerator(object):
	def __init__(self, max_rating=1):
		super(DataGenerator, self).__init__()
		self.max_rating = max_rating

	def generate_train_set_explicit(self, df, to_date=None):
		if to_date is None:
			data = df
		else:
			data = df[df["TradeDateKey"] <= to_date]
		# Group By Customer and Bond to have the sum of interests during the period.
		data = data.groupby(["CustomerIdx", "IsinIdx"]).count()["CustomerInterest"].reset_index(level=['CustomerIdx', 'IsinIdx'])
		# Clip to a maximum of 5 Interactions
		data["CustomerInterest"] = data["CustomerInterest"].apply(lambda x: self.max_rating if x > self.max_rating else x)				
		return data

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
			data = df[df["TradeDateKey"] >= from_date]
		else:
			data = df[(df["TradeDateKey"] >= from_date) & (df["TradeDateKey"] < to_date)]

		data = data.drop(["TradeDateKey", "BuySell"], axis=1)
		return data
		