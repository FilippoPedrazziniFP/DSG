from simple_ml.data_preprocessing.preprocessing import DataPreprocessor

class BondsDataPreprocessor(DataPreprocessor):
	def __init__(self):
		super(BondsDataPreprocessor, self).__init__()

	def fit(self, df):
		raise NotImplementedError

	def transform(self, df):
		raise NotImplementedError

	def fit_transform(self, df):
		"""
			The method drops the Ticker Index feature
			which represents the index given to the stock 
			and the Industry Subgroup due to the high cardinality.

			@args
				df : DataFrame

			@return
				df : DataFrame
		"""
		df = df.drop(["TickerIdx", "IndustrySubgroup"], axis=1)
		return df
