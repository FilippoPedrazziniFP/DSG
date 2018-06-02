from simple_ml.data_preprocessing.preprocessing import DataPreprocessor

class TrainDataPreprocessor(DataPreprocessor):
	def __init__(self):
		super(TrainDataPreprocessor, self).__init__()

	def fit(self, df):
		return

	def transform(self, df):
		return

	def fit_transform(self, df):
		"""
			The method drops the useless columns from
			the DataFrame and resort it.

			@args
				df : DataFrame

			@return
				df : DataFrame
		"""
		df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)
		df = df.sort_values("TradeDateKey", ascending=True)
		return df
