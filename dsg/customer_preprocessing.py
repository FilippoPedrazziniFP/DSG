from simple_ml.data_preprocessing.preprocessing import DataPreprocessor

class CustomerDataPreprocessor(DataPreprocessor):
	def __init__(self):
		super(CustomerDataPreprocessor, self).__init__()

	def fit(self, df):
		raise NotImplementedError

	def transform(self, df):
		raise NotImplementedError

	def fit_transform(self, df):
		"""
			The method drops two categorical variables
			that shouldn't be relevant with the problem.

			@args
				df : DataFrame

			@return
				df : DataFrame
		"""
		df = df.drop(["Subsector", "Country"], axis=1)
		return df
