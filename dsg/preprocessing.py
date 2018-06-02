import pandas as pd

class DataPreprocessor(object):
	def __init__(self):
		super(DataPreprocessor, self).__init__()

	def fit(self, df):
		raise NotImplementedError

	def fit_transform(self, df):
		raise NotImplementedError

	def transform(self, df):
		raise NotImplementedError

	def save_df_to_csv(self, df, file_path):
		"""
			The method saves the DF in a CSV.
			@args
				df : Data Frame
				file_path : string -> path where
					to save the file.
			
		"""
		df.to_csv(file_path, index=False)
		return

	def merge_df(self, df_trade, df_bonds, df_customer):
		"""
			The method merges the statid data and
			clean the generated one dropping the temporal 
			information.

			@args
				df : DataFrame

			@return
				df : DataFrame
		"""
		# Merge the Data Frames
		df = pd.merge(df_trade, df_customer, on=["CustomerIdx"])
		df = pd.merge(df, df_bonds, on=["IsinIdx"])
		return df

		