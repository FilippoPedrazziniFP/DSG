import scipy.sparse as sps
import numpy as np
import dsg.util as util
from dsg.recommenders.data_generator import DataGenerator

class URMPreprocessing(object):
	def __init__(self, from_date, test_date, 
			val_date, train_date):
		super(URMPreprocessing, self).__init__()
		self.val_date = val_date
		self.from_date = from_date
		self.train_date = train_date
		self.test_date = test_date

	def fit(self, df):
		return

	def test_transform(self, df):
		df = df.drop(["PredictionIdx", "DateKey", "BuySell"], axis=1)
		X = df.values
		return X

	def transform(self, df):
		return

	def filter_data(self, df, date):
		df = df[df["TradeDateKey"] >= date]
		return df

	def fit_transform(self, df):
		"""
			The method drops the useless columns from
			the DataFrame and splits the data into train 
			test set based on the data

			@args
				df : DataFrame

			@return
				X_train, y_train, X_test, y_test : numpy array
		"""
		# Delete Holding Values
		df = df[df["TradeStatus"] != "Holding"]

		# Drop Useless Columns
		df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)
		df = df.sort_values("TradeDateKey", ascending=True)

		# Filter Data 
		df = self.filter_data(df, self.from_date)

		# Train, test, val split
		data_generator = DataGenerator()
		train = data_generator.generate_train_set_explicit(df, self.val_date)
		test = data_generator.generate_test_set(df, self.test_date)
		val = data_generator.generate_test_set(df, self.val_date, self.test_date)

		# Entire Train
		data = data_generator.generate_train_set_explicit(df)		
		return train, test, val, data

	def df_to_csr(self, df, is_binary=False, user_key='CustomerIdx', item_key='IsinIdx', rating_key='CustomerInterest'):
	    df.set_index([user_key, item_key], inplace=True)
	    mat = sps.csr_matrix((df[rating_key], (df.index.labels[0], df.index.labels[1])))
	    return mat