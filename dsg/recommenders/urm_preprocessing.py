import scipy.sparse as sps
import numpy as np
import dsg.util as util
from dsg.data_generation.data_generator import FakeGeneratorFilo

class URMPreprocessing(object):
	def __init__(self, test_date, 
			val_date, train_date):
		super(URMPreprocessing, self).__init__()
		self.val_date = val_date
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

	def filter_data(self, df, date=20180101):
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
		
		# Train, test, val split
		data_generator = FakeGeneratorFilo()
		train = data_generator.generate_train_set_svd(df, self.val_date)

		# Generate Test Set
		test = data_generator.generate_test_set(
			df=df, 
			from_date=self.test_date
			)
		val = data_generator.generate_test_set(
			df=df, 
			from_date=self.val_date,
			to_date=self.test_date
			)

		# Entire Train
		data = data_generator.generate_train_set_svd(df)		
		
		return train, test, val, data

	def df_to_csr(self, df, is_binary=False, user_key='CustomerIdx', item_key='IsinIdx', rating_key='CustomerInterest'):
	    df.set_index([user_key, item_key], inplace=True)
	    mat = sps.csr_matrix((df[rating_key], (df.index.labels[0], df.index.labels[1])))
	    return mat