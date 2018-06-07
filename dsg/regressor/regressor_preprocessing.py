import dsg.util as util
from dsg.regressor.data_generator import DataGenerator

class RegressorPreprocessor(object):
	def __init__(self, from_date, test_date, 
			val_date, train_date):
		super(RegressorPreprocessor, self).__init__()
		self.from_date = from_date
		self.test_date = test_date
		self.val_date = val_date
		self.train_date = train_date

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
		X_train, y_train = data_generator.generate_train_set(df, self.train_date, self.val_date)
		
		# Generate Test Set
		# y_test = data_generator.generate_test_set_claudio(df)
		# print(y_test.head())

		# Entire Train
		X, y = data_generator.generate_train_set(df, self.test_date)

		return X_train, y_train, X, y

		