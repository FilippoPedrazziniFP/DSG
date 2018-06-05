import dsg.util as util
from dsg.classifier.data_generator_class import DataGenerator

class ClassifierPreprocessor(object):
	def __init__(self, from_date, test_date, 
			val_date, train_date):
		super(ClassifierPreprocessor, self).__init__()
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

	def create_frequency_feature(self, df):
		max_frequency = df['Frequency'].max()
		df['Frequency'] = df.groupby('CustomerIdx')['CustomerIdx'].transform('count')
		df['Probability'] = df['Frequency'].apply(lambda x: x/max_frequency)
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
		df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)
		df = df.sort_values("TradeDateKey", ascending=True)

		# Filter Data 
		df = self.filter_data(df, self.from_date)

		# Train, test, val split
		data_generator = DataGenerator()
		X_train, y_train = data_generator.generate_train_set_regression(df, self.train_date, self.val_date)
		y_test = data_generator.generate_test_set(df, self.test_date)
		y_val = data_generator.generate_test_set(df, self.val_date, self.test_date)

		# Entire Train
		X, y = data_generator.generate_train_set_regression(df, self.test_date)

		return X_train, y_train, y_test, y_val, X, y

	def train_test_validation_split(self, features):
		raise NotImplementedError

		