import dsg.util as util
from dsg.test_train_data_generation.fake_data_generator import DataGenerator

class ClassifierPreprocessor(object):
	def __init__(self, from_date, test_date, val_date):
		super(ClassifierPreprocessor, self).__init__()
		self.from_date = from_date
		self.test_date = test_date
		self.val_date = val_date

	def fit(self, df):
		return

	def test_transform(self, df):
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

	def filter_customers(self, df, min_value):
		df_freq = df.groupby("CustomerIdx").count()["TradeDateKey"].reset_index()
		df_active = df_freq[df_freq["TradeDateKey"] >= min_value]
		active_list = df_active["CustomerIdx"].values
		return active_list

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
		train = data_generator.generate_train_dataset(df, till_date=min(self.test_date, self.val_date))
		test = data_generator.generate_test_dataset(df, self.test_date)
		validation = data_generator.generate_test_dataset(df, self.val_date)

		# Entire Train
		data = data_generator.generate_train_dataset(df, till_date=min(20180424, 20180424))

		return train, test, validation, data

	def train_test_validation_split(self, features):
		"""
			The method splits the data into train/test/validation
		"""
		raise NotImplementedError

		