import dsg.util as util

class BaselinePreprocessor(object):
	def __init__(self, from_date, train_samples, 
			test_samples, val_samples):
		super(BaselinePreprocessor, self).__init__()
		self.from_date = from_date
		self.train_samples = train_samples
		self.test_samples = test_samples
		self.val_samples = val_samples

	def fit(self, df):
		return

	def test_transformation(self, df):

		print(df.head())

		# Features Labels Split
		X, y = self.features_labels_split(df)

		print(X.shape)
		print(X[0:10])

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

	def features_labels_split(self, df):
		"""
			The method splits the data into Features and Labels
			returning a numpy array
			@args
				df : Data Frame
			@return
				features : numpy array
				labels : numpy array
		"""
		labels = df[util.LABEL_TAG].values
		df = df.drop([util.LABEL_TAG], axis=1)
		features = df.values
		return features, labels

	def filter_customers(self, df, min_value):
		df_freq = df.groupby("CustomerIdx").count()["TradeDateKey"].reset_index()
		df_active = df_freq[df_freq["TradeDateKey"] >= min_value]
		active_list = df_active["CustomerIdx"].values
		return active_list

	def fit_transform(self, df):
		"""
			The method drops the useless columns from
			the DataFrame and splits the data into train 
			test set based on the data.

			@args
				df : DataFrame

			@return
				X_train, y_train, X_test, y_test : numpy array
		"""
		df = df.drop(["TradeStatus", "NotionalEUR", "Price"], axis=1)
		df = df.sort_values("TradeDateKey", ascending=True)

		# Filter Data 
		df = self.filter_data(df, self.from_date)

		# Features Labels Split
		X, y = self.features_labels_split(df)

		print(X.shape)
		print(X[0:10])

		# Train, test, val split
		X_train, y_train, X_test, y_test, X_val, y_val = self.train_test_validation_split(X, y)

		return X_train, y_train, X_test, y_test, X_val, y_val, X, y

	def train_test_validation_split(self, features, labels):

		X_test = features[-self.test_samples:]
		y_test = labels[-self.test_samples:]

		X_val = features[-(self.test_samples+self.val_samples):-self.test_samples]
		y_val = labels[-(self.test_samples+self.val_samples):-self.test_samples]

		X_train = features[-(self.test_samples+self.val_samples+self.train_samples):-(self.test_samples+self.val_samples)]
		y_train = labels[-(self.test_samples+self.val_samples+self.train_samples):-(self.test_samples+self.val_samples)]

		return X_train, y_train, X_test, y_test, X_val, y_val
