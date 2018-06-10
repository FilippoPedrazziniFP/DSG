import dsg.util as util
import numpy as np

class SequencePreprocessor(object):
	def __init__(self, from_date, test_samples, 
		val_samples, seq_length=30):
		super(SequencePreprocessor, self).__init__()
		self.from_date = from_date
		self.test_samples = test_samples
		self.val_samples = val_samples
		self.seq_length = seq_length

	def fit(self, df):
		return

	def test_transform(self, df):
		df = df.drop(["PredictionIdx", "DateKey", "BuySell"], axis=1)
		X = df.values
		return X

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

		# Create Data
		data = df.groupby("TradeDateKey").count()["CustomerInterest"].values
		last_sample = data[-self.seq_length:]

		print(len(data))

		# Train test val split
		train, test, val = self.train_test_val_split(data)

		print(len(train))
		# Generate Sequences
		X_train, y_train = self.generate_sequences(train)
		X_test, y_test = self.generate_sequences(test)
		X_val, y_val = self.generate_sequences(val)

		return X_train, y_train, X_test, y_test, X_val, y_val, last_sample

	def train_test_val_split(self, data):
		test = data[-self.test_samples:]
		val = data[-(self.test_samples+self.val_samples):-self.test_samples]
		train = data[:-(self.test_samples+self.val_samples)]
		return train, test, val

	def generate_sequences(self, data):
		features_seq = []
		labels_seq = []
		for i in range(0, len(data)-self.seq_length-7):
			features_seq.append(data[i:i+self.seq_length])
			label = data[i+self.seq_length+7].sum()
			labels_seq.append(label)

		features = np.asarray(features_seq)
		labels = np.asarray(labels_seq)
		return features, labels

		
		