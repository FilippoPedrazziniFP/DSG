import dsg.util as util
from dsg.data_generation.data_generator import FakeGeneratorFilo

class ClassifierPreprocessor(object):
	def __init__(self, test_date, 
			val_date, train_date):
		super(ClassifierPreprocessor, self).__init__()
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
		
		# Train Data
		X_train, y_train = data_generator.generate_train_set_classification(
			df=df, 
			from_date=self.train_date, 
			to_date=self.val_date
			)
		
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
		X, y = data_generator.generate_train_set_classification(
			df=df,
			from_date=self.test_date)

		return X_train, y_train, test, val, X, y

		