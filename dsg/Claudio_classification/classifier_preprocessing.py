import dsg.util as util
from dsg.test_train_data_generation.fake_data_generator import DataGenerator

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

		return df

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
		data_generator = DataGenerator()
		
		# Train Data
		train = data_generator.generate_train_dataset(df)

		# Generate Test Set
		test = data_generator.generate_test_dataset(df)

		val = data_generator.generate_validation_dataset(df)

		submission = data_generator.generate_submission_dataset(df)


		return train, test, val, submission

	def fit_transform_claudio(self, df):
		return

		