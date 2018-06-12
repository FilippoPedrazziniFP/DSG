import pandas as pd
import dsg.util as util
from dsg.data_generation.data_generator import FakeGeneratorFilo
import pickle

class ClassifierPreprocessor(object):
	def __init__(self):
		super(ClassifierPreprocessor, self).__init__()

	def fit(self, df):
		return

	def transform(self, df):
		return

	def fit_transform(self):
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

		try:
			X_train = pickle.load(open("X_train.pkl", "rb"))
			print(X_train.shape)
			
			y_train = pickle.load(open("y_train.pkl", "rb"))
			print(y_train.shape)
			
			X_test = pickle.load(open("X_test.pkl", "rb"))
			print(X_test.shape)
			
			y_test = pickle.load(open("y_test.pkl", "rb"))
			print(y_test.shape)
			
			X_val = pickle.load(open("X_val.pkl", "rb"))
			print(X_val.shape)
			
			y_val = pickle.load(open("y_val.pkl", "rb"))
			print(y_val.shape)
			
			X = pickle.load(open("X.pkl", "rb"))
			print(X.shape)
			
			y = pickle.load(open("y.pkl", "rb"))
			print(y.shape)

			X_challenge = pickle.load(open("X_challenge.pkl", "rb"))
			print(X_challenge.shape)
		except:
			# Generate Train Test and Validation
			X_train, y_train, X_test, y_test, X_val, y_val, X, y, \
				X_challenge = data_generator.generate_train_test_val()

		# Load Submission file
		submission = pd.read_csv(util.SAMPLE_SUBMISSION)

		return X_train, y_train, X_test, y_test, X_val, y_val, \
			X, y, X_challenge, submission 

	def fit_transform_claudio(self, df):
		return

		