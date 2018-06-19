import dsg.util as util
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

class SequencePreprocessor(object):
	def __init__(self):
		super(SequencePreprocessor, self).__init__()

	def define_scaler(self, X):
		matrix = np.concatenate(X, axis=0)
		self.scaler = StandardScaler()
		self.scaler.fit(matrix)
		return

	def scale_data(self, X):
		X_std = []
		for sample in X:
			sample = self.scaler.transform(sample)
			X_std.append(sample)
		X_std = np.asarray(X_std)
		return X_std

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

		X_train = pickle.load(open("./dsg/recurrent/features_train.pkl", "rb"))
		self.define_scaler(X_train)
		X_train = self.scale_data(X_train)
		print(X_train.shape)
		
		y_train = pickle.load(open("./dsg/recurrent/labels_train.pkl", "rb"))
		print(y_train.shape)
		
		X_test = pickle.load(open("./dsg/recurrent/features_test.pkl", "rb"))
		X_test = self.scale_data(X_test)
		print(X_test.shape)
		
		y_test = pickle.load(open("./dsg/recurrent/labels_test.pkl", "rb"))
		print(y_test.shape)
		
		X_val = pickle.load(open("./dsg/recurrent/features_val.pkl", "rb"))
		X_val = self.scale_data(X_val)
		print(X_val.shape)
		
		y_val = pickle.load(open("./dsg/recurrent/labels_val.pkl", "rb"))
		print(y_val.shape)

		X = []
		y = []
		for i in range(665, 680, 5):
			print(i)
			features = pickle.load(open('./dsg/recurrent/entire_data/features_' + str(i) + '_entire.pkl', 'rb'))
			labels = pickle.load(open('./dsg/recurrent/entire_data/labels_' + str(i) + '_entire.pkl', 'rb'))
			X.append(features)
			y.append(labels)
		X = np.concatenate(X, axis=0)
		X = self.scale_data(X)
		y = np.concatenate(y, axis=0)
		print(X.shape)
		print(y.shape)

		X_challenge = pickle.load(open("./dsg/recurrent/features_challenge.pkl", "rb"))
		X_challenge = self.scale_data(X_challenge)
		print(X_challenge.shape)

		# Load Submission file
		submission = pd.read_csv(util.SAMPLE_SUBMISSION)

		return X_train, y_train, X_test, y_test, X_val, y_val, \
			X, y, X_challenge, submission 

		