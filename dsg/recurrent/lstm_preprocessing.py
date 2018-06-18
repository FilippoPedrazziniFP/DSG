import dsg.util as util
import numpy as np
import pandas as pd
import pickle

class SequencePreprocessor(object):
	def __init__(self):
		super(SequencePreprocessor, self).__init__()

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
		print(X_train.shape)
		
		y_train = pickle.load(open("./dsg/recurrent/labels_train.pkl", "rb"))
		print(y_train.shape)
		
		X_test = pickle.load(open("./dsg/recurrent/features_test.pkl", "rb"))
		print(X_test.shape)
		
		y_test = pickle.load(open("./dsg/recurrent/labels_test.pkl", "rb"))
		print(y_test.shape)
		
		X_val = pickle.load(open("./dsg/recurrent/features_val.pkl", "rb"))
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
		y = np.concatenate(y, axis=0)
		print(X.shape)
		print(y.shape)

		X_challenge = pickle.load(open("./dsg/recurrent/features_challenge.pkl", "rb"))
		print(X_challenge.shape)

		# Load Submission file
		submission = pd.read_csv(util.SAMPLE_SUBMISSION)

		return X_train, y_train, X_test, y_test, X_val, y_val, \
			X, y, X_challenge, submission 

		