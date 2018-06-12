import argparse
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from itertools import islice
import pickle
import operator
from itertools import chain
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, LinearRegression
from dsg.data_loader import DataLoader
from sklearn.preprocessing import StandardScaler
from dsg.visualizer import Explorer
from catboost import CatBoostClassifier
from dsg.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True, 
	help='If True train and validate the model locally.')
parser.add_argument('--sub', type=bool, default=False, 
	help='If True train the model on the entire data and creates the submission.')
""" General Parameters """
args = parser.parse_args()

SUBMISSION = "./data/submission.csv"

class Classifier(object):
	def __init__(self):
		super(Classifier, self).__init__()

	def fit(self, X_train, y_train):
		# Train Classifier
		self.classifier = self.train_classifier(X_train, y_train)
		return

	def train_classifier(self, X_train, y_train):
		"""
			Simple classifier to put a weight 
			to the frequency feature.
		"""
		self.scaler = StandardScaler()
		self.scaler.fit(X_train)
		X_train = self.scaler.transform(X_train)

		# Plot Distribution of Labels
		# Explorer.plot_array(y_train)
		print(y_train.mean())
		print(y_train.std())
		print(y_train.max())

		# Fit the model
		model = CatBoostClassifier(verbose=False)
		model.fit(X_train, y_train)
		return model

	def predict(self, X):
		X = self.scaler.transform(X)
		predictions = self.classifier.predict_proba(X)[:,1]
		print(predictions.max())
		print(predictions[0:10])
		return predictions

	def evaluate(self, X_test, y_test):
		y_pred = self.predict(X_test)
		score = roc_auc_score(y_test, y_pred)
		return score

	def load_data(self):
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
		
		# Load Submission file
		submission = pd.read_csv(SUBMISSION)

		return X_train, y_train, X_test, y_test, X_val, y_val, \
			X, y, X_challenge, submission 


def main():

	# Fixing the seed
	np.random.seed(0)
	tf.set_random_seed(0)
	
	# Fit and Evaluate the model
	model = Classifier()
	X_train, y_train, X_test, y_test, X_val, y_val, X, y, X_challenge, submission \
		= model.load_data()

	if args.train == True:

		t = time.clock()
		
		print(X_train.shape)
		print(y_train.shape)
		model.fit(X_train, y_train)
		
		print("TRAINED FINISHED, STARTING TEST..")

		# Evaluate the model
		score = model.evaluate(X_test, y_test)
		print("TEST SCORE: ", score)
		
		print("TIME TO FIT AND EVALUATE THE MODEL: ", time.clock() - t)

	if args.sub == True:

		t = time.clock()

		# Fit on the entire data 
		print(X.shape)
		print(y.shape)
		model.fit(X, y)
				
		# Create the submission file
		preds = model.predict(X_challenge)
		submission["CustomerInterest"] = preds
		submission.to_csv(util.SUBMISSION, index=False)

		print("TIME TO FIT THE ENTIRE DATA and CREATE SUBMISSION: ", time.clock() - t)

	return

main()